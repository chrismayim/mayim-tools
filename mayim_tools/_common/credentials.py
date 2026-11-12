"""
Credentials from the QGIS authentication database.

Tools that log in to a data service (Earthdata for IMERG and MERRA2, the
Copernicus CDS for ERA5) take a QGIS authentication configuration
(QgsProcessingParameterAuthConfig) instead of a typed username, password
or key. QGIS stores the configuration encrypted under the user's master
password, so the secret never appears in the processing history, the
log, a model file or a project file.

Set up once in QGIS: Settings > Options > Authentication > Add (+),
method "Basic authentication":

- Earthdata: Username = Earthdata username, Password = Earthdata
  password.
- CDS: Username = anything (for example "cds"), Password = the
  Personal Access Token from https://cds.climate.copernicus.eu/profile.

Without a configuration the tools fall back to the service's own files
(~/.netrc for Earthdata, ~/.cdsapirc for the CDS), as before.
"""

from __future__ import annotations

import contextlib
import os

SETUP_HELP = (
    "Set up the login once in QGIS: Settings > Options > Authentication > "
    "Add (+), method 'Basic authentication', then pick it in this tool. "
    "QGIS keeps it encrypted under your master password, so it is never "
    "written to the processing history, the log or a project file."
)


def basic_auth_values(authcfg_id: str) -> tuple[str, str]:
    """(username, password) of a QGIS Basic authentication configuration.

    Raises ValueError when the configuration cannot be loaded (wrong id,
    master password not entered) or has no password.
    """
    from qgis.core import QgsApplication, QgsAuthMethodConfig

    manager = QgsApplication.authManager()
    config = QgsAuthMethodConfig()
    loaded = manager.loadAuthenticationConfig(authcfg_id, config, True)
    if not loaded or not config.id():
        raise ValueError(
            f"The QGIS authentication configuration '{authcfg_id}' could not "
            "be loaded. Check that it exists and that the QGIS master password "
            "has been entered."
        )
    username = config.config("username", "")
    password = config.config("password", "")
    if not password:
        raise ValueError(
            f"The QGIS authentication configuration '{config.name()}' has no "
            "password. Use the 'Basic authentication' method and fill in the "
            "password (or the API key) field."
        )
    return username, password


@contextlib.contextmanager
def temporary_environment(**values):
    """Set environment variables for the duration of the block only, then
    restore the previous values. Nothing is written to disk."""
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def earthdata_login():
    """Log in to NASA Earthdata with earthaccess and return the Auth object.

    Credentials in the EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment
    variables (set for the run from the tool's QGIS authentication
    configuration) are used first; otherwise ~/.netrc. Raises RuntimeError
    when neither works.
    """
    try:
        import earthaccess
    except ImportError as missing_dependency:
        raise RuntimeError(
            "The 'earthaccess' package is required. Install it in the OSGeo4W "
            "Shell: python -m pip install earthaccess"
        ) from missing_dependency

    # Older earthaccess versions take the first token on the account even
    # when it has expired (Earthdata tokens live 60 days), and in a long
    # QGIS session earthaccess keeps the token it got at the first login.
    # An expired token gives CMR "401 ... has expired" and Giovanni "500
    # Internal Server Error". So: revoke expired tokens first, log in, and
    # make sure the session holds a valid one.
    creds = earthdata_credentials()
    fresh = None
    if creds is not None:
        try:
            fresh = valid_edl_token(*creds)
        except Exception:  # noqa: BLE001 - earthaccess login still runs
            fresh = None

    auth = None
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        auth = earthaccess.login(strategy="environment")
    if auth is None or not auth.authenticated:
        auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError(
            "Could not log in to NASA Earthdata. Choose an Earthdata login in "
            "the tool's 'Earthdata login' parameter (" + SETUP_HELP + ") Also "
            "confirm that 'NASA GESDISC DATA ARCHIVE' is an authorized "
            "application on your Earthdata profile: "
            "https://urs.earthdata.nasa.gov/profile"
        )
    current = getattr(auth, "token", None) or {}
    if fresh is not None and (
        not isinstance(current, dict)
        or token_expired(current)
        or current.get("access_token") != fresh.get("access_token")
    ):
        auth.token = fresh
    elif isinstance(current, dict) and token_expired(current):
        raise RuntimeError(
            "Your Earthdata token has expired and could not be renewed. Delete "
            "the expired token at https://urs.earthdata.nasa.gov (Generate Token "
            "page), restart QGIS and run the tool again."
        )
    return auth


EDL_URL = "https://urs.earthdata.nasa.gov"


def earthdata_credentials() -> tuple[str, str] | None:
    """(username, password) for Earthdata: the environment variables set
    from the tool's QGIS login, else the urs.earthdata.nasa.gov entry of
    ~/.netrc or ~/_netrc. None when neither has them."""
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if user and password:
        return user, password
    import netrc
    from pathlib import Path

    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        if not path.exists():
            continue
        try:
            entry = netrc.netrc(str(path)).authenticators("urs.earthdata.nasa.gov")
        except (netrc.NetrcParseError, OSError):
            continue
        if entry and entry[0] and entry[2]:
            return entry[0], entry[2]
    return None


def _expiry(token: dict):
    """Expiry date of an Earthdata token dict, or None if not given."""
    from datetime import datetime

    # EDL gives "MM/DD/YYYY"; also accept an ISO date or date-time
    text = str(token.get("expiration_date") or "").strip()[:10]
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def token_expired(token: dict, today=None) -> bool:
    """True when the token's expiration date is today or earlier."""
    from datetime import date

    exp = _expiry(token)
    return exp is not None and exp <= (today or date.today())


def valid_edl_token(username: str, password: str, session=None, today=None) -> dict:
    """A valid Earthdata Login user token: expired tokens on the account
    are revoked, the first valid one is returned, and a new one is made
    when none is left (an account holds at most two)."""
    import requests

    s = session or requests.Session()
    s.auth = (username, password)
    r = s.get(f"{EDL_URL}/api/users/tokens", timeout=30)
    r.raise_for_status()
    valid = []
    for tok in r.json() or []:
        if token_expired(tok, today):
            s.post(
                f"{EDL_URL}/api/users/revoke_token",
                params={"token": tok.get("access_token", "")},
                timeout=30,
            )
        else:
            valid.append(tok)
    if valid:
        return valid[0]
    r = s.post(f"{EDL_URL}/api/users/token", timeout=30)
    r.raise_for_status()
    return r.json()
