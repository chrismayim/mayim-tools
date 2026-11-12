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
    return auth
