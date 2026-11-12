"""Tests for _common.credentials (no QGIS needed for these)."""

from __future__ import annotations

import os

from mayim_tools._common.credentials import temporary_environment


def test_temporary_environment_sets_and_restores():
    os.environ.pop("MAYIM_TEST_A", None)
    os.environ["MAYIM_TEST_B"] = "old"
    with temporary_environment(MAYIM_TEST_A="1", MAYIM_TEST_B="new"):
        assert os.environ["MAYIM_TEST_A"] == "1"
        assert os.environ["MAYIM_TEST_B"] == "new"
    assert "MAYIM_TEST_A" not in os.environ
    assert os.environ["MAYIM_TEST_B"] == "old"
    del os.environ["MAYIM_TEST_B"]


def test_temporary_environment_restores_after_error():
    os.environ.pop("MAYIM_TEST_C", None)
    try:
        with temporary_environment(MAYIM_TEST_C="x"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "MAYIM_TEST_C" not in os.environ


def test_temporary_environment_empty_is_noop():
    before = dict(os.environ)
    with temporary_environment():
        pass
    assert dict(os.environ) == before


def test_token_expired():
    from datetime import date

    from mayim_tools._common.credentials import token_expired

    today = date(2026, 11, 12)
    assert token_expired({"expiration_date": "11/01/2026"}, today)
    assert token_expired({"expiration_date": "11/12/2026"}, today)
    assert not token_expired({"expiration_date": "01/05/2027"}, today)
    assert not token_expired({"expiration_date": "2027-01-05T10:00:00Z"}, today)
    assert not token_expired({}, today)


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    def __init__(self, tokens):
        self.tokens = tokens
        self.revoked = []
        self.created = 0

    def get(self, url, timeout):
        assert url.endswith("/api/users/tokens")
        return _Resp(self.tokens)

    def post(self, url, params=None, timeout=None):
        if url.endswith("/revoke_token"):
            self.revoked.append(params["token"])
            return _Resp({})
        self.created += 1
        return _Resp({"access_token": "new", "expiration_date": "01/11/2027"})


def test_valid_edl_token_revokes_expired_and_creates():
    from datetime import date

    from mayim_tools._common.credentials import valid_edl_token

    today = date(2026, 11, 12)
    s = _Session([{"access_token": "old", "expiration_date": "09/01/2026"}])
    tok = valid_edl_token("u", "p", session=s, today=today)
    assert s.revoked == ["old"] and s.created == 1 and tok["access_token"] == "new"
    assert s.auth == ("u", "p")

    s = _Session(
        [
            {"access_token": "old", "expiration_date": "09/01/2026"},
            {"access_token": "ok", "expiration_date": "12/30/2026"},
        ]
    )
    tok = valid_edl_token("u", "p", session=s, today=today)
    assert s.revoked == ["old"] and s.created == 0 and tok["access_token"] == "ok"


def test_earthdata_credentials_from_environment(monkeypatch, tmp_path):
    import pathlib

    from mayim_tools._common import credentials as c

    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("EARTHDATA_USERNAME", "me")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pw")
    assert c.earthdata_credentials() == ("me", "pw")
    monkeypatch.delenv("EARTHDATA_USERNAME")
    (tmp_path / ".netrc").write_text(
        "machine urs.earthdata.nasa.gov login nu password np\n"
    )
    assert c.earthdata_credentials() == ("nu", "np")


def test_earthdata_login_replaces_expired_session_token(monkeypatch):
    """An earthaccess session that holds an expired token gets the valid
    token from Earthdata Login (old earthaccess picks the first, expired,
    token; a long QGIS session keeps its first token)."""
    import sys
    import types

    from mayim_tools._common import credentials as c

    class Auth:
        authenticated = True
        token = {"access_token": "old", "expiration_date": "01/01/2020"}

    fake = types.ModuleType("earthaccess")
    fake.login = lambda strategy: Auth()
    monkeypatch.setitem(sys.modules, "earthaccess", fake)
    monkeypatch.setattr(c, "earthdata_credentials", lambda: ("u", "p"))
    monkeypatch.setattr(
        c,
        "valid_edl_token",
        lambda u, p: {"access_token": "new", "expiration_date": "01/01/2099"},
    )
    monkeypatch.setenv("EARTHDATA_USERNAME", "u")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p")
    auth = c.earthdata_login()
    assert auth.token["access_token"] == "new"
