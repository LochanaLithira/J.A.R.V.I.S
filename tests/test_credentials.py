import sys

import pytest
from keyring.errors import KeyringError, PasswordDeleteError

from backend import credentials as cred

GOOD = "ABCDEFGH12345678"


class FakeKeyring:
    """Stands in for the Windows vault so tests never touch your real credentials."""

    def __init__(self):
        self.data = {}

    def set_password(self, service, name, value):
        self.data[(service, name)] = value

    def get_password(self, service, name):
        return self.data.get((service, name))

    def delete_password(self, service, name):
        if (service, name) not in self.data:
            raise PasswordDeleteError("not found")
        del self.data[(service, name)]


@pytest.fixture
def fake(monkeypatch):
    fk = FakeKeyring()
    monkeypatch.setattr(cred, "keyring", fk)
    monkeypatch.setattr(cred, "_is_secure_backend", lambda: True)
    return fk


def test_roundtrip(fake):
    cred.set_secret("gemini_api_key", GOOD)
    assert cred.get_secret("gemini_api_key") == GOOD
    assert cred.get_secret("stt_api_key") is None


def test_unknown_names_rejected_everywhere(fake):
    for call in (lambda: cred.set_secret("other", GOOD), lambda: cred.get_secret("other"),
                 lambda: cred.delete_secret("other")):
        with pytest.raises(cred.CredentialError):
            call()
    assert fake.data == {}


@pytest.mark.parametrize("bad", [
    "", "short", "has space 12345", "tab\t12345678", "newline\n12345678", "x" * 257, None, 12345678,
])
def test_invalid_values_rejected_and_not_stored(fake, bad):
    with pytest.raises(cred.CredentialError):
        cred.set_secret("gemini_api_key", bad)
    assert fake.data == {}


def test_bad_value_never_appears_in_the_error(fake):
    with pytest.raises(cred.CredentialError) as e:
        cred.set_secret("gemini_api_key", "SECRETSECRET with spaces")
    assert "SECRET" not in str(e.value)


def test_delete_reports_whether_anything_was_deleted(fake):
    cred.set_secret("gemini_api_key", GOOD)
    assert cred.delete_secret("gemini_api_key") is True
    assert cred.delete_secret("gemini_api_key") is False


def test_refuses_when_windows_vault_is_not_the_backend(fake, monkeypatch):
    monkeypatch.setattr(cred, "_is_secure_backend", lambda: False)
    with pytest.raises(cred.CredentialError):
        cred.set_secret("gemini_api_key", GOOD)
    with pytest.raises(cred.CredentialError):
        cred.get_secret("gemini_api_key")
    assert fake.data == {}


def test_keyring_errors_become_safe_errors(fake):
    def boom(*a):
        raise KeyringError("backend exploded with ABCDEFGH12345678")

    fake.set_password = boom
    with pytest.raises(cred.CredentialError) as e:
        cred.set_secret("gemini_api_key", GOOD)
    assert "ABCDEFGH" not in str(e.value) and e.value.__cause__ is None


def test_describe_shows_length_never_the_value(fake):
    assert cred.describe("gemini_api_key") == "not set"
    cred.set_secret("gemini_api_key", GOOD)
    text = cred.describe("gemini_api_key")
    assert "16 characters" in text and "ABCDEFGH" not in text


def test_cli_set_and_status_never_print_the_value(fake, monkeypatch, capsys):
    monkeypatch.setattr(cred.getpass, "getpass", lambda prompt="": GOOD)
    assert cred.main(["set", "gemini_api_key"]) == 0
    assert cred.main(["status", "gemini_api_key"]) == 0
    out = capsys.readouterr().out
    assert "set (16 characters)" in out and "ABCDEFGH" not in out


def test_cli_bad_usage_and_bad_name(fake, capsys):
    assert cred.main(["bogus"]) == 2
    assert cred.main(["set", "nope"]) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="checks the real Windows backend")
def test_real_backend_on_this_machine_is_the_windows_vault():
    assert cred._is_secure_backend()