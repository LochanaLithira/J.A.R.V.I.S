"""API keys live in Windows Credential Manager (via keyring), never in files.

Rules:
- Only names listed in KNOWN_NAMES can be stored or read.
- Values are validated, and are never logged, printed, or put in error messages.
- A key is stored only if the Windows vault is the active backend, so a
  plain-text fallback backend can never be used by accident.

Set a key from the terminal (the value is typed hidden, so it never lands in
your shell history):
    python -m backend.credentials set gemini_api_key
"""
from __future__ import annotations

import getpass
import re
import sys

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE = "jarvis"
KNOWN_NAMES = ("gemini_api_key", "stt_api_key", "fish_api_key")

# Visible ASCII only: no spaces, tabs or newlines; 8 to 256 characters.
_VALUE_RE = re.compile(r"[\x21-\x7e]{8,256}")


class CredentialError(Exception):
    """Safe to show to the user: never contains a key value."""


def _check_name(name: str) -> str:
    if name not in KNOWN_NAMES:
        raise CredentialError("Unknown credential name.")
    return name


def _is_secure_backend() -> bool:
    return type(keyring.get_keyring()).__name__ == "WinVaultKeyring"


def _require_secure_backend() -> None:
    if not _is_secure_backend():
        raise CredentialError("The Windows Credential Manager is not the active keyring backend.")


def set_secret(name: str, value: str) -> None:
    _check_name(name)
    if not isinstance(value, str) or _VALUE_RE.fullmatch(value) is None:
        raise CredentialError("That key looks invalid (8-256 visible characters, no spaces).")
    _require_secure_backend()
    try:
        keyring.set_password(SERVICE, name, value)
    except KeyringError:
        raise CredentialError("Could not save to the Credential Manager.") from None


def get_secret(name: str) -> str | None:
    _check_name(name)
    _require_secure_backend()
    try:
        return keyring.get_password(SERVICE, name)
    except KeyringError:
        raise CredentialError("Could not read from the Credential Manager.") from None


def delete_secret(name: str) -> bool:
    """Return True if a key was deleted, False if there was nothing to delete."""
    _check_name(name)
    _require_secure_backend()
    try:
        keyring.delete_password(SERVICE, name)
        return True
    except PasswordDeleteError:
        return False
    except KeyringError:
        raise CredentialError("Could not delete from the Credential Manager.") from None


def describe(name: str) -> str:
    """Safe status text: says whether a key is set and how long it is, never the value."""
    value = get_secret(name)
    return "not set" if value is None else f"set ({len(value)} characters)"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("set", "status", "delete"):
        print("usage: python -m backend.credentials [set|status|delete] <name>")
        print("names: " + ", ".join(KNOWN_NAMES))
        return 2
    command, name = argv
    try:
        _check_name(name)  # reject unknown names before asking for any secret
        if command == "set":
            value = getpass.getpass(f"Paste the value for {name} (input is hidden): ").strip()
            set_secret(name, value)
            print("Saved.")
        elif command == "status":
            print(f"{name}: {describe(name)}")
        else:
            print("Deleted." if delete_secret(name) else "Nothing to delete.")
    except CredentialError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())