"""
Password gate for the administrator ticket queue.

One shared administrator credential: a salted scrypt hash supplied through the
environment. With no hash configured the queue stays closed. This is an MVP gate,
not per-user RBAC, and the attempt limit applies per browser session only.

Generate a hash with:  python -m src.support.admin_auth
"""
import getpass
import hashlib
import hmac
import os
import secrets
from typing import MutableMapping

from .config import ADMIN_PASSWORD_HASH_ENV, MAX_ADMIN_LOGIN_ATTEMPTS

_MIN_PASSWORD_CHARS = 8
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)


def hash_password(password: str) -> str:
    if len(password) < _MIN_PASSWORD_CHARS:
        raise ValueError(f"The password must have at least {_MIN_PASSWORD_CHARS} characters.")
    salt = secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${_derive(password, salt).hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check. Malformed hashes never verify."""
    parts = encoded.split("$")
    if len(parts) != 3 or parts[0] != "scrypt":
        return False
    try:
        salt, expected = bytes.fromhex(parts[1]), bytes.fromhex(parts[2])
    except ValueError:
        return False
    if not salt or not expected:
        return False
    return hmac.compare_digest(_derive(password, salt), expected)


def admin_is_configured() -> bool:
    return bool(os.environ.get(ADMIN_PASSWORD_HASH_ENV))


def is_locked_out(state: MutableMapping) -> bool:
    return state["admin_failures"] >= MAX_ADMIN_LOGIN_ATTEMPTS


def attempt_login(state: MutableMapping, password: object) -> bool:
    """Check a password and update session state. Fails closed in every other case."""
    if is_locked_out(state) or not admin_is_configured() or not isinstance(password, str):
        return False
    if verify_password(password, os.environ[ADMIN_PASSWORD_HASH_ENV]):
        state["admin_authenticated"] = True
        state["admin_failures"] = 0
        return True
    state["admin_failures"] += 1
    return False


def main() -> None:
    password = getpass.getpass("New administrator password: ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("The passwords do not match.")
    try:
        print(hash_password(password))
    except ValueError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
