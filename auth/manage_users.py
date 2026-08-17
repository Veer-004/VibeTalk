"""
manage_users.py — Owner-only CLI to add, update, or remove VibeTalk users.

This is NOT imported by the app and has no UI inside it — the only way to
change who can log in, or anyone's password, is to run this script with
filesystem access to the repo. That's the whole point: logged-in users can
never modify credentials from within the app itself.

Usage:
    python auth/manage_users.py add veer "Veer" "newpassword"
    python auth/manage_users.py remove veer
    python auth/manage_users.py list
"""

import json
import os
import sys

import bcrypt

_USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load() -> dict:
    if not os.path.exists(_USERS_FILE):
        return {}
    with open(_USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(users: dict) -> None:
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
        f.write("\n")


def add_or_update(username: str, display_name: str, password: str) -> None:
    users = _load()
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users[username.strip().lower()] = {
        "display_name": display_name,
        "password_hash": hashed,
    }
    _save(users)
    print(f"Saved user '{username}'.")


def remove(username: str) -> None:
    users = _load()
    key = username.strip().lower()
    if key in users:
        del users[key]
        _save(users)
        print(f"Removed user '{username}'.")
    else:
        print(f"No such user '{username}'.")


def list_users() -> None:
    users = _load()
    if not users:
        print("No users yet.")
        return
    for key, rec in users.items():
        print(f"{key} -> {rec['display_name']}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cmd = args[0]
    if cmd == "add" and len(args) == 4:
        add_or_update(args[1], args[2], args[3])
    elif cmd == "remove" and len(args) == 2:
        remove(args[1])
    elif cmd == "list":
        list_users()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
