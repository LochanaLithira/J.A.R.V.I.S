import os

import pytest

from backend.gate.models import Param, ValidationError
from backend.gate.validate import safe_file_in, validate_args

P = {"name": Param(str, max_len=5, pattern=r"[a-z]+"), "n": Param(int, min_value=1, max_value=3)}


def test_valid_args_pass():
    assert validate_args(P, {"name": "abc", "n": 2}) == {"name": "abc", "n": 2}


@pytest.mark.parametrize("bad", [
    None, [], "x", {"name": "abc"}, {"name": "abc", "n": 2, "extra": 1},
    {"name": 5, "n": 2}, {"name": "abc", "n": True}, {"name": "abc", "n": "2"},
    {"name": "abcdef", "n": 2}, {"name": "ABC", "n": 2}, {"name": "ab\x00", "n": 2},
    {"name": "abc", "n": 0}, {"name": "abc", "n": 4},
])
def test_bad_args_rejected(bad):
    with pytest.raises(ValidationError):
        validate_args(P, bad)


def test_choices_enforced():
    with pytest.raises(ValidationError):
        validate_args({"app": Param(str, choices=("a", "b"))}, {"app": "c"})


def test_error_messages_do_not_echo_values():
    with pytest.raises(ValidationError) as e:
        validate_args(P, {"name": "SECRET!!", "n": 2})
    assert "SECRET" not in str(e.value)


@pytest.mark.parametrize("name", [
    "../x", "..", "a/b", "a\\b", "C:evil", "C:\\evil", "/etc/passwd", "note.txt", "x:stream",
    "con", "NUL", "com1", "LPT9", "trailing ", " leading", "", "a" * 65, ".env", "x\x00y", "~root", "a\nb",
])
def test_unsafe_names_rejected(tmp_path, name):
    with pytest.raises(ValidationError):
        safe_file_in(tmp_path, name, ".txt")


@pytest.mark.parametrize("name", ["ideas", "My note-1", "a", "Shopping_list 2"])
def test_good_names_stay_inside_base(tmp_path, name):
    p = safe_file_in(tmp_path, name, ".txt")
    assert p.parent == tmp_path.resolve() and p.suffix == ".txt"


def test_symlink_escape_rejected(tmp_path):
    base, outside = tmp_path / "notes", tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    try:
        os.symlink(outside / "secret.txt", base / "leak.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    with pytest.raises(ValidationError):
        safe_file_in(base, "leak", ".txt")