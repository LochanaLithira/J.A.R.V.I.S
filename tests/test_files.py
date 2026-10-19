import ctypes
import os
import subprocess
import sys
import time

import pytest

from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import ToolError
from backend.tools import files as ft
from backend.tools.files import file_folders, known_folder, make_file_tools


@pytest.fixture
def setup(tmp_path):
    docs, down, trash = tmp_path / "Documents", tmp_path / "Downloads", tmp_path / "ws" / "trash"
    docs.mkdir()
    down.mkdir()
    opened = []
    tools = make_file_tools({"documents": docs, "downloads": down}, trash, opener=opened.append)
    gate = Gate(tools, AuditLog(tmp_path / "log.jsonl"), max_calls=1000)
    return gate, docs, down, trash, opened


def run(gate, tool, **args):
    return gate.execute(tool, args)


def make(path, text="x", age=0):
    path.write_text(text)
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return path


def set_attrs(path, attrs):
    assert ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs)


# ---- listing ---------------------------------------------------------------------------

def test_list_shows_newest_first_and_only_plain_files(setup):
    gate, docs, *_ = setup
    make(docs / "old.txt", age=500)
    make(docs / "new.txt", age=10)
    (docs / "subfolder").mkdir()
    make(docs / "desktop.ini")
    make(docs / "~$draft.docx")
    r = run(gate, "list_files", folder="documents")
    assert r.ok and r.private
    assert r.message == "Documents has 2 files. Newest first: new.txt, old.txt."


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file attributes")
@pytest.mark.parametrize("attrs", [0x2, 0x4])                     # hidden, system
def test_hidden_and_system_files_are_invisible(setup, attrs):
    gate, docs, *_ = setup
    secret = make(docs / "secret.txt")
    set_attrs(secret, attrs)
    assert run(gate, "list_files", folder="documents").message == "Documents has no files."
    assert not run(gate, "open_file", folder="documents", name="secret.txt").ok


def test_unknown_folder_refused(setup):
    gate, *_ = setup
    assert not run(gate, "list_files", folder="windows").ok
    assert not run(gate, "list_files", folder="C:\\").ok


# ---- names are names, never paths ------------------------------------------------------------

@pytest.mark.parametrize("name", ["..\\secret.txt", "a/b.txt", "C:secret.txt", "x\x00.txt", "", "a" * 121,
                                  "what?.txt", "pipe|.txt"])
def test_path_syntax_is_refused_before_any_tool_runs(setup, name):
    gate, docs, down, trash, opened = setup
    make(docs.parent / "secret.txt")
    r = run(gate, "open_file", folder="documents", name=name)
    assert not r.ok and opened == []


@pytest.mark.parametrize("name", ["CON.txt", "nul", " lead.txt", "trail.txt ", "dots.", "..", "..."])
def test_odd_names_are_refused_by_the_tool(name):
    with pytest.raises(ToolError):
        ft._check_name(name)


def test_matching_by_full_name_or_without_its_ending(setup):
    gate, docs, *_, opened = setup
    make(docs / "Report.DOCX")
    assert run(gate, "open_file", folder="documents", name="report").ok
    assert run(gate, "open_file", folder="documents", name="report.docx").ok
    assert opened == [str((docs / "Report.DOCX").resolve())] * 2


def test_ambiguous_names_ask_for_the_full_name(setup):
    gate, docs, *_, opened = setup
    make(docs / "budget.xlsx")
    make(docs / "budget.pdf")
    r = run(gate, "open_file", folder="documents", name="budget")
    assert not r.ok and "budget.pdf, budget.xlsx" in r.message and r.private and opened == []


def test_missing_file_is_explained_privately(setup):
    gate, *_ = setup
    r = run(gate, "open_file", folder="documents", name="tax return")
    assert not r.ok and "tax return" in r.message and r.private      # the name is never spoken


# ---- links and junctions can't lead outside ----------------------------------------------------

def test_a_file_link_is_invisible(setup, tmp_path):
    gate, docs, *_, opened = setup
    outside = make(tmp_path / "outside.txt", "private")
    try:
        os.symlink(outside, docs / "link.txt")
    except OSError:
        pytest.skip("symlinks not permitted here")
    assert run(gate, "list_files", folder="documents").message == "Documents has no files."
    assert not run(gate, "open_file", folder="documents", name="link.txt").ok and opened == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
def test_a_junction_is_never_followed(setup, tmp_path):
    gate, docs, *_ = setup
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    make(elsewhere / "budget-secret.txt")
    subprocess.run(["cmd", "/c", "mklink", "/J", str(docs / "jump"), str(elsewhere)], check=True,
                   capture_output=True)
    r = run(gate, "find_files", name="budget")
    assert r.ok and "budget-secret" not in r.message


def test_a_file_swapped_after_listing_is_refused(setup, tmp_path):
    _, docs, *_ = setup
    make(docs / "a.txt")
    with pytest.raises(ToolError):
        ft._inside(docs / "missing.txt", docs)


# ---- opening -------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["setup.exe", "run.bat", "go.cmd", "s.ps1", "x.vbs", "link.lnk", "page.html",
                                  "macro.docm", "invoice.pdf.exe", "noending"])
def test_programs_and_scripts_are_never_opened(setup, name):
    gate, docs, *_, opened = setup
    make(docs / name)
    r = run(gate, "open_file", folder="documents", name=name)
    assert not r.ok and opened == []


@pytest.mark.parametrize("name", ["notes.txt", "photo.JPG", "song.mp3", "talk.pdf", "film.mp4"])
def test_documents_and_media_open(setup, name):
    gate, docs, *_, opened = setup
    make(docs / name)
    r = run(gate, "open_file", folder="documents", name=name)
    assert r.ok and opened == [str((docs / name).resolve())]


def test_open_folder_is_not_private(setup):
    gate, docs, *_, opened = setup
    r = run(gate, "open_folder", folder="documents")
    assert r.ok and not r.private and opened == [str(docs.resolve())]


# ---- moving and renaming never overwrite -----------------------------------------------------

def test_move_file(setup):
    gate, docs, down, *_ = setup
    make(down / "cv.pdf", "mine")
    assert run(gate, "move_file", folder="downloads", name="cv", to_folder="documents").ok
    assert (docs / "cv.pdf").read_text() == "mine" and not (down / "cv.pdf").exists()


def test_move_never_overwrites(setup):
    gate, docs, down, *_ = setup
    make(down / "cv.pdf", "new")
    make(docs / "cv.pdf", "old")
    r = run(gate, "move_file", folder="downloads", name="cv.pdf", to_folder="documents")
    assert not r.ok and (docs / "cv.pdf").read_text() == "old" and (down / "cv.pdf").exists()


def test_move_to_the_same_folder_is_refused(setup):
    gate, docs, *_ = setup
    make(docs / "a.txt")
    assert not run(gate, "move_file", folder="documents", name="a.txt", to_folder="documents").ok


def test_rename_keeps_the_file_type(setup):
    gate, docs, *_ = setup
    make(docs / "draft.docx")
    assert run(gate, "rename_file", folder="documents", name="draft", new_name="final").ok
    assert (docs / "final.docx").exists()


@pytest.mark.parametrize("new_name", ["final.exe", "final.bat", "final.txt"])
def test_rename_cannot_change_the_type(setup, new_name):
    gate, docs, *_ = setup
    make(docs / "draft.docx")
    r = run(gate, "rename_file", folder="documents", name="draft.docx", new_name=new_name)
    assert not r.ok and (docs / "draft.docx").exists()


def test_rename_never_overwrites(setup):
    gate, docs, *_ = setup
    make(docs / "a.txt", "a")
    make(docs / "b.txt", "b")
    assert not run(gate, "rename_file", folder="documents", name="a.txt", new_name="b.txt").ok
    assert (docs / "b.txt").read_text() == "b"


@pytest.mark.parametrize("new_name", ["..\\up.txt", "sub/x.txt", "CON.txt", "x."])
def test_rename_refuses_tricky_new_names(setup, new_name):
    gate, docs, *_ = setup
    make(docs / "a.txt")
    assert not run(gate, "rename_file", folder="documents", name="a.txt", new_name=new_name).ok
    assert (docs / "a.txt").exists()


# ---- delete is reversible and needs a keypress ---------------------------------------------------

def test_delete_needs_a_keypress(setup):
    gate, docs, *_ = setup
    make(docs / "plan.txt")
    r = run(gate, "delete_file", folder="documents", name="plan.txt")
    assert not r.ok and r.message == "Cancelled." and (docs / "plan.txt").exists()


def test_confirmed_delete_moves_to_the_trash(tmp_path):
    docs, trash = tmp_path / "Documents", tmp_path / "ws" / "trash"
    docs.mkdir()
    make(docs / "plan.txt", "keep me")
    asked = []
    gate = Gate(make_file_tools({"documents": docs}, trash), AuditLog(tmp_path / "log.jsonl"),
                confirm=lambda tool, args: asked.append((tool, args)) or True)
    r = gate.execute("delete_file", {"folder": "documents", "name": "plan.txt"})
    assert r.ok and r.private and asked == [("delete_file", {"folder": "documents", "name": "plan.txt"})]
    assert not (docs / "plan.txt").exists()
    (kept,) = list(trash.iterdir())
    assert kept.name.endswith("-plan.txt") and kept.read_text() == "keep me"     # nothing is lost


# ---- finding ------------------------------------------------------------------------------

def test_find_searches_folders_and_subfolders(setup):
    gate, docs, down, *_ = setup
    (docs / "finance" / "2026").mkdir(parents=True)
    make(docs / "finance" / "2026" / "budget.xlsx")
    make(down / "Budget-old.pdf")
    r = run(gate, "find_files", name="budget")
    assert r.ok and r.private
    assert "budget.xlsx in documents\\finance\\2026" in r.message and "Budget-old.pdf in downloads" in r.message


def test_find_stops_at_the_depth_limit(setup):
    gate, docs, *_ = setup
    deep = docs / "a" / "b" / "c"
    deep.mkdir(parents=True)
    make(deep / "deep-budget.txt")
    assert "deep-budget" not in run(gate, "find_files", name="budget").message


def test_find_caps_the_results(setup):
    gate, docs, *_ = setup
    for i in range(40):
        make(docs / f"photo{i}.jpg")
    assert run(gate, "find_files", name="photo").message.count("photo") == ft.MAX_FOUND


def test_find_skips_the_trash_and_hidden_folders(setup):
    gate, docs, down, trash, _ = setup
    trash.mkdir(parents=True)
    make(trash / "20260101-budget.txt")
    hidden = docs / "private"
    hidden.mkdir()
    make(hidden / "budget-hidden.txt")
    set_attrs(hidden, 0x2)
    message = run(gate, "find_files", name="budget").message
    assert "budget-hidden" not in message and "20260101" not in message


def test_find_needs_two_letters(setup):
    gate, *_ = setup
    assert not run(gate, "find_files", name="a").ok


# ---- folders ------------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows known folders")
def test_known_folders_are_found_where_windows_keeps_them():
    docs = known_folder("documents")
    assert docs is not None and docs.is_dir()
    assert known_folder("system32") is None


def test_only_existing_folders_are_offered(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "known_folder", lambda name: tmp_path / "missing" if name == "music" else None)
    folders = file_folders(tmp_path, ["workspace", "music", "videos"])
    assert folders == {"workspace": tmp_path}


def test_no_folders_no_tools(tmp_path):
    assert make_file_tools({}, tmp_path / "trash") == []
