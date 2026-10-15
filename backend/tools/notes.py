"""Notes tools. All files live in <workspace>/notes and nowhere else."""
from __future__ import annotations

from pathlib import Path

from ..gate.models import Param, Risk, Tool, ToolError, ValidationError
from ..gate.validate import NAME_PATTERN, safe_file_in

MAX_TEXT = 5000
MAX_LIST = 50


def make_note_tools(workspace: Path) -> list[Tool]:
    notes_dir = Path(workspace) / "notes"

    def _path(title: str) -> Path:
        notes_dir.mkdir(parents=True, exist_ok=True)
        try:
            return safe_file_in(notes_dir, title, ".txt")
        except ValidationError as exc:
            raise ToolError("That note name isn't allowed.") from exc

    def create_note(title: str, text: str) -> str:
        path = _path(title)
        try:
            with open(path, "x", encoding="utf-8") as f:  # "x" means: never overwrite
                f.write(text)
        except FileExistsError:
            raise ToolError("A note with that title already exists.")
        return f"Created note {title}."

    def read_note(title: str) -> str:
        path = _path(title)
        if not path.is_file():
            raise ToolError("No such note.")
        return path.read_text(encoding="utf-8")[:MAX_TEXT]

    def list_notes() -> str:
        notes_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(p.stem for p in notes_dir.glob("*.txt") if p.is_file())[:MAX_LIST]
        return ", ".join(names) if names else "No notes yet."

    title = Param(str, max_len=64, pattern=NAME_PATTERN)
    return [
        Tool("create_note", "Create a new text note (never overwrites).", Risk.MEDIUM,
             {"title": title, "text": Param(str, max_len=MAX_TEXT)}, create_note),
        Tool("read_note", "Read a note by title.", Risk.LOW, {"title": title}, read_note),
        Tool("list_notes", "List note titles.", Risk.LOW, {}, list_notes),
    ]