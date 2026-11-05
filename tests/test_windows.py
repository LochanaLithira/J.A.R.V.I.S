"""Switching, minimizing and closing windows, with a fake desktop (no real window is touched)."""
import pytest

from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import ToolError
from backend.tools.apps import AppCatalog, AppEntry
from backend.tools.windows import Window, make_window_tools

OFFICE = r"C:\Program Files\Microsoft Office\root\Office16"
FRAME = r"C:\Windows\System32\ApplicationFrameHost.exe"

ENTRIES = [
    AppEntry("Word", "Microsoft.Office.WINWORD.EXE.15", OFFICE + r"\WINWORD.EXE"),
    AppEntry("Excel", "Microsoft.Office.EXCEL.EXE.15", OFFICE + r"\EXCEL.EXE"),
    AppEntry("Brave", "Brave", r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"),
]

# Front to back, as Windows lists them.
WINDOWS = [
    Window(1, 100, r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe", "Bank statement - Brave"),
    Window(2, 200, OFFICE + r"\WINWORD.EXE", "Secret plan.docx - Word"),
    Window(3, 200, OFFICE + r"\WINWORD.EXE", "Letter.docx - Word"),
    Window(4, 300, OFFICE + r"\EXCEL.EXE", "budget 2026.xlsx - Excel"),
    Window(5, 400, FRAME, "Calculator"),
    Window(6, 500, r"C:\Windows\explorer.exe", "Downloads"),
    Window(7, 600, r"C:\Users\u\AppData\Local\Discord\Discord.exe", "Discord"),
    Window(8, 999, r"C:\Python314\pythonw.exe", "J.A.R.V.I.S"),
]


class FakeDesktop:
    def __init__(self, windows=WINDOWS, focus_works=True):
        self._windows, self.focus_works = list(windows), focus_works
        self.focused, self.closed, self.minimized = [], [], 0

    def windows(self):
        return list(self._windows)

    def focus(self, window):
        self.focused.append(window.handle)
        return self.focus_works

    def minimize_all(self):
        self.minimized += 1

    def close(self, window):
        self.closed.append(window.handle)


def gate(tmp_path, desktop, own_pid=12345):
    catalog = AppCatalog(lister=lambda: ENTRIES, allowed=lambda e: True)
    return Gate(make_window_tools(catalog, desktop, own_pid=own_pid), AuditLog(tmp_path / "log.jsonl"),
                max_calls=1000)


def run(tmp_path, desktop, tool, **args):
    return gate(tmp_path, desktop).execute(tool, args)


def test_risk_levels(tmp_path):
    tools = {t.name: t for t in make_window_tools(AppCatalog(lister=lambda: []), FakeDesktop())}
    assert {n: t.risk.value for n, t in tools.items()} == {"switch_to_app": "low", "minimize_all": "low",
                                                           "close_app": "medium"}
    assert not any(t.private_output for t in tools.values())


# ---- switching ------------------------------------------------------------------------------------

def test_switch_by_start_menu_name_picks_the_front_window(tmp_path):
    d = FakeDesktop()
    r = run(tmp_path, d, "switch_to_app", name="word")
    assert r.ok and r.message == "Switching to Word." and d.focused == [2]


def test_switch_by_program_name(tmp_path):
    d = FakeDesktop()
    assert run(tmp_path, d, "switch_to_app", name="discord").ok and d.focused == [7]


def test_store_apps_are_found_by_their_title(tmp_path):
    d = FakeDesktop()
    r = run(tmp_path, d, "switch_to_app", name="calculator")
    assert r.ok and d.focused == [5] and r.message == "Switching to calculator."


def test_a_title_match_never_puts_the_title_in_the_reply(tmp_path):
    d = FakeDesktop()
    r = run(tmp_path, d, "switch_to_app", name="budget 2026")
    assert r.ok and d.focused == [4] and r.message == "Switching to Excel."
    r = run(tmp_path, d, "switch_to_app", name="secret plan")
    assert r.ok and "docx" not in r.message and r.message == "Switching to Word."


def test_app_names_beat_titles(tmp_path):
    windows = [Window(1, 1, r"C:\x\notes.exe", "Word count notes"), *WINDOWS]
    d = FakeDesktop(windows)
    run(tmp_path, d, "switch_to_app", name="word")
    assert d.focused == [2]                    # Word itself, not a window whose title says "word"


def test_nothing_open_says_so(tmp_path):
    r = run(tmp_path, FakeDesktop(), "switch_to_app", name="spotify")
    assert not r.ok and r.message == "I can't see that app open."


def test_two_apps_matching_are_asked_about(tmp_path):
    windows = [Window(1, 1, r"C:\a\Google Chrome.exe", "x"), Window(2, 2, r"C:\b\Chrome Remote.exe", "y")]
    r = run(tmp_path, FakeDesktop(windows), "switch_to_app", name="chrome")
    assert not r.ok and r.message.startswith("Did you mean")


def test_a_refused_switch_is_reported(tmp_path):
    r = run(tmp_path, FakeDesktop(focus_works=False), "switch_to_app", name="word")
    assert not r.ok and "wouldn't let me" in r.message


def test_minimize_all(tmp_path):
    d = FakeDesktop()
    assert run(tmp_path, d, "minimize_all").ok and d.minimized == 1


# ---- closing --------------------------------------------------------------------------------------

def test_close_sends_one_polite_close_to_one_window(tmp_path):
    d = FakeDesktop()
    r = run(tmp_path, d, "close_app", name="word")
    assert r.ok and r.message == "Asking Word to close." and d.closed == [2]      # not window 3 too


@pytest.mark.parametrize("name", ["explorer", "downloads", "python", "pythonw", "jarvis"])
def test_explorer_and_jarvis_are_never_closed(tmp_path, name):
    d = FakeDesktop()
    r = run(tmp_path, d, "close_app", name=name)
    assert not r.ok and d.closed == []


def test_own_process_is_never_closed(tmp_path):
    d = FakeDesktop([Window(1, 4242, r"C:\x\jarvis.exe", "J.A.R.V.I.S screen")])
    r = gate(tmp_path, d, own_pid=4242).execute("close_app", {"name": "jarvis"})
    assert not r.ok and r.message == "I'm not allowed to close that one." and d.closed == []


def test_unknown_program_is_never_closed(tmp_path):
    d = FakeDesktop([Window(1, 4, "", "Mystery")])       # couldn't read which program owns it
    assert not run(tmp_path, d, "close_app", name="mystery").ok and d.closed == []


@pytest.mark.parametrize("name", ["", "word; taskkill /f", "*", r"C:\Windows\explorer.exe", "x" * 65])
def test_odd_names_are_refused_before_anything_happens(tmp_path, name):
    d = FakeDesktop()
    assert not run(tmp_path, d, "close_app", name=name).ok and d.closed == []


def test_a_broken_app_list_still_lets_windows_be_found(tmp_path):
    def broken():
        raise OSError("no shell")

    d = FakeDesktop()
    g = Gate(make_window_tools(AppCatalog(lister=broken), d, own_pid=1), AuditLog(tmp_path / "log.jsonl"))
    assert g.execute("switch_to_app", {"name": "winword"}).ok and d.focused == [2]


def test_real_desktop_lists_windows_without_crashing():
    import sys
    if sys.platform != "win32":
        pytest.skip("Windows only")
    from backend.tools.windows import WindowsDesktop
    windows = WindowsDesktop().windows()
    assert all(isinstance(w.handle, int) and isinstance(w.title, str) for w in windows)
