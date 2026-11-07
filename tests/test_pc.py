"""PC controls, with a fake PC: the tests never touch your real volume, screen or lock."""
import pytest

from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import ToolError
from backend.tools import pc as pc_module
from backend.tools.pc import make_pc_tools


class FakePC:
    def __init__(self, volume=50, muted=False, brightness=50, battery=(80, False), cpu=12):
        self.vol, self.mute, self.bright, self.batt, self.load = volume, muted, brightness, battery, cpu
        self.keys, self.locked = [], 0

    def volume(self):
        return self.vol

    def set_volume(self, percent):
        self.vol = percent

    def muted(self):
        return self.mute

    def set_mute(self, mute):
        self.mute = mute

    def press(self, key):
        self.keys.append(key)

    def brightness(self):
        return self.bright

    def set_brightness(self, percent):
        if self.bright is None:
            raise ToolError("I can't change the brightness of this screen.")
        self.bright = percent

    def battery(self):
        return self.batt

    def cpu(self):
        return self.load

    def lock(self):
        self.locked += 1


@pytest.fixture
def pc():
    return FakePC()


def gate(tmp_path, pc):
    return Gate(make_pc_tools(pc), AuditLog(tmp_path / "log.jsonl"), max_calls=1000)


def run(tmp_path, pc, tool, **args):
    return gate(tmp_path, pc).execute(tool, args)


def test_every_pc_tool_is_low_risk_and_not_private(pc):
    for tool in make_pc_tools(pc):
        assert tool.risk.value == "low" and tool.private_output is False, tool.name


# ---- sound -------------------------------------------------------------------------------------

def test_set_volume(tmp_path, pc):
    r = run(tmp_path, pc, "set_volume", percent=30)
    assert r.ok and r.message == "Volume at 30 percent." and pc.vol == 30


def test_setting_a_volume_unmutes(tmp_path):
    pc = FakePC(muted=True)
    run(tmp_path, pc, "set_volume", percent=40)
    assert pc.mute is False


@pytest.mark.parametrize("percent", [-1, 101, 1000, "50", 50.5, True, None])
def test_volume_out_of_range_is_refused(tmp_path, pc, percent):
    assert not run(tmp_path, pc, "set_volume", percent=percent).ok
    assert pc.vol == 50


@pytest.mark.parametrize("action, level", [("up", 60), ("down", 40)])
def test_volume_steps(tmp_path, pc, action, level):
    r = run(tmp_path, pc, "volume", action=action)
    assert r.ok and pc.vol == level and r.message == f"Volume {action} to {level} percent."


def test_volume_steps_stop_at_the_ends(tmp_path):
    loud, quiet = FakePC(volume=95), FakePC(volume=4)
    run(tmp_path, loud, "volume", action="up")
    run(tmp_path, quiet, "volume", action="down")
    assert (loud.vol, quiet.vol) == (100, 0)


def test_volume_up_unmutes_like_the_keyboard_key(tmp_path):
    pc = FakePC(muted=True)
    run(tmp_path, pc, "volume", action="up")
    assert pc.mute is False


def test_mute_and_unmute(tmp_path, pc):
    assert run(tmp_path, pc, "volume", action="mute").message == "Muted." and pc.mute
    assert run(tmp_path, pc, "volume", action="unmute").ok and not pc.mute


def test_unknown_volume_action_is_refused(tmp_path, pc):
    for action in ("max", "louder", "up; shutdown", ""):
        assert not run(tmp_path, pc, "volume", action=action).ok


# ---- media keys --------------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["play_pause", "next", "previous", "stop"])
def test_media_keys(tmp_path, pc, action):
    assert run(tmp_path, pc, "media", action=action).ok and pc.keys == [action]


def test_only_media_keys_can_be_pressed(tmp_path, pc):
    for action in ("enter", "alt+f4", "win+r", "lock", "PLAY_PAUSE"):
        assert not run(tmp_path, pc, "media", action=action).ok
    assert pc.keys == []


def test_media_keys_are_the_windows_media_codes():
    assert pc_module.MEDIA_KEYS == {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}


# ---- brightness --------------------------------------------------------------------------------

def test_set_brightness(tmp_path, pc):
    r = run(tmp_path, pc, "set_brightness", percent=70)
    assert r.ok and pc.bright == 70


def test_brightness_never_goes_fully_black(tmp_path, pc):
    r = run(tmp_path, pc, "set_brightness", percent=0)
    assert r.ok and pc.bright == pc_module.MIN_BRIGHTNESS and r.message == "Brightness at 5 percent."
    dim = FakePC(brightness=8)
    run(tmp_path, dim, "brightness", action="down")
    assert dim.bright == pc_module.MIN_BRIGHTNESS


@pytest.mark.parametrize("action, level", [("up", 60), ("down", 40)])
def test_brightness_steps(tmp_path, pc, action, level):
    assert run(tmp_path, pc, "brightness", action=action).ok and pc.bright == level


def test_a_screen_without_brightness_control_says_so(tmp_path):
    pc = FakePC(brightness=None)
    for tool, args in (("brightness", {"action": "up"}), ("set_brightness", {"percent": 50})):
        r = gate(tmp_path, pc).execute(tool, args)
        assert not r.ok and r.message == "I can't change the brightness of this screen."


# ---- status ------------------------------------------------------------------------------------

def test_status_battery_and_cpu(tmp_path, pc):
    assert run(tmp_path, pc, "pc_status", what="battery").message == "Battery at 80 percent, on battery."
    assert run(tmp_path, pc, "pc_status", what="cpu").message == "The processor is at 12 percent."


def test_status_all(tmp_path):
    msg = run(tmp_path, FakePC(battery=(41, True), muted=True), "pc_status", what="all").message
    assert msg == ("Battery at 41 percent, plugged in. The processor is at 12 percent. Sound is muted. "
                   "Brightness is at 50 percent.")


def test_status_on_a_desktop_pc(tmp_path):
    pc = FakePC(battery=(None, True), brightness=None)
    assert "no battery" in run(tmp_path, pc, "pc_status", what="battery").message
    assert "can't read" in run(tmp_path, pc, "pc_status", what="brightness").message
    assert "Brightness" not in run(tmp_path, pc, "pc_status", what="all").message


# ---- lock --------------------------------------------------------------------------------------

def test_lock(tmp_path, pc):
    r = run(tmp_path, pc, "lock_pc")
    assert r.ok and pc.locked == 1 and r.message == "Locking the PC."


def test_lock_takes_no_arguments(tmp_path, pc):
    assert not run(tmp_path, pc, "lock_pc", delay=5).ok and pc.locked == 0


# ---- the real Windows calls that are safe to run in a test --------------------------------------

def test_real_battery_and_cpu_readings_are_sane():
    real = pc_module.WindowsPC()
    percent, plugged = real.battery()
    assert percent is None or 0 <= percent <= 100
    assert isinstance(plugged, bool) and 0 <= real.cpu() <= 100
