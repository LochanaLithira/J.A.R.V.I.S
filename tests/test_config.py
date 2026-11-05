"""Tests for the configuration settings."""
import re

from backend import config


def test_the_blocked_app_list_is_not_a_setting():
    assert not hasattr(config, "ALLOWED_APPS")          # apps come from the Start menu now
    assert not hasattr(config, "BLOCKED_PROGRAMS")      # security code, kept out of config


def test_audit_log_is_outside_the_workspace():
    assert config.AUDIT_LOG != config.WORKSPACE
    assert config.WORKSPACE not in config.AUDIT_LOG.parents


def test_limits_are_sane():
    assert 1 <= config.RATE_LIMIT_CALLS <= 100
    assert 1 <= config.RATE_LIMIT_WINDOW_SECONDS <= 3600



def test_models_are_valid_pinned_names():
    models = config.GEMINI_MODELS
    assert 1 <= len(models) <= 3 and len(set(models)) == len(models)
    for name in models:
        assert re.fullmatch(r"gemini-[a-z0-9][a-z0-9.\-]{1,40}", name)
        assert "latest" not in name                      # aliases silently change under you
        assert not name.startswith("gemini-2.5")         # scheduled for shutdown on 16 October 2026


def test_ai_timeout_is_sane():
    assert 1 <= config.GEMINI_TIMEOUT_SECONDS <= 60


def test_models_dir_is_outside_the_workspace():
    assert config.WORKSPACE != config.MODELS_DIR
    assert config.WORKSPACE not in config.MODELS_DIR.parents


def test_wake_settings_are_sane():
    assert 0.05 <= config.WAKE_THRESHOLD <= 0.95
    assert 1 <= config.WAKE_PATIENCE <= 10
    assert 0.5 <= config.WAKE_COOLDOWN_SECONDS <= 10
    assert 300 <= config.CLAP_MIN_LEVEL <= 32767
    assert config.MIC_DEVICE is None or (isinstance(config.MIC_DEVICE, str) and config.MIC_DEVICE.strip())


def test_follow_up_settings_are_sane():
    assert config.FOLLOW_UP_ON_QUESTION is True                 # answering a question needs no wake word
    assert not hasattr(config, "FOLLOW_UP_ALWAYS")              # replaced by conversation mode


def test_conversation_settings_are_sane():
    assert isinstance(config.CONVERSATION_MODE, bool)
    assert 3 <= config.CONVERSATION_SECONDS <= 15
    assert 1 <= config.CONVERSATION_MAX_TURNS <= 20
    assert 30 <= config.CONVERSATION_MAX_SECONDS <= 600       # a conversation can't stay open for long
    assert 1 <= config.CONVERSATION_MEMORY_TURNS <= 6
    assert 1.0 <= config.LISTEN_END_SILENCE_SECONDS <= 2.0
    assert 0 <= config.LISTEN_UNFINISHED_EXTRA_SECONDS <= 4
    assert 2 <= config.FOLLOW_UP_SECONDS <= 10
    assert 1 <= config.FOLLOW_UP_MAX_TURNS <= 5


def test_speech_input_settings_are_sane():
    assert re.fullmatch(r"whisper-[a-z0-9.\-]+", config.STT_MODEL)
    assert re.fullmatch(r"[a-z]{2}", config.STT_LANGUAGE)
    assert 3 <= config.LISTEN_MAX_SECONDS <= 30               # caps how much audio can leave per command
    assert 1 <= config.LISTEN_NO_SPEECH_SECONDS <= config.LISTEN_MAX_SECONDS
    assert 0.3 <= config.LISTEN_END_SILENCE_SECONDS <= 3