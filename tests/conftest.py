import pytest

from backend.gate.audit import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "logs" / "audit.jsonl")