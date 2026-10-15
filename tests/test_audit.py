import json

from backend.gate.audit import AuditLog


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_chain_verifies(audit):
    for i in range(5):
        audit.record("e", i=i)
    assert audit.verify() == (True, None)


def test_edit_detected(audit):
    for i in range(4):
        audit.record("e", i=i)
    ls = lines(audit.path)
    obj = json.loads(ls[1])
    obj["i"] = 99
    ls[1] = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    audit.path.write_text("\n".join(ls) + "\n", encoding="utf-8")
    assert audit.verify() == (False, 2)


def test_deleted_line_detected(audit):
    for i in range(4):
        audit.record("e", i=i)
    ls = lines(audit.path)
    del ls[1]
    audit.path.write_text("\n".join(ls) + "\n", encoding="utf-8")
    ok, bad = audit.verify()
    assert not ok and bad == 2


def test_garbage_line_detected(audit):
    audit.record("e")
    with open(audit.path, "a", encoding="utf-8") as f:
        f.write("not json\n")
    assert audit.verify() == (False, 2)


def test_chain_continues_across_instances(audit):
    audit.record("a")
    again = AuditLog(audit.path)
    again.record("b")
    assert again.verify() == (True, None)


def test_long_values_are_clipped(audit):
    audit.record("e", text="x" * 5000)
    assert len(lines(audit.path)[0]) < 1000