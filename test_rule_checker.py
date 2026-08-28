"""
Unit tests for NetSage AI.

    python -m pytest tests -q          (or: python tests/test_rule_checker.py)

These are deliberately small and specific. Each one pins a rule that was wrong at some
point during development — the four marked REGRESSION correspond to entries RAI-01, RAI-02,
RAI-03 and RAI-07 in data/responsible_ai_log.csv.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ai_engine import Diagnosis, DiagnosisEngine, extract_json  # noqa: E402
from src.review_manager import ReviewManager, ReviewStore  # noqa: E402
from src.rule_checker import CHECKS, RuleChecker, split_blocks  # noqa: E402


def case(show_outputs, **kw):
    base = {"case_id": "T-00", "title": "t", "symptom": "", "topology_note": "",
            "show_outputs": show_outputs, "expected_fault": "", "expected_root_cause": "",
            "osi_layer": "L3", "concept_tag": "addressing", "severity": "Medium",
            "expected_next_command": "", "expected_fix_steps": ""}
    base.update(kw)
    return base


def ids(findings):
    return {f.check_id for f in findings}


RC = RuleChecker()


# ======================================================================================
# Parsing
# ======================================================================================

def test_split_blocks_separates_commands():
    blocks = split_blocks(
        "R1# show ip route\nS* 0.0.0.0/0 via 1.1.1.1\n\nPC1> ipconfig\n   IP Address...: 1.2.3.4"
    )
    assert [b.device for b in blocks] == ["R1", "PC1"]
    assert blocks[0].command == "show ip route"
    assert "S* 0.0.0.0/0" in blocks[0].body


# ======================================================================================
# Core checks
# ======================================================================================

def test_interface_admin_down():
    f = RC.run(case(
        "R1# show ip interface brief\n"
        "GigabitEthernet0/1     10.0.0.1        YES manual administratively down down"))
    assert "IFACE_ADMIN_DOWN" in ids(f)


def test_gateway_outside_host_subnet():
    f = RC.run(case(
        "PC1> ipconfig\n"
        "   IP Address......................: 192.168.10.77\n"
        "   Subnet Mask.....................: 255.255.255.0\n"
        "   Default Gateway.................: 192.168.1.1"))
    assert "GATEWAY_MISMATCH" in ids(f)


def test_gateway_inside_subnet_is_clean():
    f = RC.run(case(
        "PC1> ipconfig\n"
        "   IP Address......................: 192.168.10.77\n"
        "   Subnet Mask.....................: 255.255.255.0\n"
        "   Default Gateway.................: 192.168.10.1"))
    assert "GATEWAY_MISMATCH" not in ids(f)


def test_duplicate_ip_across_two_hosts():
    f = RC.run(case(
        "A> ipconfig\n   IP Address......................: 10.0.0.5\n"
        "   Subnet Mask.....................: 255.255.255.0\n"
        "   Default Gateway.................: 10.0.0.1\n\n"
        "B> ipconfig\n   IP Address......................: 10.0.0.5\n"
        "   Subnet Mask.....................: 255.255.255.0\n"
        "   Default Gateway.................: 10.0.0.1"))
    assert "DUPLICATE_IP" in ids(f)


def test_no_default_route():
    f = RC.run(case("R1# show ip route\nGateway of last resort is not set\n"
                    "C    10.0.0.0/24 is directly connected, GigabitEthernet0/0"))
    assert "NO_DEFAULT_ROUTE" in ids(f)


def test_vlan_inactive_on_access_port():
    f = RC.run(case("SW1# show interfaces Fa0/1 switchport\n"
                    "Access Mode VLAN: 30 (Inactive)"))
    assert "VLAN_NOT_IN_DATABASE" in ids(f)


def test_subinterface_without_encapsulation():
    f = RC.run(case(
        "R1# show running-config | section interface\n"
        "interface GigabitEthernet0/0.20\n"
        " ip address 192.168.20.1 255.255.255.0\n"))
    assert "SUBIF_NO_ENCAPSULATION" in ids(f)


def test_static_route_to_unreachable_next_hop():
    f = RC.run(case(
        "R1# show ip route\n"
        "C       10.0.0.0/30 is directly connected, GigabitEthernet0/2\n\n"
        "R1# show running-config | include ip route\n"
        "ip route 192.168.60.0 255.255.255.0 10.0.0.6"))
    assert "NEXT_HOP_UNREACHABLE" in ids(f)


def test_every_finding_quotes_a_real_line():
    """No finding may cite text that is not present in the source evidence."""
    with open(ROOT / "data" / "cases.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        haystack = " ".join(row["show_outputs"].split()).lower()
        for f in RC.run(row):
            if f.check_id == "CHECK_ERROR":
                continue
            for ev in f.evidence:
                assert " ".join(ev.split()).lower() in haystack, \
                    f"{row['case_id']} {f.check_id} cited a line that is not in the output: {ev!r}"


def test_no_check_raises_on_garbage():
    for junk in ("", "   ", "not cli output at all", "R1#\n\n\n", "%%%%", "R1# show ip route\n"):
        assert all(f.check_id != "CHECK_ERROR" for f in RC.run(case(junk)))


# ======================================================================================
# REGRESSIONS — each maps to an entry in data/responsible_ai_log.csv
# ======================================================================================

def test_regression_rai01_default_vlan1_is_not_a_fault():
    """RAI-01: a shut, unassigned Vlan1 is the Cisco default and must not be reported."""
    f = RC.run(case(
        "SW1# show ip interface brief\n"
        "FastEthernet0/1        unassigned      YES unset  up                    up\n"
        "Vlan1                  unassigned      YES unset  administratively down down"))
    assert "IFACE_ADMIN_DOWN" not in ids(f)


def test_regression_rai02_nat_pool_is_not_a_dhcp_fault():
    """RAI-02: 'allocated 8 (100%)' in NAT output must not be read as DHCP exhaustion."""
    f = RC.run(case(
        "R1# show ip nat statistics\n"
        "Outside interfaces:\n  GigabitEthernet0/1\n"
        "Inside interfaces:\n  GigabitEthernet0/0\n"
        "        type generic, total addresses 8, allocated 8 (100%), misses 1276"))
    assert "DHCP_POOL_EXHAUSTED" not in ids(f)


def test_regression_rai03_documentation_range_is_not_inside_space():
    """RAI-03: 203.0.113.0/24 is WAN space in Packet Tracer, not an un-NATed LAN."""
    f = RC.run(case(
        "R1# show ip interface brief\n"
        "GigabitEthernet0/0.10      192.168.10.1    YES manual up                    up\n"
        "GigabitEthernet0/1         203.0.113.2     YES manual up                    up\n\n"
        "R1# show running-config | section nat\n"
        "ip nat inside source list NAT-LAN interface GigabitEthernet0/1 overload\n"
        "ip access-list standard NAT-LAN\n"
        " permit 192.168.10.0 0.0.0.255"))
    assert "NAT_ACL_NOT_COVERING_LAN" not in ids(f)


def test_regression_rai07_empty_nat_interface_list_is_detected():
    """RAI-07: an empty list under a header must be seen as empty."""
    f = RC.run(case(
        "R1# show ip nat statistics\n"
        "Total translations: 0 (0 static, 0 dynamic, 0 extended)\n"
        "Outside interfaces:\n"
        "Inside interfaces:\n"
        "Hits: 0  Misses: 0"))
    assert "NAT_INTERFACES_MISSING" in ids(f)


def test_populated_nat_interface_list_is_clean():
    f = RC.run(case(
        "R1# show ip nat statistics\n"
        "Outside interfaces:\n  GigabitEthernet0/1\n"
        "Inside interfaces:\n  GigabitEthernet0/0.10\n"
        "Hits: 4821  Misses: 12"))
    assert "NAT_INTERFACES_MISSING" not in ids(f)


# ======================================================================================
# AI engine
# ======================================================================================

def test_extract_json_from_fenced_and_chatty_output():
    obj, repaired = extract_json('Here you go:\n```json\n{"root_cause": "x"}\n```\nHope that helps!')
    assert obj == {"root_cause": "x"} and repaired


def test_extract_json_tolerates_trailing_comma():
    obj, _ = extract_json('{"root_cause": "x", "severity": "High",}')
    assert obj["severity"] == "High"


def test_diagnosis_coerces_sloppy_model_output():
    d = Diagnosis(root_cause="x", confidence="HIGH", osi_layer="Layer 3", concept_tag="ACLs",
                  evidence="line one | line two", next_command="show ip route\nshow vlan brief",
                  fix_steps=["a", "b"], severity="critical")
    assert (d.confidence, d.osi_layer, d.concept_tag) == ("high", "L3", "acl")
    assert d.evidence == ["line one", "line two"]
    assert d.next_command == "show ip route"      # one command only
    assert d.severity == "Critical"


def test_diagnosis_falls_back_on_nonsense_values():
    d = Diagnosis(root_cause="x", confidence="banana", osi_layer="L99", concept_tag="quantum")
    assert (d.confidence, d.osi_layer, d.concept_tag) == ("medium", "L3", "addressing")


def test_ungrounded_evidence_can_never_score_correct():
    engine = DiagnosisEngine.from_env("mock")
    c = case("R1# show ip route\nGateway of last resort is not set",
             concept_tag="static_routing", osi_layer="L3")
    res = engine.diagnose(c)
    res.diagnosis.evidence = ["a line that was never in the output"]
    g = res.grade(c)
    assert g["evidence_grounded"] is False
    assert g["ai_correct"] is False


def test_mock_engine_runs_every_case_and_grounds_its_evidence():
    with open(ROOT / "data" / "cases.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    engine = DiagnosisEngine.from_env("mock")
    grounded = 0
    for row in rows:
        res = engine.diagnose(row)
        assert res.diagnosis.root_cause
        assert res.diagnosis.fix_steps
        grounded += res.evidence_is_grounded(row)
    assert grounded == len(rows), f"only {grounded}/{len(rows)} grounded"


# ======================================================================================
# Review manager
# ======================================================================================

def test_review_round_trip_and_latest_wins(tmp_path=None):
    path = ROOT / "data" / "_pytest_reviews.csv"
    mgr = ReviewManager(ReviewStore(path))
    mgr.store.clear()
    engine = DiagnosisEngine.from_env("mock")
    c = case("R1# show ip route\nGateway of last resort is not set", case_id="T-01",
             concept_tag="static_routing")
    res = engine.diagnose(c)

    mgr.accept("T-01", res, reviewer="alice")
    assert mgr.store.latest()["T-01"]["decision"] == "Accepted"

    mgr.reject("T-01", res, "actually an ACL problem", "acl", "L4", reviewer="bob")
    latest = mgr.store.latest()["T-01"]
    assert latest["decision"] == "Rejected" and latest["reviewer"] == "bob"
    assert len(mgr.store.for_case("T-01")) == 2      # history is kept

    m = mgr.metrics([c], [res])
    assert m.reviewed == 1 and m.rejected == 1 and m.agreement_rate == 0.0
    path.unlink(missing_ok=True)


def test_regression_empty_store_is_not_swapped_for_the_default():
    """An empty ReviewStore is falsy (it defines __len__). ReviewManager must still honour
    the store it was handed, or a test run would write into the real data/reviews.csv."""
    path = ROOT / "data" / "_pytest_isolated.csv"
    store = ReviewStore(path)
    store.clear()
    assert len(store) == 0 and not store          # empty store really is falsy
    mgr = ReviewManager(store)
    assert mgr.store.path == path
    path.unlink(missing_ok=True)


def test_metrics_are_zero_safe():
    path = ROOT / "data" / "_pytest_empty.csv"
    mgr = ReviewManager(ReviewStore(path))
    mgr.store.clear()
    m = mgr.metrics([], [])
    assert m.agreement_rate == 0.0 and m.auto_accuracy == 0.0 and m.summary_lines()
    path.unlink(missing_ok=True)


# ======================================================================================
# Dataset integrity
# ======================================================================================

def test_dataset_is_complete_and_covers_every_family():
    with open(ROOT / "data" / "cases.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 30
    assert len({r["case_id"] for r in rows}) == len(rows)
    required = {"vlan", "trunking", "inter_vlan_routing", "dhcp", "dns", "static_routing",
                "ospf", "nat", "acl", "wireless", "addressing"}
    assert required <= {r["concept_tag"] for r in rows}
    for r in rows:
        for col in ("symptom", "topology_note", "show_outputs", "expected_fault",
                    "expected_root_cause", "osi_layer", "severity", "expected_next_command",
                    "expected_fix_steps"):
            assert r[col].strip(), f"{r['case_id']} has an empty {col}"
        assert r["osi_layer"] in {"L1", "L2", "L3", "L4", "L7"}
        assert r["severity"] in {"Critical", "High", "Medium", "Low"}


def test_every_case_yields_at_least_one_finding():
    with open(ROOT / "data" / "cases.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bare = [r["case_id"] for r in rows
            if not [f for f in RC.run(r) if f.check_id != "CHECK_ERROR"]]
    assert not bare, f"cases with no deterministic finding: {bare}"


def test_responsible_ai_log_has_at_least_five_entries():
    with open(ROOT / "data" / "responsible_ai_log.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 5
    for r in rows:
        for col in ("exact_flaw", "human_correction", "guardrail_added", "reviewer_decision"):
            assert r[col].strip(), f"{r['log_id']} has an empty {col}"
        assert r["reviewer_decision"] in {"Accepted", "Edited", "Rejected"}


def test_check_registry_has_no_duplicates():
    names = [c.__name__ for c in CHECKS]
    assert len(names) == len(set(names))


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
