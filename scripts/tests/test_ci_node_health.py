"""Tests for scripts/ci_node_health.py.

Everything here runs without network: Prometheus and GitHub are replaced by
fakes that return canned API payloads.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci_node_health.py"


def _load():
    spec = importlib.util.spec_from_file_location("ci_node_health", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_node_health"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


NOW = datetime(2026, 9, 20, 6, 17, tzinfo=UTC)
H100 = {"10.0.121.183", "10.0.98.28"}


def _sample(metric: dict, value: float) -> dict:
    return {"metric": metric, "value": [NOW.timestamp(), str(value)]}


class FakeProm:
    def __init__(self, table: dict[str, list[dict]]):
        self.table = table
        self.queries: list[str] = []

    def query(self, promql: str) -> list[dict]:
        self.queries.append(promql)
        return self.table.get(promql, [])


def _promql(mod, key: str) -> str:
    return next(pc.promql for pc in mod.PROM_CHECKS if pc.check.key == key)


# --- node_of ---------------------------------------------------------------


def test_node_of_prefers_node_then_hostname_then_instance(mod):
    assert mod.node_of({"node": "10.0.1.1", "Hostname": "x"}) == "10.0.1.1"
    assert mod.node_of({"Hostname": "10.0.1.2"}) == "10.0.1.2"
    assert mod.node_of({"kubernetes_node": "10.0.1.4"}) == "10.0.1.4"
    assert mod.node_of({"instance": "10.0.1.3:9100"}) == "10.0.1.3"
    assert mod.node_of({}) == ""


# --- Prometheus evaluators ---------------------------------------------------


def test_h100_nodes_uses_gpu_capacity(mod):
    prom = FakeProm(
        {
            mod.H100_NODES_QUERY: [
                _sample({"node": "10.0.121.183"}, 8),
                _sample({"node": "10.0.98.28"}, 8),
            ]
        }
    )
    assert mod.h100_nodes(prom) == H100


def test_prom_findings_filters_to_h100_nodes(mod):
    q = _promql(mod, "node_cordoned")
    prom = FakeProm(
        {
            q: [
                _sample({"node": "10.0.98.28"}, 1),
                _sample({"node": "10.0.108.220"}, 1),  # an A10 node: ignored
            ]
        }
    )
    findings = mod.prom_findings(prom, H100)
    assert [(f.check.key, f.scope) for f in findings] == [("node_cordoned", "10.0.98.28")]


def test_prom_findings_detail_uses_metric_labels(mod):
    prom = FakeProm(
        {
            _promql(mod, "npd_condition"): [
                _sample({"node": "10.0.121.183", "condition": "GpuEcc"}, 1)
            ],
            _promql(mod, "gpu_xid"): [_sample({"Hostname": "10.0.121.183", "gpu": "3"}, 2)],
            _promql(mod, "disk_full"): [
                _sample({"instance": "10.0.121.183:9100", "mountpoint": "/"}, 0.91)
            ],
        }
    )
    details = {f.check.key: f.detail for f in mod.prom_findings(prom, H100)}
    assert details["npd_condition"] == "GpuEcc"
    assert details["gpu_xid"] == "gpu3: 2 XID change(s) in the last hour"
    assert details["disk_full"] == "/ 91% used"


def test_prom_findings_empty_when_healthy(mod):
    assert mod.prom_findings(FakeProm({}), H100) == []


def test_every_prom_check_has_a_registered_check(mod):
    for pc in mod.PROM_CHECKS:
        assert mod.CHECKS[pc.check.key] is pc.check
        assert pc.check.severity in {"CRIT", "WARN"}


def test_prom_query_raises_prom_error_on_failure(mod, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    with pytest.raises(mod.PromError):
        mod.Prom("http://127.0.0.1:1").query("up")
