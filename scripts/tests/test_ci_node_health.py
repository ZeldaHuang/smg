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


# --- GitHub evaluators -------------------------------------------------------


class FakeGitHub:
    """Serves canned GET payloads keyed by path; records writes."""

    def __init__(self, table: dict[str, object]):
        self.table = table
        self.writes: list[tuple[str, str, dict]] = []

    def get(self, path: str, params: dict | None = None):
        page = int((params or {}).get("page", 1))
        payload = self.table.get(path, [])
        if isinstance(payload, dict):
            key = next(k for k in ("workflow_runs", "jobs", "runners") if k in payload)
            items = payload[key] if page == 1 else []
            return {key: items}
        return payload if page == 1 else []

    def paginate(self, path, key, params=None, max_pages=3):
        payload = self.get(path, dict(params or {}, page=1))
        return payload[key] if key else payload

    def post(self, path, body):
        self.writes.append(("POST", path, body))
        return {"number": 999, "html_url": "https://github.com/x/y/issues/999"}

    def patch(self, path, body):
        self.writes.append(("PATCH", path, body))
        return {}


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _job(name, label, created, started=None, status="completed", url="https://j"):
    return {
        "name": name,
        "labels": [label],
        "created_at": _ts(created),
        "started_at": _ts(started) if started else None,
        "status": status,
        "html_url": url,
    }


def _runners(*names):
    return {"runners": [{"name": n, "status": "online", "labels": []} for n in names]}


ALL_ONLINE = _runners(
    "1-gpu-h100-abc-runner-1", "2-gpu-h100-abc-runner-1", "4-gpu-h100-abc-runner-1"
)


def test_parse_ts_reads_github_timestamps(mod):
    assert mod.parse_ts("2026-09-20T05:22:31Z") == datetime(2026, 9, 20, 5, 22, 31, tzinfo=UTC)


def test_queue_wait_uses_job_created_to_started(mod):
    gh = FakeGitHub(
        {
            "actions/runs": {
                "workflow_runs": [{"id": 1, "name": "PR Test", "created_at": _ts(NOW)}]
            },
            "actions/runs/1/jobs": {
                "jobs": [
                    _job(
                        "e2e-4gpu / run",
                        "4-gpu-h100",
                        NOW - timedelta(minutes=90),
                        NOW - timedelta(minutes=20),
                    ),
                    _job(
                        "e2e-1gpu / run",
                        "1-gpu-h100",
                        NOW - timedelta(minutes=60),
                        NOW - timedelta(minutes=10),
                    ),
                    _job(
                        "pre-commit",
                        "k8s-runner-cpu",
                        NOW - timedelta(minutes=60),
                        NOW - timedelta(minutes=59),
                    ),
                ]
            },
            "actions/runners": ALL_ONLINE,
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert sorted(stats.waits_min) == [50.0, 70.0]  # the CPU job is not counted
    assert [f.check.key for f in findings] == ["queue_wait"]
    assert findings[0].detail.startswith("p50 60 min")


def test_queued_h100_job_over_an_hour_is_starved(mod):
    gh = FakeGitHub(
        {
            "actions/runs": {
                "workflow_runs": [{"id": 1, "name": "PR Test", "created_at": _ts(NOW)}]
            },
            "actions/runs/1/jobs": {
                "jobs": [
                    _job(
                        "e2e-4gpu / run",
                        "4-gpu-h100",
                        NOW - timedelta(minutes=75),
                        status="queued",
                    )
                ]
            },
            "actions/runners": ALL_ONLINE,
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert stats.queued_h100 == 1
    assert [(f.check.key, f.scope) for f in findings] == [("runner_starved", "4-gpu-h100")]
    assert "queued 75 min" in findings[0].detail


def test_label_with_no_online_runner_is_starved(mod):
    gh = FakeGitHub(
        {
            "actions/runs": {"workflow_runs": []},
            "actions/runners": _runners("1-gpu-h100-abc-runner-1", "4-gpu-h100-abc-runner-1"),
        }
    )
    findings, stats = mod.github_findings(gh, NOW)
    assert stats.online_runners == {"1-gpu-h100": 1, "2-gpu-h100": 0, "4-gpu-h100": 1}
    assert [(f.check.key, f.scope, f.detail) for f in findings] == [
        ("runner_starved", "2-gpu-h100", "no online runner registered")
    ]


def test_stale_queued_runs_older_than_a_day(mod):
    old = NOW - timedelta(days=3)
    fresh = NOW - timedelta(hours=2)
    gh = FakeGitHub(
        {
            "actions/runs": {
                "workflow_runs": [
                    {
                        "id": 5,
                        "name": "Nightly tau2",
                        "created_at": _ts(old),
                        "html_url": "https://r/5",
                    },
                    {
                        "id": 6,
                        "name": "PR Test",
                        "created_at": _ts(fresh),
                        "html_url": "https://r/6",
                    },
                ]
            },
            "actions/runs/5/jobs": {"jobs": []},
            "actions/runs/6/jobs": {"jobs": []},
            "actions/runners": ALL_ONLINE,
        }
    )
    findings, _ = mod.github_findings(gh, NOW)
    stale = [f for f in findings if f.check.key == "stale_queued_runs"]
    assert len(stale) == 1
    assert stale[0].scope == "github"
    assert "Nightly tau2 run 5 queued 3d" in stale[0].detail
