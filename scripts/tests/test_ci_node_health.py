"""Tests for scripts/ci_node_health.py.

Everything here runs without network: Prometheus and GitHub are replaced by
fakes that return canned API payloads.
"""

from __future__ import annotations

import importlib.util
import json
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


# --- issue reconciliation ----------------------------------------------------


def _finding(mod, key, scope, detail="d"):
    return mod.Finding(mod.CHECKS[key], scope, detail)


def _issue(mod, number, key, first_seen, last_seen, last_comment=None, nodes=None):
    marker = {
        "key": key,
        "first_seen": first_seen.isoformat(),
        "last_seen": last_seen.isoformat(),
        "last_comment": last_comment.isoformat() if last_comment else None,
        "nodes": {n: first_seen.isoformat() for n in (nodes or [])},
    }
    return {
        "number": number,
        "body": f"old body\n\n{mod.MARKER_PREFIX}{json.dumps(marker)}{mod.MARKER_SUFFIX}",
        "title": "old title",
    }


def test_parse_issue_reads_marker_and_ignores_foreign_issues(mod):
    issue = _issue(
        mod,
        7,
        "node_cordoned",
        NOW - timedelta(hours=3),
        NOW - timedelta(hours=1),
        nodes=["10.0.98.28"],
    )
    state = mod.parse_issue(issue)
    assert state and state.number == 7 and state.key == "node_cordoned"
    assert state.nodes == {"10.0.98.28": NOW - timedelta(hours=3)}
    assert mod.parse_issue({"number": 8, "body": "a human wrote this", "title": "x"}) is None


def test_body_round_trips_through_marker(mod):
    check = mod.CHECKS["node_cordoned"]
    findings = [_finding(mod, "node_cordoned", "10.0.98.28", "unschedulable for over 2h")]
    state = mod.IssueState(0, "node_cordoned", NOW, NOW, None, {"10.0.98.28": NOW})
    body = mod.render_body(check, findings, state, NOW, "https://run/1")
    assert check.meaning in body and check.first_action in body
    assert "| 10.0.98.28 | unschedulable for over 2h |" in body
    assert "https://run/1" in body
    parsed = mod.parse_issue({"number": 1, "body": body, "title": ""})
    assert parsed and parsed.nodes == {"10.0.98.28": NOW}


def test_render_title_lists_scopes(mod):
    check = mod.CHECKS["gpu_xid"]
    fs = [
        _finding(mod, "gpu_xid", "10.0.1.2"),
        _finding(mod, "gpu_xid", "10.0.1.1"),
        _finding(mod, "gpu_xid", "10.0.1.2"),
    ]
    assert mod.render_title(check, fs) == "[node-health][CRIT] GPU XID error: 10.0.1.1, 10.0.1.2"


def test_plan_ops_creates_issue_for_new_finding(mod):
    active = mod.group_by_check([_finding(mod, "node_cordoned", "10.0.98.28")])
    ops = mod.plan_ops([], active, NOW, "https://run/1")
    assert [(o.kind, o.key) for o in ops] == [("create", "node_cordoned")]
    assert ops[0].title.startswith("[node-health][WARN] H100 node cordoned: 10.0.98.28")
    assert mod.MARKER_PREFIX in ops[0].body


def test_plan_ops_updates_open_issue_without_comment(mod):
    first = NOW - timedelta(hours=5)
    state = mod.parse_issue(
        _issue(mod, 7, "node_cordoned", first, NOW - timedelta(hours=1), nodes=["10.0.98.28"])
    )
    active = mod.group_by_check(
        [
            _finding(mod, "node_cordoned", "10.0.98.28"),
            _finding(mod, "node_cordoned", "10.0.69.208"),
        ]
    )
    ops = mod.plan_ops([state], active, NOW, "https://run/2")
    assert [(o.kind, o.number) for o in ops] == [("update", 7)]
    parsed = mod.parse_issue({"number": 7, "body": ops[0].body, "title": ""})
    assert parsed.first_seen == first  # first_seen survives updates
    assert parsed.last_seen == NOW
    assert parsed.nodes["10.0.98.28"] == first  # existing node keeps its first-seen
    assert parsed.nodes["10.0.69.208"] == NOW  # new node gets now


def test_plan_ops_closes_state_finding_only_after_two_missed_runs(mod):
    recent = mod.parse_issue(
        _issue(mod, 1, "node_cordoned", NOW - timedelta(hours=4), NOW - timedelta(minutes=60))
    )
    old = mod.parse_issue(
        _issue(mod, 2, "disk_full", NOW - timedelta(hours=4), NOW - timedelta(minutes=95))
    )
    ops = mod.plan_ops([recent, old], {}, NOW, "https://run/3")
    assert [(o.kind, o.number) for o in ops] == [("close", 2)]


def test_plan_ops_never_closes_event_findings(mod):
    xid = mod.parse_issue(
        _issue(mod, 3, "gpu_xid", NOW - timedelta(days=2), NOW - timedelta(days=2))
    )
    assert mod.plan_ops([xid], {}, NOW, "https://run/4") == []


def test_plan_ops_comments_on_event_at_most_daily(mod):
    fresh = mod.parse_issue(
        _issue(
            mod,
            3,
            "gpu_xid",
            NOW - timedelta(hours=2),
            NOW - timedelta(hours=1),
            last_comment=NOW - timedelta(hours=2),
        )
    )
    stale = mod.parse_issue(
        _issue(
            mod,
            4,
            "gpu_xid",
            NOW - timedelta(days=2),
            NOW - timedelta(days=1),
            last_comment=NOW - timedelta(days=2),
        )
    )
    active = mod.group_by_check(
        [_finding(mod, "gpu_xid", "10.0.1.1", "gpu0: 1 XID change(s) in the last hour")]
    )
    ops_fresh = mod.plan_ops([fresh], active, NOW, "https://run/5")
    assert [o.kind for o in ops_fresh] == ["update"]
    ops_stale = mod.plan_ops([stale], active, NOW, "https://run/5")
    assert [o.kind for o in ops_stale] == ["update", "comment"]
    assert "gpu0: 1 XID change(s)" in ops_stale[1].body
    parsed = mod.parse_issue({"number": 4, "body": ops_stale[0].body, "title": ""})
    assert parsed.last_comment == NOW


def test_plan_ops_ignores_issues_with_unknown_keys(mod):
    weird = mod.parse_issue(
        _issue(mod, 9, "retired_check", NOW - timedelta(days=9), NOW - timedelta(days=9))
    )
    assert mod.plan_ops([weird], {}, NOW, "https://run/6") == []


def test_apply_ops_dry_run_writes_nothing(mod, capsys):
    gh = FakeGitHub({})
    ops = [
        mod.Op("create", None, "node_cordoned", "t", "b"),
        mod.Op("close", 5, "disk_full", "", "b"),
    ]
    mod.apply_ops(gh, ops, dry_run=True)
    assert gh.writes == []
    out = capsys.readouterr().out
    assert "create" in out and "close #5" in out


def test_apply_ops_performs_writes(mod):
    gh = FakeGitHub({})
    ops = [
        mod.Op("create", None, "node_cordoned", "t", "b"),
        mod.Op("update", 7, "node_cordoned", "t2", "b2"),
        mod.Op("comment", 7, "gpu_xid", "", "c"),
        mod.Op("close", 5, "disk_full", "", "b5"),
    ]
    mod.apply_ops(gh, ops, dry_run=False)
    assert gh.writes == [
        ("POST", "issues", {"title": "t", "body": "b", "labels": [mod.ISSUE_LABEL]}),
        ("PATCH", "issues/7", {"title": "t2", "body": "b2"}),
        ("POST", "issues/7/comments", {"body": "c"}),
        ("PATCH", "issues/5", {"body": "b5", "state": "closed", "state_reason": "completed"}),
    ]
