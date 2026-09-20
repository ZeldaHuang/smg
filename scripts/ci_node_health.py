#!/usr/bin/env python3
"""Hourly health monitor for the H100 CI nodes.

Reads Prometheus (node-problem-detector, DCGM, node-exporter, kube-state-metrics)
and the GitHub Actions API, evaluates the checks from
docs/superpowers/specs/2026-09-20-h100-ci-node-health-design.md, and keeps one
GitHub issue per active check under the ``ci-node-health`` label. Slack is fed by
the GitHub Slack app subscribed to that label, so the only notifications are
"issue opened" and "issue closed".

Runs on the in-cluster ``k8s-runner-cpu`` runner, which reaches Prometheus over
its ClusterIP without auth. Needs no kubeconfig and no secret beyond GITHUB_TOKEN.

Usage:
    ci_node_health.py [--dry-run] [--prom-url URL] [--repo OWNER/NAME]

Environment:
    GITHUB_TOKEN         required to write issues (reads work without it)
    GITHUB_REPOSITORY    default for --repo
    PROM_URL             default for --prom-url
    GITHUB_STEP_SUMMARY  when set, the fleet summary is appended there
    GITHUB_SERVER_URL, GITHUB_REPOSITORY, GITHUB_RUN_ID  build the run link
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

ISSUE_LABEL = "ci-node-health"
H100_LABELS = ("1-gpu-h100", "2-gpu-h100", "4-gpu-h100")
CLOSE_AFTER = timedelta(minutes=90)
COMMENT_EVERY = timedelta(hours=24)
QUEUE_WINDOW = timedelta(hours=2)
STARVED_AFTER = timedelta(minutes=60)
STALE_RUN_AFTER = timedelta(hours=24)
QUEUE_WAIT_P50_MIN = 30.0
MARKER_PREFIX = "<!-- ci-node-health "
MARKER_SUFFIX = " -->"
SPEC = "docs/superpowers/specs/2026-09-20-h100-ci-node-health-design.md"
DEFAULT_PROM_URL = "http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090"


# --- check registry ----------------------------------------------------------


@dataclass(frozen=True)
class Check:
    key: str
    severity: str  # "CRIT" or "WARN"
    title: str
    meaning: str
    first_action: str
    event: bool = False  # event findings never auto-close


@dataclass(frozen=True)
class Finding:
    check: Check
    scope: str  # node name, runner label, or "github"
    detail: str


CHECKS: dict[str, Check] = {
    c.key: c
    for c in (
        Check(
            "node_not_ready",
            "CRIT",
            "H100 node not Ready",
            "The kubelet stopped reporting. Every runner pod on the node is dead and its "
            "jobs will time out.",
            "`kubectl describe node <node>`. If it does not recover in 15 minutes, cordon it "
            "and open an OCI ticket with the serial from the "
            "`oci.oraclecloud.com/host.serial_number` label.",
        ),
        Check(
            "npd_condition",
            "CRIT",
            "NPD hardware condition",
            "node-problem-detector's OKE GPU/RDMA plugin flagged the node. NPD only sets the "
            "condition; nothing cordons the node on its own.",
            "`kubectl cordon <node>`, read the condition message in `kubectl describe node`, "
            "and open an OCI hardware ticket.",
        ),
        Check(
            "gpu_xid",
            "CRIT",
            "GPU XID error",
            "The NVIDIA driver logged an XID on this GPU in the last hour. Lanes on it may "
            "fail or hang. This issue is closed by a human, not by the monitor.",
            "`dmesg | grep -i xid` on the node. XID 79, 48, 95 mean reset or RMA: cordon "
            "first. Close this issue after the GPU is reset or the XID is confirmed benign.",
            event=True,
        ),
        Check(
            "gpu_row_remap_failure",
            "CRIT",
            "GPU row-remap failure",
            "DCGM reports a failed HBM row remap. The GPU needs a reset or an RMA.",
            "Cordon the node and open an OCI hardware ticket.",
        ),
        Check(
            "disk_full",
            "CRIT",
            "Node disk over 85%",
            "Root (runner emptyDirs and dind storage) or /raid (model cache) is nearly full. "
            "The kubelet starts evicting pods at 85 to 90%.",
            "Root: delete finished runner pods and run `crictl rmi --prune` on the node. "
            "/raid: prune `hub/` snapshots that are not in `e2e_test/infra/model_specs.py`.",
        ),
        Check(
            "runner_starved",
            "CRIT",
            "GPU runner label starved",
            "A job has waited over an hour for a *-h100 runner, or no runner for that label "
            "is registered online. PR #2283 is what this looks like when nobody notices.",
            "`kubectl get autoscalingrunnersets -n actions-runner-system`, then the listener "
            "pod logs and `kubectl get events -n actions-runner-system | grep FailedScheduling`.",
        ),
        Check(
            "monitor_blind",
            "CRIT",
            "Monitor cannot reach Prometheus",
            "Prometheus did not answer, so every node check was skipped this run.",
            "`kubectl -n monitoring get pods | grep prometheus`. The monitor recovers on its own.",
        ),
        Check(
            "node_cordoned",
            "WARN",
            "H100 node cordoned",
            "The node has been unschedulable for over two hours. Its 8 GPUs are out of the CI "
            "pool while 4-GPU lanes queue.",
            "If intentional, assign this issue to yourself and leave it open. Otherwise "
            "`kubectl uncordon <node>`.",
        ),
        Check(
            "queue_wait",
            "WARN",
            "GPU jobs waiting for runners",
            "The median wait for a *-h100 runner exceeded 30 minutes over the last two hours.",
            "Check scale-set occupancy in the run summary. Free GPUs (cordoned nodes, stuck "
            "runner pods) or raise `maxRunners` in `scripts/k8s-runner-resources/`.",
        ),
        Check(
            "npd_unknown",
            "WARN",
            "NPD condition Unknown",
            "The node-problem-detector plugin timed out or refused to run for most of the last "
            "six hours, so a real fault on this node would go unnoticed.",
            "`kubectl -n monitoring logs <npd pod on the node> | grep -iE 'timeout|not RDMA'`.",
        ),
        Check(
            "gpu_mem_leak",
            "WARN",
            "GPU memory held by no pod",
            "A GPU has had over 2 GiB allocated with no pod attached for 30 minutes: an orphan "
            "process left by a killed runner.",
            "`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` on the node and "
            "kill the orphan.",
        ),
        Check(
            "gpu_hot",
            "WARN",
            "GPU over 85C",
            "Sustained temperature near the H100 throttle point.",
            "Check clocks and power with DCGM on the node. Open an OCI ticket if it persists "
            "while idle.",
        ),
        Check(
            "stale_queued_runs",
            "WARN",
            "Workflow runs queued over 24h",
            "Runs that will never be picked up (dead runner label, offline runner) sit as "
            "queued and later report `cancelled`, which hides the real cause.",
            "`gh run cancel <id>`; fix the `runs-on` label if the runner no longer exists.",
        ),
    )
}


# --- Prometheus --------------------------------------------------------------


class PromError(RuntimeError):
    """Prometheus was unreachable or rejected the query."""


class Prom:
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def query(self, promql: str) -> list[dict]:
        data = urllib.parse.urlencode({"query": promql}).encode()
        req = urllib.request.Request(f"{self.url}/api/v1/query", data=data)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except (OSError, ValueError) as exc:
            raise PromError(f"prometheus query failed: {exc}") from exc
        if payload.get("status") != "success":
            raise PromError(f"prometheus error: {payload.get('error', payload)}")
        return payload["data"]["result"]


def node_of(metric: dict) -> str:
    """Node name from whichever label the exporter used."""
    for key in ("node", "Hostname", "kubernetes_node"):
        if metric.get(key):
            return metric[key]
    instance = metric.get("instance", "")
    return instance.rsplit(":", 1)[0] if ":" in instance else instance


@dataclass(frozen=True)
class PromCheck:
    check: Check
    promql: str
    detail: Callable[[dict, float], str]


H100_NODES_QUERY = 'max by (node) (kube_node_status_capacity{resource="nvidia_com_gpu"}) == 8'

NPD_CONDITIONS = (
    "GpuEcc|GpuRowRemap|GpuBus|GpuCount|RdmaLink|RdmaLinkFlapping|RdmaWpaAuth|RdmaRttcc|"
    "KernelDeadlock|ReadonlyFilesystem"
)

PROM_CHECKS: tuple[PromCheck, ...] = (
    PromCheck(
        CHECKS["node_not_ready"],
        'max by (node) (kube_node_status_condition{condition="Ready",status="true"}) == 0',
        lambda m, v: "Ready is not True",
    ),
    PromCheck(
        CHECKS["npd_condition"],
        "max by (node, condition) (kube_node_status_condition"
        f'{{condition=~"{NPD_CONDITIONS}",status="true"}}) == 1',
        lambda m, v: m["condition"],
    ),
    PromCheck(
        CHECKS["gpu_xid"],
        "max by (Hostname, gpu) (changes(DCGM_FI_DEV_XID_ERRORS[1h])) > 0",
        lambda m, v: f"gpu{m['gpu']}: {int(v)} XID change(s) in the last hour",
    ),
    PromCheck(
        CHECKS["gpu_row_remap_failure"],
        "max by (Hostname, gpu) (DCGM_FI_DEV_ROW_REMAP_FAILURE) > 0",
        lambda m, v: f"gpu{m['gpu']}: row-remap failure",
    ),
    PromCheck(
        CHECKS["disk_full"],
        "max by (instance, mountpoint) (1 - "
        'node_filesystem_avail_bytes{mountpoint=~"/|/raid"} / '
        'node_filesystem_size_bytes{mountpoint=~"/|/raid"}) > 0.85',
        lambda m, v: f"{m['mountpoint']} {v * 100:.0f}% used",
    ),
    PromCheck(
        CHECKS["node_cordoned"],
        "max by (node) (min_over_time(kube_node_spec_unschedulable[2h])) == 1",
        lambda m, v: "unschedulable for over 2h",
    ),
    PromCheck(
        CHECKS["npd_unknown"],
        "max by (node, condition) "
        '(avg_over_time(kube_node_status_condition{status="unknown"}[6h])) > 0.5',
        lambda m, v: f"{m['condition']} Unknown {v * 100:.0f}% of the last 6h",
    ),
    PromCheck(
        CHECKS["gpu_mem_leak"],
        'max by (Hostname, gpu) (min_over_time(DCGM_FI_DEV_FB_USED{pod=""}[30m])) > 2048',
        lambda m, v: f"gpu{m['gpu']}: {v / 1024:.1f} GiB used with no pod for 30m",
    ),
    PromCheck(
        CHECKS["gpu_hot"],
        "max by (Hostname, gpu) (min_over_time(DCGM_FI_DEV_GPU_TEMP[15m])) > 85",
        lambda m, v: f"gpu{m['gpu']}: {v:.0f}C for 15m",
    ),
)


def h100_nodes(prom: Prom) -> set[str]:
    return {node_of(s["metric"]) for s in prom.query(H100_NODES_QUERY)}


def prom_findings(prom: Prom, nodes: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for pc in PROM_CHECKS:
        for sample in prom.query(pc.promql):
            node = node_of(sample["metric"])
            if node not in nodes:
                continue
            value = float(sample["value"][1])
            findings.append(Finding(pc.check, node, pc.detail(sample["metric"], value)))
    return findings


def main(argv: list[str] | None = None) -> int:
    raise SystemExit("main is implemented in Task 4")


if __name__ == "__main__":
    sys.exit(main())
