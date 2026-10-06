import subprocess

import pytest

from evaluation.run_labeled_rca_scenario import ScaleToZeroFault, normalize_service, summarize


def test_scale_fault_restores_original_replicas_after_failure():
    commands = []

    def runner(command, timeout=None):
        commands.append(command)
        stdout = "2" if "get" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    fault = ScaleToZeroFault(
        ["kubectl"],
        "sock-shop",
        "payment",
        30,
        runner=runner,
    )

    with pytest.raises(RuntimeError, match="analysis failed"):
        with fault:
            raise RuntimeError("analysis failed")

    assert ["kubectl", "-n", "sock-shop", "scale", "deployment/payment", "--replicas=0"] in commands
    assert ["kubectl", "-n", "sock-shop", "scale", "deployment/payment", "--replicas=2"] in commands
    assert ["kubectl", "-n", "sock-shop", "rollout", "status", "deployment/payment", "--timeout=30s"] in commands


def test_summary_scores_ground_truth_metrics():
    records = [
        {
            "scenario_id": "payment-down",
            "system_id": "sock_shop",
            "expected_root_cause": "payment",
            "success": True,
            "duration_ms": 1000,
            "predicted_is_anomaly": True,
            "top1_correct": True,
            "top3_hit": True,
        },
        {
            "scenario_id": "payment-down",
            "system_id": "sock_shop",
            "expected_root_cause": "payment",
            "success": True,
            "duration_ms": 2000,
            "predicted_is_anomaly": False,
            "top1_correct": False,
            "top3_hit": True,
        },
    ]

    result = summarize(records, {"attempted": 4, "successful": 1, "failed": 3})

    assert result["anomaly_detection_rate"] == 0.5
    assert result["rca_top1_accuracy"] == 0.5
    assert result["rca_top3_hit_rate"] == 1.0
    assert result["mean_ms"] == 1500.0


def test_service_normalization_handles_model_and_kubernetes_names():
    assert normalize_service("payment_service") == "payment-service"
    assert normalize_service(" Payment-Service ") == "payment-service"
