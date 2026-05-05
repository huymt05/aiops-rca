param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [string]$GraphDatasetPath = "D:\doanchuyennganh_aiops\FINAL_SYSTEM\aiops_rca_benchmark\outputs\re2_ob\common\graph_dataset.pt",
    [string]$CaseId = "",
    [int]$CaseIndex = 0,
    [string]$PythonExe = "D:\doanchuyennganh_aiops\doan_env\Scripts\python.exe"
)

$script = @"
import json
import sys
from pathlib import Path
import torch

graph_path = Path(r"$GraphDatasetPath")
case_id = r"$CaseId".strip()
case_index = int("$CaseIndex")
graphs = torch.load(graph_path, map_location="cpu")
if case_id:
    selected = None
    for item in graphs:
        if str(item.get("case_id")) == case_id:
            selected = item
            break
    if selected is None:
        raise SystemExit(f"Case not found: {case_id}")
else:
    if case_index < 0 or case_index >= len(graphs):
        raise SystemExit(f"Case index out of range: {case_index}")
    selected = graphs[case_index]

edge_rows = selected["edge_index"].t().tolist() if selected["edge_index"].numel() > 0 else []
payload = {
    "graph_id": str(selected["case_id"]),
    "node_names": list(selected["services"]),
    "x": selected["x"].tolist(),
    "edge_index": edge_rows,
    "top_k": 3,
    "metadata": {
        "fault_type": str(selected.get("fault_type", "")),
        "true_root_cause": str(selected.get("root_cause_service", "")),
        "source": "re2_ob_benchmark_test",
    },
}
print(json.dumps(payload, ensure_ascii=False))
"@

$body = $script | & $PythonExe -
Invoke-RestMethod -Uri "$BaseUrl/predict/graph" -Method Post -ContentType "application/json" -Body $body
