# AIOps feedback retrain workflow

Thư mục này chứa skeleton cho luồng retrain MLOps bằng Argo Workflows.

## Có gì ở đây

- `aiops-feedback-retrain-workflowtemplate.yaml`
  - export feedback snapshot từ dashboard state
  - train anomaly model
  - train RCA model
- `aiops-feedback-retrain-cronworkflow.yaml`
  - ví dụ chạy theo lịch hàng tuần
  - mặc định `suspend: true`

## Lưu ý thực tế

Workflow này là skeleton để bạn hoàn thiện dần. Trước khi chạy thật trong cluster, cần chốt thêm:

1. image tags cụ thể cho `dashboard`, `anomaly-service`, `rca-service`
2. cách mount dashboard state DB hoặc feedback snapshot
3. data roots dùng cho retrain trong container/job
4. bước promote model và commit registry/model artifacts nếu bạn giữ flow GitOps hiện tại

## Apply

```powershell
kubectl apply -k deploy\mlops\workflows
```

## Chạy tay một workflow

```powershell
argo submit --from workflowtemplate/aiops-feedback-retrain -n mlops `
  -p system-id=online-boutique `
  -p mlflow-tracking-uri=http://mlflow.mlops.svc.cluster.local:5000
```

## Export feedback snapshot ngoài workflow

Nếu muốn export feedback trên máy host trước, dùng:

```powershell
python tools\export_feedback_training_set.py `
  --output artifacts\feedback\feedback_training_snapshot.jsonl `
  --system-id online-boutique
```
