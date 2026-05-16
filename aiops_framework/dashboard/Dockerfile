FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV AIOPS_DASHBOARD_GRAPH_ROOT=/app/data_rca_balanced_v3/processed/rca/graph_tensors

COPY aiops_framework/dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY aiops_framework /app/aiops_framework
COPY data_rca_balanced_v3/processed/rca/graph_tensors /app/data_rca_balanced_v3/processed/rca/graph_tensors

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "aiops_framework.dashboard.app:app", "--host", "0.0.0.0", "--port", "8010"]
