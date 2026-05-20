FROM apache/airflow:3.2.1-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY pipeline/airflow/requirements-airflow-local.txt /opt/airflow/requirements-airflow-local.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements-airflow-local.txt

COPY --chown=airflow:0 . /opt/airflow/repo

ENV PYTHONPATH="/opt/airflow/repo:/opt/airflow/repo/src:/opt/airflow/repo/pipeline:/opt/airflow/repo/pipeline/rca_data_pipeline"
ENV AIRFLOW__CORE__DAGS_FOLDER="/opt/airflow/repo/pipeline/airflow/dags"
ENV MLFLOW_TRACKING_URI="http://mlflow.mlops.svc.cluster.local:5000"
ENV AIOPS_PYTHON="/usr/local/bin/python"

USER root
RUN ln -sf /home/airflow/.local/bin/airflow /usr/local/bin/airflow
USER airflow

WORKDIR /opt/airflow/repo