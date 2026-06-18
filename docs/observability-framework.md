# Observability Framework

This project uses two complementary observability pipelines for the AIOps
runtime.

## Metrics Pipeline

Prometheus scrapes Kubernetes and service metrics, Grafana visualizes the
metrics, and Alertmanager receives alerts from Prometheus rules.

```text
Microservices / Kubernetes Pods
        |
        | metrics scraping
        v
Prometheus
        | dashboard queries
        +------------------> Grafana
        | firing alerts
        +------------------> Alertmanager
        | metric queries
        +------------------> Anomaly Service / RCA Service / Dashboard
```

The default manifest is intentionally lightweight and self-contained. It can
scrape annotated pods and services, and it also includes static targets for the
AIOps runtime services in the `aiops-dev` namespace.

## Tracing Pipeline

Instrumented services export OTLP traces to OpenTelemetry Collector. The
collector forwards traces to Jaeger, where operators can inspect distributed
requests and the RCA pipeline can query service dependencies.

```text
Instrumented Microservices
        |
        | OTLP traces
        v
OpenTelemetry Collector
        |
        | OTLP export
        v
Jaeger
        |
        | trace queries
        +------------------> Orchestrator / RCA Service / Dashboard
```

## Deploy

Deploy Jaeger, Prometheus, Grafana, and Alertmanager:

```powershell
kubectl apply -k observability
```

Deploy the application-level OpenTelemetry Collector if the monitored workload
expects the collector service in the `app` namespace:

```powershell
kubectl apply -f observability\otel-collector-app.yaml
```

If Online Boutique is deployed in the `default` namespace and currently sends
traces directly to Jaeger, patch it to send OTLP traces through the collector:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\patch_online_boutique_otel_collector.ps1 -Namespace default -Restart
```

If Online Boutique is deployed through the staging overlay, the overlay already
sets `COLLECTOR_SERVICE_ADDR` to
`opentelemetrycollector.app.svc.cluster.local:4317`.

## Verify

```powershell
kubectl get pods -n observability
kubectl get pods -n monitoring
kubectl get svc -n observability
kubectl get svc -n monitoring
```

Port-forward the UIs:

```powershell
kubectl port-forward -n observability svc/jaeger 16686:16686
kubectl port-forward -n monitoring svc/prometheus 9090:9090
kubectl port-forward -n monitoring svc/aiops-grafana 3000:3000
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
```

Open:

- Jaeger: http://127.0.0.1:16686
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000
- Alertmanager: http://127.0.0.1:9093

Grafana default credentials are `admin` / `admin`.

## Paper Text

The observability layer combines Prometheus, Grafana, and Alertmanager for
metric-based monitoring, and OpenTelemetry Collector with Jaeger for
distributed tracing. Prometheus scrapes service and Kubernetes metrics, Grafana
visualizes operational dashboards, and Alertmanager handles rule-based alert
routing. In parallel, OpenTelemetry Collector receives trace data from
instrumented services and forwards them to Jaeger. The AIOps pipeline consumes
both metrics and traces to support anomaly detection and graph-based RCA.
