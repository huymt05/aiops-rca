#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-sock-shop}"
ZIPKIN_HOST="${ZIPKIN_HOST:-jaeger.observability.svc.cluster.local}"
ZIPKIN_ENDPOINT="http://${ZIPKIN_HOST}:9411/api/v1/spans"

JAVA_OPTS="-Xms64m -Xmx128m -XX:+UseG1GC -Djava.security.egd=file:/dev/urandom -Dspring.zipkin.enabled=true"

for service in carts orders shipping; do
  sudo k3s kubectl set env deployment/"${service}" \
    -n "${NAMESPACE}" \
    ZIPKIN="${ZIPKIN_HOST}" \
    JAVA_OPTS="${JAVA_OPTS}"
done

sudo k3s kubectl set env \
  deployment/payment deployment/user \
  -n "${NAMESPACE}" \
  ZIPKIN="${ZIPKIN_ENDPOINT}"

for service in carts orders shipping payment user; do
  sudo k3s kubectl rollout status deployment/"${service}" \
    -n "${NAMESPACE}" \
    --timeout=300s
done

echo "Sock Shop tracing configured successfully."
echo "Jaeger: http://${ZIPKIN_HOST}:9411"
