# Sock Shop Deploy

Deploy Sock Shop from local manifests:

```bash
kubectl apply -f deploy/systems/sock-shop/namespace.yaml
kubectl apply -n sock-shop -f deploy/systems/sock-shop/complete-demo.yaml
kubectl get pods -n sock-shop
kubectl get svc -n sock-shop
```

Expected frontend service:
- front-end

If exposed as NodePort, access it from the cluster master IP.
