# Live framework evaluation

## Repeated live benchmark

`run_live_benchmark.py` measures API success rate and end-to-end latency for one or more registered systems.

## Labeled RCA scenario

`run_labeled_rca_scenario.py` evaluates the live pipeline against a known root cause. It temporarily scales one Kubernetes Deployment to zero, generates optional HTTP traffic, calls the dashboard Live Analyze API, and compares RCA Top-1/Top-3 with the expected service.

The command records the Deployment's original replica count and restores it in a `finally`-style context manager, including when analysis raises an exception. `--confirm-disruption` is required.

Example from the K3s master for a local test cluster:

```bash
python3 evaluation/run_labeled_rca_scenario.py \
  --base-url http://127.0.0.1:8010 \
  --system-id sock_shop \
  --namespace sock-shop \
  --deployment carts \
  --expected-root-cause carts \
  --scenario-id sock-shop-carts-down \
  --traffic-url http://127.0.0.1:30001/cart \
  --kubectl-command "sudo k3s kubectl" \
  --runs 3 \
  --confirm-disruption
```

Run this only against a disposable experiment environment. The generated CSV, JSON, and Markdown files are written under `evaluation/results/labeled-rca/` by default and are ignored by Git.
