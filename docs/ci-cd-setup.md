# CI/CD setup for AIOps GitHub Actions

This document completes the operational setup for the CI/CD and GitOps flow used in this repository.

## Pipeline overview

The repository now implements the following delivery flow:

1. GitHub push or pull request triggers CI.
2. Python entrypoints, dashboard JavaScript, and Kubernetes overlays are validated.
3. Unit tests run for GitOps tag update helpers.
4. SonarQube or Snyk Code performs source-code scanning when secrets are configured.
5. Docker images are built for the AIOps services.
6. Trivy scans each built image for HIGH and CRITICAL vulnerabilities.
7. Clean images are pushed to GHCR.
8. The development GitOps overlay is updated with the new image tags.
9. Argo CD synchronizes the updated manifests into Kubernetes.

## Required GitHub repository configuration

Open the GitHub repository settings for `huymt05/aiops-rca` and configure these items.

### Secrets

Create the following repository secrets:

- `SONAR_TOKEN`
  - Token generated from your SonarQube server or SonarCloud project.
- `SNYK_TOKEN`
  - API token generated from your Snyk account.

### Variables

Create the following repository variable:

- `SONAR_HOST_URL`
  - Example: `https://sonarcloud.io`
  - For self-hosted SonarQube, use your server URL.

## Recommended external services

### SonarQube or SonarCloud

Recommended minimum configuration:

- Create project key: `aiops-rca`
- Set default branch to `main`
- Enable pull-request decoration if available
- Set a quality gate that fails on new bugs, vulnerabilities, or low coverage

### Snyk Code

Recommended minimum configuration:

- Import the GitHub repository into Snyk
- Enable code scanning for pull requests
- Treat `high` severity issues as blocking

### GHCR

The workflows use the default GitHub Actions token to push to:

- `ghcr.io/huymt05/aiops-anomaly-service`
- `ghcr.io/huymt05/aiops-rca-service`
- `ghcr.io/huymt05/aiops-orchestrator`
- `ghcr.io/huymt05/aiops-dashboard`

Make sure GitHub Packages is enabled for the repository owner and the workflow has package write access.

## Verification checklist

After configuring secrets and variables, verify the following:

1. Create a test branch and modify one AIOps service file.
2. Open a pull request.
3. Confirm `aiops-ci` runs validation, unit tests, image builds, and code-scan stages.
4. Merge into `main`.
5. Confirm `aiops-build-push` runs quality gates, Trivy scan, image push, and manifest update.
6. Confirm Argo CD detects the manifest change and syncs the target environment.

## Notes

- SonarQube and Snyk are intentionally optional in the workflow so the repository can still run CI in environments where those services are not provisioned yet.
- Trivy is configured as a hard gate. HIGH or CRITICAL findings will fail the image stage before any push occurs.
- The current unit-test scope is intentionally small and focused on GitOps manifest update safety. You can extend it later with service-level tests for dashboard, anomaly, RCA, and orchestrator modules.
