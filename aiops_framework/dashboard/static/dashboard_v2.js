const config = window.DASHBOARD_CONFIG || {};
let systemDetails = new Map();
const ALL_PERMISSIONS = [
  "read",
  "live_analyze",
  "recovery_execute",
  "feedback_write",
  "model_select",
  "model_promote",
  "audit_view",
  "user_manage",
];

const MODEL_LABELS = {
  rf_ml_ranker: "RF ML Ranker",
  gat_baseline: "GAT baseline",
  hgnn_rca: "HGNN RCA",
};

const ROLE_DEFINITIONS = {
  admin: {
    label: "admin",
    summary: "Full platform access, including user management, recovery, feedback, and model governance.",
  },
  operator: {
    label: "operator",
    summary: "Runs live analysis, executes recovery actions, and confirms or rejects incidents.",
  },
  viewer: {
    label: "viewer",
    summary: "Read-only access to health, metadata, and analysis results without control actions.",
  },
  ml_engineer: {
    label: "ml_engineer",
    summary: "Monitors outcomes and handles model-related workflows without direct recovery execution.",
  },
};

const healthCards = document.getElementById("healthCards");
const sampleSelect = document.getElementById("sampleSelect");
const systemSelect = document.getElementById("systemSelect");
const modelSelect = document.getElementById("modelSelect");
const presetSelect = document.getElementById("presetSelect");
const forceRcaToggle = document.getElementById("forceRcaToggle");
const refreshHealthBtn = document.getElementById("refreshHealthBtn");
const runHealthyBtn = document.getElementById("runHealthyBtn");
const runIncidentBtn = document.getElementById("runIncidentBtn");
const runLiveBtn = document.getElementById("runLiveBtn");
const runDemoBtn = document.getElementById("runDemoBtn");
const metadataBox = document.getElementById("metadataBox");
const pipelineState = document.getElementById("pipelineState");
const anomalyScore = document.getElementById("anomalyScore");
const anomalyThreshold = document.getElementById("anomalyThreshold");
const anomalyDecision = document.getElementById("anomalyDecision");
const anomalyModel = document.getElementById("anomalyModel");
const anomalyBar = document.getElementById("anomalyBar");
const top1Service = document.getElementById("top1Service");
const top1Score = document.getElementById("top1Score");
const topkList = document.getElementById("topkList");
const recoveryCard = document.getElementById("recoveryCard");
const recoveryLog = document.getElementById("recoveryLog");
const recoveryHistoryList = document.getElementById("recoveryHistoryList");
const restartPodBtn = document.getElementById("restartPodBtn");
const scaleServiceBtn = document.getElementById("scaleServiceBtn");
const alertOnlyBtn = document.getElementById("alertOnlyBtn");
const liveSnapshot = document.getElementById("liveSnapshot");
const acceptIncidentBtn = document.getElementById("acceptIncidentBtn");
const rejectIncidentBtn = document.getElementById("rejectIncidentBtn");
const unknownIncidentBtn = document.getElementById("unknownIncidentBtn");
const feedbackLog = document.getElementById("feedbackLog");
const logValidationBox = document.getElementById("logValidationBox");
const topbarAnomaly = document.getElementById("topbarAnomaly");
const topbarRca = document.getElementById("topbarRca");

const authOverlay = document.getElementById("authOverlay");
const authTitle = document.getElementById("authTitle");
const authSubtitle = document.getElementById("authSubtitle");
const authMessage = document.getElementById("authMessage");
const authHint = document.getElementById("authHint");
const loginForm = document.getElementById("loginForm");
const bootstrapForm = document.getElementById("bootstrapForm");
const loginUsernameInput = document.getElementById("loginUsername");
const loginPasswordInput = document.getElementById("loginPassword");
const bootstrapUsernameInput = document.getElementById("bootstrapUsername");
const bootstrapDisplayNameInput = document.getElementById("bootstrapDisplayName");
const bootstrapPasswordInput = document.getElementById("bootstrapPassword");
const loginSubmitBtn = document.getElementById("loginSubmitBtn");
const bootstrapSubmitBtn = document.getElementById("bootstrapSubmitBtn");
const authStatusBadge = document.getElementById("authStatusBadge");
const authUserBox = document.getElementById("authUserBox");
const authDisplayName = document.getElementById("authDisplayName");
const authRoleLabel = document.getElementById("authRoleLabel");
const logoutBtn = document.getElementById("logoutBtn");
const navItems = Array.from(document.querySelectorAll(".sidebar-nav .nav-item"));
const viewPanels = Array.from(document.querySelectorAll("[data-view-panel]"));
const accessRoleBadge = document.getElementById("accessRoleBadge");
const accessSummaryBox = document.getElementById("accessSummaryBox");
const accessPermissionList = document.getElementById("accessPermissionList");
const navAccessControl = document.getElementById("navAccessControl");
const navAuditLogs = document.getElementById("navAuditLogs");
const navModelManagement = document.getElementById("navModelManagement");
const userAdminSection = document.getElementById("userAdminSection");
const userAdminState = document.getElementById("userAdminState");
const createUserForm = document.getElementById("createUserForm");
const createUserUsername = document.getElementById("createUserUsername");
const createUserDisplayName = document.getElementById("createUserDisplayName");
const createUserPassword = document.getElementById("createUserPassword");
const createUserRole = document.getElementById("createUserRole");
const createUserActive = document.getElementById("createUserActive");
const createUserSubmitBtn = document.getElementById("createUserSubmitBtn");
const userList = document.getElementById("userList");
const auditFilterActor = document.getElementById("auditFilterActor");
const auditFilterAction = document.getElementById("auditFilterAction");
const auditRefreshButton = document.getElementById("auditRefreshButton");
const auditLogState = document.getElementById("auditLogState");
const auditLogList = document.getElementById("auditLogList");
const modelManagementState = document.getElementById("modelManagementState");
const modelPanelNote = document.getElementById("modelPanelNote");
const modelSummary = document.getElementById("modelSummary");
const anomalyCandidates = document.getElementById("anomalyCandidates");
const rcaCandidates = document.getElementById("rcaCandidates");

let latestAnalysis = null;
let systems = [];
let managedUsers = [];
let currentView = "dashboard";
let auditLogItems = [];

const authState = {
  config: {
    enabled: false,
    bootstrap_required: false,
    session_ttl_hours: 0,
  },
  user: null,
  permissions: new Set(ALL_PERMISSIONS),
};

document.getElementById("anomalyUrl").textContent = config.anomalyBaseUrl || "-";
document.getElementById("rcaUrl").textContent = config.rcaBaseUrl || "-";
document.getElementById("orchUrl").textContent = config.orchestratorBaseUrl || "-";
document.getElementById("jaegerUrl").textContent = config.jaegerUrl || "-";
document.getElementById("promUrl").textContent = config.prometheusUrl || "-";

async function getSelectedSystemDetail() {
  const system = selectedSystem();
  if (!system?.system_id) {
    return null;
  }

  if (systemDetails.has(system.system_id)) {
    return systemDetails.get(system.system_id);
  }

  const detail = await fetchJson(`/api/systems/${encodeURIComponent(system.system_id)}`);
  systemDetails.set(system.system_id, detail);
  return detail;
}

function badgeClass(status) {
  if (status === "ok") return "ok";
  if (status === "down") return "bad";
  return "warn";
}

function setPipelineState(text, klass = "muted") {
  pipelineState.textContent = text;
  pipelineState.className = `state-pill ${klass}`;
}

function setTopbarBadge(node, text, klass) {
  node.textContent = text;
  node.className = `ts-badge ${klass}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setAuthMessage(message = "", kind = "error") {
  if (!message) {
    authMessage.textContent = "";
    authMessage.className = "auth-message hidden";
    return;
  }
  authMessage.textContent = message;
  authMessage.className = `auth-message ${kind === "ok" ? "ok" : ""}`.trim();
}

function setAuthStatus(text, klass = "muted") {
  authStatusBadge.textContent = text;
  authStatusBadge.className = `status-pill ${klass}`;
}

function sessionTtlHint() {
  const ttl = Number(authState.config?.session_ttl_hours || 0);
  return ttl > 0 ? `Session TTL: ${ttl} hour${ttl === 1 ? "" : "s"}.` : "";
}

function showAuthOverlay(mode, message = "", kind = "error") {
  authOverlay.classList.remove("hidden");
  loginForm.classList.add("hidden");
  bootstrapForm.classList.add("hidden");

  if (mode === "bootstrap") {
    authTitle.textContent = "Create the first admin account";
    authSubtitle.textContent = "Authentication is enabled and no users exist yet. Bootstrap the initial administrator to unlock the dashboard.";
    authHint.textContent = `Use a strong password with at least three character groups. ${sessionTtlHint()}`.trim();
    bootstrapForm.classList.remove("hidden");
    bootstrapUsernameInput.focus();
    setAuthStatus("Bootstrap required", "warn");
  } else {
    authTitle.textContent = "Sign in";
    authSubtitle.textContent = "Authenticate to access live analysis, recovery actions, and feedback workflows.";
    authHint.textContent = `Use your dashboard credentials. ${sessionTtlHint()}`.trim();
    loginForm.classList.remove("hidden");
    loginUsernameInput.focus();
    setAuthStatus("Sign-in required", "warn");
  }

  setAuthMessage(message, kind);
  applyPermissionState();
}

function hideAuthOverlay() {
  authOverlay.classList.add("hidden");
  setAuthMessage("");
}

function hasPermission(permission) {
  return !authState.config?.enabled || authState.permissions.has(permission);
}

function activeRole() {
  if (!authState.config?.enabled) {
    return "admin";
  }
  return authState.user?.role || "viewer";
}

function roleSummary(role) {
  return ROLE_DEFINITIONS[role]?.summary || "Custom role permissions are active for this user.";
}

function humanizePermission(permission) {
  return permission.replaceAll("_", " ");
}

function formatTimestamp(value) {
  if (!value) return "never";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
}

function setSectionVisible(node, visible) {
  if (!node) return;
  node.classList.toggle("hidden", !visible);
}

function switchView(viewName, scrollTargetId = "") {
  currentView = viewName || "dashboard";
  for (const panel of viewPanels) {
    const matches = panel.dataset.viewPanel === currentView;
    panel.classList.toggle("hidden", !matches);
  }
  if (scrollTargetId) {
    const target = document.getElementById(scrollTargetId);
    if (target) {
      requestAnimationFrame(() => {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }
}

function setActiveNav(clickedItem) {
  for (const item of navItems) {
    item.classList.toggle("active", item === clickedItem);
  }
}

function setControlEnabled(node, enabled, tooltip = "") {
  node.disabled = !enabled;
  if (enabled) {
    node.removeAttribute("title");
  } else if (tooltip) {
    node.title = tooltip;
  }
}

function applyPermissionState() {
  const readOnlyTooltip = "Your role cannot access this action.";
  setControlEnabled(refreshHealthBtn, hasPermission("read"), readOnlyTooltip);
  // setControlEnabled(runHealthyBtn, hasPermission("read"), readOnlyTooltip);
  // setControlEnabled(runIncidentBtn, hasPermission("read"), readOnlyTooltip);
  // setControlEnabled(runDemoBtn, hasPermission("read"), readOnlyTooltip);
  setControlEnabled(runHealthyBtn, hasPermission("demo_analyze"), "Your role cannot run demo analysis.");
  setControlEnabled(runIncidentBtn, hasPermission("demo_analyze"), "Your role cannot run demo analysis.");
  setControlEnabled(runDemoBtn, hasPermission("demo_analyze"), "Your role cannot run demo analysis.");
  setControlEnabled(runLiveBtn, hasPermission("live_analyze"), "Your role cannot run live analysis.");
  setControlEnabled(restartPodBtn, hasPermission("recovery_execute"), "Your role cannot execute recovery actions.");
  setControlEnabled(scaleServiceBtn, hasPermission("recovery_execute"), "Your role cannot execute recovery actions.");
  setControlEnabled(alertOnlyBtn, hasPermission("recovery_execute"), "Your role cannot execute recovery actions.");
  setControlEnabled(acceptIncidentBtn, hasPermission("feedback_write"), "Your role cannot submit incident feedback.");
  setControlEnabled(rejectIncidentBtn, hasPermission("feedback_write"), "Your role cannot submit incident feedback.");
  setControlEnabled(unknownIncidentBtn, hasPermission("feedback_write"), "Your role cannot submit incident feedback.");
  setSectionVisible(navAccessControl, authState.config?.enabled && hasPermission("user_manage"));
  setSectionVisible(navAuditLogs, hasPermission("audit_view"));
  setSectionVisible(navModelManagement, hasPermission("model_select"));
  setSectionVisible(userAdminSection, authState.config?.enabled && hasPermission("user_manage"));
  if (
    (currentView === "access-control" && navAccessControl.classList.contains("hidden")) ||
    (currentView === "audit-logs" && navAuditLogs.classList.contains("hidden")) ||
    (currentView === "model-management" && navModelManagement.classList.contains("hidden"))
  ) {
    switchView("dashboard");
    const dashboardNav = navItems.find((item) => item.dataset.view === "dashboard" && !item.dataset.scrollTarget);
    if (dashboardNav) {
      setActiveNav(dashboardNav);
    }
  }
}

function updateAuthChrome() {
  if (!authState.config?.enabled) {
    authUserBox.classList.add("hidden");
    logoutBtn.classList.add("hidden");
    setAuthStatus("Auth off", "muted");
    accessRoleBadge.textContent = "auth off";
    accessRoleBadge.className = "status-pill muted";
    return;
  }

  if (!authState.user) {
    authUserBox.classList.add("hidden");
    logoutBtn.classList.add("hidden");
    return;
  }

  const displayName = authState.user.display_name || authState.user.username;
  authDisplayName.textContent = displayName;
  authRoleLabel.textContent = authState.user.role || "viewer";
  authUserBox.classList.remove("hidden");
  logoutBtn.classList.remove("hidden");
  setAuthStatus("Signed in", "ok");
  accessRoleBadge.textContent = authState.user.role || "viewer";
  accessRoleBadge.className = `role-pill role-${authState.user.role || "viewer"}`;
}

function renderAccessSummary() {
  const role = activeRole();
  const permissions = authState.config?.enabled
    ? ALL_PERMISSIONS.filter((permission) => authState.permissions.has(permission))
    : ALL_PERMISSIONS;
  const identity = authState.config?.enabled
    ? (authState.user?.display_name || authState.user?.username || "Guest")
    : "Anonymous admin-mode";

  accessSummaryBox.innerHTML = `
    <div><strong>Identity:</strong> ${escapeHtml(identity)}</div>
    <div><strong>Role:</strong> ${escapeHtml(role)}</div>
    <div><strong>Mode:</strong> ${authState.config?.enabled ? "Authenticated RBAC" : "Authentication disabled"}</div>
    <div class="section-hidden-note">${escapeHtml(roleSummary(role))}</div>
  `;

  accessPermissionList.innerHTML = ALL_PERMISSIONS.map((permission) => {
    const enabled = permissions.includes(permission);
    return `<span class="permission-chip ${enabled ? "enabled" : ""}">${enabled ? "OK" : "--"} ${escapeHtml(humanizePermission(permission))}</span>`;
  }).join("");

  if (!authState.config?.enabled) {
    userAdminState.textContent = "Enable auth to create and manage named users.";
    userList.innerHTML = `<div class="history-empty">Authentication is disabled, so named user accounts are not being enforced.</div>`;
  } else if (!hasPermission("user_manage")) {
    userAdminState.textContent = "This role cannot create or modify dashboard users.";
  } else {
    userAdminState.textContent = "Admin only";
  }
}

async function readErrorResponse(resp) {
  const contentType = resp.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await resp.json().catch(() => ({}));
    if (typeof payload?.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
    return JSON.stringify(payload);
  }
  return (await resp.text()).trim();
}

async function fetchJson(url, options = {}, authOptions = {}) {
  const { authRedirect = true } = authOptions;
  const resp = await fetch(url, options);
  if (resp.status === 401) {
    const detail = (await readErrorResponse(resp)) || "Authentication required";
    if (authRedirect && authState.config?.enabled) {
      authState.user = null;
      authState.permissions = new Set();
      updateAuthChrome();
      showAuthOverlay(authState.config.bootstrap_required ? "bootstrap" : "login", "Your session expired. Please sign in again.");
      setPipelineState("Sign in required", "warn");
    }
    const err = new Error(detail);
    err.code = "AUTH_REQUIRED";
    throw err;
  }
  if (!resp.ok) {
    throw new Error((await readErrorResponse(resp)) || `Request failed with ${resp.status}`);
  }
  return resp.json();
}

async function fetchAuthConfig() {
  const resp = await fetch("/api/auth/config");
  if (!resp.ok) {
    throw new Error((await readErrorResponse(resp)) || "Failed to load authentication config");
  }
  return resp.json();
}

async function fetchCurrentUser() {
  const resp = await fetch("/api/auth/me");
  if (resp.status === 401) {
    return null;
  }
  if (!resp.ok) {
    throw new Error((await readErrorResponse(resp)) || "Failed to load current user");
  }
  return resp.json();
}

function completeAuthentication(payload) {
  authState.user = payload.user || null;
  authState.permissions = new Set(payload.permissions || []);
  authState.config.bootstrap_required = false;
  updateAuthChrome();
  renderAccessSummary();
  applyPermissionState();
  hideAuthOverlay();
}

function renderHealth(data) {
  const items = [
    { label: "Dashboard", payload: data.dashboard },
    { label: "Anomaly Service", payload: data.anomaly },
    { label: "RCA Service", payload: data.rca },
    { label: "Orchestrator", payload: data.orchestrator?.orchestrator || data.orchestrator },
  ];
  healthCards.innerHTML = items.map((item) => {
    const status = item.payload?.status || "unknown";
    const meta = Object.entries(item.payload || {})
      .filter(([key]) => key !== "status")
      .slice(0, 2)
      .map(([key, value]) => `<div><small>${key}: ${String(value)}</small></div>`)
      .join("");
    return `
      <article class="health-card">
        <span class="label">${item.label}</span>
        <span class="status-pill ${badgeClass(status)}">${status}</span>
        <div style="margin-top:10px">${meta}</div>
      </article>
    `;
  }).join("");
}

function renderMetadata(data, systemDetail = null) {
  metadataBox.textContent = JSON.stringify({
    selected_system: systemDetail ? {
      system_id: systemDetail.system_id,
      display_name: systemDetail.display_name,
      namespace: systemDetail.namespace,
      entry_services: systemDetail.entry_services || [],
      jaeger_services: systemDetail.jaeger_services || [],
      prometheus_labels: systemDetail.prometheus_labels || {},
      trace_adapter: systemDetail.trace_adapter || null,
      metric_adapter: systemDetail.metric_adapter || null,
      log_adapter: systemDetail.log_adapter || null,
      service_catalog: systemDetail.service_catalog || null,
    } : null,
    global_defaults: {
      presets: data.window_presets,
      sample_count: data.samples?.length || 0,
      systems: data.systems?.map((item) => item.system_id) || [],
      live_defaults: data.live_defaults,
      recovery: data.recovery,
      auth: data.auth,
    },
    model_metadata: {
      anomaly_model: data.anomaly?.inference_config?.model_name,
      rca_model: data.rca?.inference_config?.model_name,
      rca_default_model_key: data.rca?.default_model_key || null,
      rca_available_models: data.rca?.available_models || [],
    },
    orchestrator: data.orchestrator,
  }, null, 2);
}


function renderSystems(items, defaultSystemId) {
  systems = Array.isArray(items) ? items : [];
  systemSelect.innerHTML = "";
  for (const item of systems) {
    const option = document.createElement("option");
    option.value = item.system_id;
    option.textContent = item.display_name || item.system_id;
    systemSelect.appendChild(option);
  }
  if (systems.length > 0) {
    systemSelect.value = defaultSystemId || systems[0].system_id;
  }
}

function modelLabelFor(key) {
  return MODEL_LABELS[key] || key || "Unknown model";
}

function renderRcaModels(rcaMetadata) {
  const available = Array.isArray(rcaMetadata?.available_models) && rcaMetadata.available_models.length
    ? rcaMetadata.available_models
    : [{ model_key: config.defaultModelKey || "rf_ml_ranker", label: modelLabelFor(config.defaultModelKey || "rf_ml_ranker") }];

  modelSelect.innerHTML = "";
  for (const item of available) {
    const option = document.createElement("option");
    option.value = item.model_key;
    option.textContent = item.label || modelLabelFor(item.model_key);
    modelSelect.appendChild(option);
  }

  const defaultKey = rcaMetadata?.default_model_key || config.defaultModelKey || available[0]?.model_key || "rf_ml_ranker";
  modelSelect.value = available.some((item) => item.model_key === defaultKey) ? defaultKey : available[0]?.model_key || defaultKey;
}

function selectedSystem() {
  return systems.find((item) => item.system_id === systemSelect.value) || systems[0] || null;
}

function renderSamples(items) {
  sampleSelect.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name;
    sampleSelect.appendChild(option);
  }
  if (items.length > 0) {
    sampleSelect.value = items[0].name;
  }
}

function renderLiveSnapshot(result) {
  const live = result.live_context || null;
  if (!live) return;

  const trace = live.trace_snapshot || {};
  const metrics = live.metrics_snapshot || {};
  const features = live.window_features || {};
  const metricText = metrics.values && Object.keys(metrics.values).length
    ? Object.entries(metrics.values).map(([key, value]) => `${key}=${Number(value.value).toFixed(3)}`).join(" | ")
    : "No compatible Prometheus metric found";

  liveSnapshot.innerHTML = `
    <div><strong>System:</strong> ${live.system_id || "-"}</div>
    <div><strong>Event:</strong> ${result.event_id || "-"}</div>
    <div><strong>Source:</strong> ${live.source_service}</div>
    <div><strong>Trace count:</strong> ${trace.trace_count ?? "-"}</div>
    <div><strong>Span count:</strong> ${trace.span_count ?? "-"}</div>
    <div><strong>Service count:</strong> ${trace.service_count ?? "-"}</div>
    <div><strong>Error trace ratio:</strong> ${Number(features.error_trace_ratio ?? 0).toFixed(2)}</div>
    <div><strong>Error span ratio:</strong> ${Number(features.error_span_ratio ?? 0).toFixed(2)}</div>
    <div><strong>RCA model:</strong> ${escapeHtml(live.model_key || latestAnalysis?.rca?.model_key || modelSelect.value || "-")}</div>
    <div><strong>P95 latency:</strong> ${Number(features.p95_trace_duration_ms ?? 0).toFixed(1)} ms</div>
    <div><strong>Prometheus:</strong> ${metrics.status || "-"}</div>
    <div><strong>Metrics:</strong> ${metricText}</div>
  `;
}

function renderRecoveryHistory(items) {
  if (!Array.isArray(items) || items.length === 0) {
    recoveryHistoryList.innerHTML = `<div class="history-empty">No recovery action recorded yet.</div>`;
    return;
  }

  recoveryHistoryList.innerHTML = items.map((item) => `
    <article class="history-item">
      <div class="history-item-top">
        <span class="history-action">${item.action_label}</span>
        <span class="history-status ${item.status === "executed" ? "ok" : "warn"}">${item.status}</span>
      </div>
      <div class="history-meta">
        <span>service=${item.service_name}</span>
        <span>severity=${item.severity}</span>
        <span>mode=${item.mode || "-"}</span>
      </div>
      <div class="history-note">${item.notes || "-"}</div>
      <div class="history-time">${item.timestamp || "-"}</div>
    </article>
  `).join("");
}

function renderUsers(items) {
  managedUsers = Array.isArray(items) ? items : [];

  if (!managedUsers.length) {
    userList.innerHTML = `<div class="history-empty">No dashboard users found yet.</div>`;
    return;
  }

  userList.innerHTML = managedUsers.map((user) => {
    const isSelf = authState.user?.username === user.username;
    return `
      <article class="user-card" data-username="${escapeHtml(user.username)}">
        <div class="user-card-top">
          <div class="user-name-stack">
            <span class="user-display">${escapeHtml(user.display_name || user.username)}</span>
            <span class="user-username">${escapeHtml(user.username)}</span>
          </div>
          <div class="user-badges">
            <span class="role-pill">${escapeHtml(user.role)}</span>
            <span class="status-pill ${user.is_active ? "ok" : "bad"}">${user.is_active ? "active" : "inactive"}</span>
          </div>
        </div>
        <div class="user-meta">
          <span>created=${escapeHtml(formatTimestamp(user.created_at))}</span>
          <span>updated=${escapeHtml(formatTimestamp(user.updated_at))}</span>
          <span>last_login=${escapeHtml(formatTimestamp(user.last_login_at))}</span>
        </div>
        <div class="user-controls">
          <select class="user-role-select" ${isSelf ? "disabled" : ""}>
            ${Object.keys(ROLE_DEFINITIONS).map((role) => `<option value="${role}" ${role === user.role ? "selected" : ""}>${role}</option>`).join("")}
          </select>
          <button class="btn btn-action" data-action="update-role" ${isSelf ? "disabled title=\"Update another user from a separate admin session if needed.\"" : ""} type="button">Update role</button>
          <button class="btn ${user.is_active ? "btn-warn" : "btn-action"}" data-action="toggle-active" ${isSelf ? "disabled title=\"Avoid disabling the account currently in use.\"" : ""} type="button">${user.is_active ? "Deactivate" : "Activate"}</button>
          <button class="btn btn-ghost-sm" data-action="reset-password" type="button">Set password</button>
        </div>
      </article>
    `;
  }).join("");
}

function summarizeAuditPayload(payload) {
  if (!payload || typeof payload !== "object" || !Object.keys(payload).length) {
    return "";
  }
  return JSON.stringify(payload, null, 2);
}

function renderAuditLogs(items) {
  auditLogItems = Array.isArray(items) ? items : [];

  if (!auditLogItems.length) {
    auditLogList.innerHTML = `<div class="history-empty">No audit events match the current filters.</div>`;
    return;
  }

  auditLogList.innerHTML = auditLogItems.map((item) => {
    const payloadText = summarizeAuditPayload(item.payload);
    return `
      <article class="audit-log-card">
        <div><strong>${escapeHtml(item.action || "unknown_action")}</strong></div>
        <div class="audit-log-meta">
          <span>${escapeHtml(formatTimestamp(item.created_at))}</span>
          <span>actor=${escapeHtml(item.actor || "system")}</span>
          <span>target=${escapeHtml(item.target_type || "-")}:${escapeHtml(item.target_id || "-")}</span>
        </div>
        ${payloadText ? `<div class="audit-log-payload">${escapeHtml(payloadText)}</div>` : ""}
      </article>
    `;
  }).join("");
}

function formatMetricPairs(metrics) {
  if (!metrics || typeof metrics !== "object" || !Object.keys(metrics).length) {
    return "No metrics recorded.";
  }
  return Object.entries(metrics)
    .map(([key, value]) => `${key}=${typeof value === "number" ? value.toFixed(4) : String(value)}`)
    .join(" | ");
}

function formatMlflowMeta(item) {
  if (!item || typeof item !== "object") {
    return "";
  }

  const parts = [];
  if (item.mlflow_experiment_name) {
    parts.push(`<span>experiment=${escapeHtml(item.mlflow_experiment_name)}</span>`);
  }
  if (item.mlflow_run_id) {
    parts.push(`<span>run=${escapeHtml(item.mlflow_run_id)}</span>`);
  }
  if (item.mlflow_artifact_uri) {
    parts.push(`<span class="model-uri">artifact_uri=${escapeHtml(item.mlflow_artifact_uri)}</span>`);
  }
  return parts.join("");
}

function renderModelCard(systemId, modelType, item, canPromote) {
  if (!item) {
    return `<div class="history-empty">No ${escapeHtml(modelType)} model available.</div>`;
  }

  return `
    <article class="model-card">
      <div><strong>${escapeHtml(item.model_name || item.model_id || "unknown_model")}</strong></div>
      <div class="model-meta">
        <span>score=${Number(item.rank_score ?? 0).toFixed(4)}</span>
        <span>trained=${escapeHtml(formatTimestamp(item.trained_at))}</span>
        <span>status=${escapeHtml(item.status || modelType)}</span>
      </div>
      <div class="model-metrics">${escapeHtml(formatMetricPairs(item.metrics || {}))}</div>
      <div class="model-meta">
        <span>artifact=${escapeHtml(item.artifact_dir || "-")}</span>
      </div>
      ${formatMlflowMeta(item) ? `<div class="model-meta model-meta-mlflow">${formatMlflowMeta(item)}</div>` : ""}
      <div class="model-card-actions">
        <button
          type="button"
          class="btn btn-action promote-model-button"
          data-system-id="${escapeHtml(systemId)}"
          data-model-type="${escapeHtml(modelType)}"
          data-model-name="${escapeHtml(item.model_name || "")}"
          ${canPromote ? "" : "disabled title=\"Your role cannot promote models.\""}
        >
          Promote
        </button>
      </div>
    </article>
  `;
}

function renderModelSummary(payload) {
  const selected = selectedSystem();
  modelPanelNote.textContent = selected
    ? `Registry-backed production and candidate models for ${selected.display_name || selected.system_id}`
    : "Registry-backed production and candidate models";

  modelSummary.innerHTML = [
    {
      title: "Anomaly Production",
      item: payload?.anomaly?.production,
    },
    {
      title: "RCA Production",
      item: payload?.rca?.production,
    },
  ].map(({ title, item }) => `
    <article class="model-card">
      <div><strong>${escapeHtml(title)}</strong></div>
      <div class="model-meta">
        <span>model=${escapeHtml(item?.model_name || "-")}</span>
        <span>updated=${escapeHtml(formatTimestamp(item?.updated_at))}</span>
      </div>
      <div class="model-metrics">${escapeHtml(formatMetricPairs(item?.metrics || {}))}</div>
      <div class="model-meta">
        <span>artifact=${escapeHtml(item?.artifact_dir || "-")}</span>
      </div>
      ${formatMlflowMeta(item) ? `<div class="model-meta model-meta-mlflow">${formatMlflowMeta(item)}</div>` : ""}
    </article>
  `).join("");

  const canPromote = hasPermission("model_promote");
  anomalyCandidates.innerHTML = (payload?.anomaly?.candidates || []).length
    ? payload.anomaly.candidates.map((item) => renderModelCard(payload.system_id, "anomaly", item, canPromote)).join("")
    : `<div class="history-empty">No anomaly candidates recorded for this system.</div>`;
  rcaCandidates.innerHTML = (payload?.rca?.candidates || []).length
    ? payload.rca.candidates.map((item) => renderModelCard(payload.system_id, "rca", item, canPromote)).join("")
    : `<div class="history-empty">No RCA candidates recorded for this system.</div>`;
}

async function refreshUsers() {
  if (!authState.config?.enabled || !hasPermission("user_manage")) {
    return;
  }
  userAdminState.textContent = "Loading users...";
  try {
    const payload = await fetchJson("/api/users");
    renderUsers(payload.items || []);
    userAdminState.textContent = `${(payload.items || []).length} user account(s) loaded.`;
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") return;
    userAdminState.textContent = `Failed to load users: ${err.message}`;
    userList.innerHTML = `<div class="history-empty">Failed to load users: ${escapeHtml(err.message)}</div>`;
  }
}

async function refreshAuditLogs() {
  if (!hasPermission("audit_view")) {
    return;
  }

  auditLogState.textContent = "Loading audit log...";
  try {
    const params = new URLSearchParams({ limit: "100" });
    if (auditFilterActor?.value.trim()) {
      params.set("actor", auditFilterActor.value.trim());
    }
    if (auditFilterAction?.value.trim()) {
      params.set("action", auditFilterAction.value.trim());
    }
    const system = selectedSystem();
    if (system?.system_id) {
      params.set("system_id", system.system_id);
    }
    const payload = await fetchJson(`/api/audit-logs?${params.toString()}`);
    renderAuditLogs(payload.items || []);
    auditLogState.textContent = `${(payload.items || []).length} audit event(s) loaded.`;
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") return;
    auditLogState.textContent = `Failed to load audit logs: ${err.message}`;
    auditLogList.innerHTML = `<div class="history-empty">Failed to load audit logs: ${escapeHtml(err.message)}</div>`;
  }
}

async function refreshModelManagement() {
  if (!hasPermission("model_select")) {
    return;
  }

  const system = selectedSystem();
  if (!system?.system_id) {
    modelManagementState.textContent = "No system selected.";
    modelSummary.innerHTML = "";
    anomalyCandidates.innerHTML = `<div class="history-empty">No anomaly candidates loaded.</div>`;
    rcaCandidates.innerHTML = `<div class="history-empty">No RCA candidates loaded.</div>`;
    return;
  }

  modelManagementState.textContent = `Loading models for ${system.display_name || system.system_id}...`;
  try {
    const payload = await fetchJson(`/api/models?system_id=${encodeURIComponent(system.system_id)}`);
    renderModelSummary(payload);
    modelManagementState.textContent = `Loaded registry for ${system.display_name || system.system_id}.`;
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") return;
    modelManagementState.textContent = `Failed to load model registry: ${err.message}`;
    modelSummary.innerHTML = "";
    anomalyCandidates.innerHTML = `<div class="history-empty">Failed to load anomaly candidates: ${escapeHtml(err.message)}</div>`;
    rcaCandidates.innerHTML = `<div class="history-empty">Failed to load RCA candidates: ${escapeHtml(err.message)}</div>`;
  }
}

async function refreshMetadataPanel() {
  try {
    const [metadata, systemDetail] = await Promise.all([
      fetchJson("/api/metadata"),
      getSelectedSystemDetail(),
    ]);
    renderMetadata(metadata, systemDetail);
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") return;
    metadataBox.textContent = `Failed to load metadata: ${err.message}`;
  }
}


async function refreshRecoveryHistory() {
  try {
    const history = await fetchJson("/api/recovery/history");
    renderRecoveryHistory(history.items || []);
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") return;
    recoveryHistoryList.innerHTML = `<div class="history-empty">Failed to load recovery history: ${err.message}</div>`;
  }
}

function renderAnalysis(result) {
  latestAnalysis = result;
  const anomaly = result.anomaly || {};
  const rca = result.rca || null;
  const rec = result.recommendation || {};
  const isAnomaly = Boolean(anomaly.is_anomaly);

  const rcaModelLabel = rca ? (modelLabelFor(rca.model_key || rca.metadata?.model_key || rca.model_name)) : modelLabelFor(modelSelect.value);

  anomalyScore.textContent = Number(anomaly.anomaly_score || 0).toFixed(4);
  anomalyThreshold.textContent = `threshold: ${Number(anomaly.threshold || 0).toFixed(2)}`;
  anomalyDecision.textContent = isAnomaly ? "Anomaly" : "Normal";
  anomalyModel.textContent = `model: ${anomaly.model_name || "-"}`;
  anomalyBar.style.width = `${Math.max(2, Math.min(100, (anomaly.anomaly_score || 0) * 100))}%`;
  setPipelineState(result.pipeline_state || "unknown", isAnomaly ? "warn" : "ok");
  anomalyDecision.style.color = isAnomaly ? "#f85149" : "#3fb950";
  setTopbarBadge(topbarAnomaly, isAnomaly ? "Alert" : "OK", isAnomaly ? "bad" : "ok");

  if (rca && rca.top1) {
    top1Service.textContent = rca.top1.service_name;
    top1Score.textContent = `score: ${Number(rca.top1.score || 0).toFixed(3)} | model: ${rcaModelLabel}`;
    topkList.innerHTML = (rca.topk || []).map((item, idx) => `
      <li>
        <span><strong>#${idx + 1}</strong> <span class="rank-service">${item.service_name}</span></span>
        <span>${Number(item.score || 0).toFixed(3)}</span>
      </li>
    `).join("");
    setTopbarBadge(topbarRca, rca.top1.service_name, "warn");
  } else {
    top1Service.textContent = "Skipped";
    top1Score.textContent = `score: - | model: ${rcaModelLabel}`;
    topkList.innerHTML = `<li><span>RCA was not triggered for this run.</span><span>-</span></li>`;
    setTopbarBadge(topbarRca, "Ready", "ok");
  }

  recoveryCard.innerHTML = `
    <p><strong>Status:</strong> ${rec.status || "-"}</p>
    <p><strong>Severity:</strong> ${rec.severity || "-"}</p>
    <p><strong>Primary action:</strong> ${rec.primary_action || "-"}</p>
    <p><strong>Secondary action:</strong> ${rec.secondary_action || "-"}</p>
    <p>${rec.notes || "No recommendation yet."}</p>
  `;

  renderLiveSnapshot(result);
  // if (isAnomaly && rca?.top1?.service_name) {
  //   refreshLogValidation(rca.top1.service_name).catch((err) => {
  //     if (err.code !== "AUTH_REQUIRED") {
  //       logValidationBox.textContent = `Log validation failed: ${err.message}`;
  //     }
  //   });
  // } else {
  //   logValidationBox.textContent = "No anomaly RCA target yet.";
  // }
  if (rca?.top1?.service_name) {
    refreshLogValidation(rca.top1.service_name);
  } else {
    logValidationBox.innerHTML = `<div class="history-empty">No anomaly RCA target yet.</div>`;
  }
}

function currentRecoveryContext() {
  const recommendation = latestAnalysis?.recommendation || {};
  const rca = latestAnalysis?.rca || {};
  return {
    service_name: recommendation.predicted_service || rca?.top1?.service_name || "unknown",
    severity: recommendation.severity || "unknown",
    context: {
      pipeline_state: latestAnalysis?.pipeline_state || "unknown",
      sample_name: latestAnalysis?.demo_context?.sample_name || latestAnalysis?.live_context?.source_service || sampleSelect.value,
      preset: latestAnalysis?.demo_context?.preset || presetSelect.value,
      anomaly_score: latestAnalysis?.anomaly?.anomaly_score ?? null,
      event_id: latestAnalysis?.event_id ?? null,
      system_id: latestAnalysis?.live_context?.system_id ?? systemSelect.value,
    },
  };
}

// async function executeRecovery(action) {
//   if (!latestAnalysis) {
//     alert("Run an analysis first so the dashboard has a target service.");
//     return;
//   }

//   const context = currentRecoveryContext();
//   try {
//     const result = await fetchJson("/api/recovery/execute", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         action,
//         service_name: context.service_name,
//         severity: context.severity,
//         source: "dashboard_demo",
//         context: context.context,
//       }),
//     });
//     recoveryLog.innerHTML = `
//       <strong>${result.action_label}</strong> accepted for <strong>${result.service_name}</strong><br/>
//       <span>Status: ${result.status} | Severity: ${result.severity} | Mode: ${result.mode || "-"}</span><br/>
//       <span>${result.notes}</span><br/>
//       <small>${result.timestamp}</small>
//     `;
//     await refreshRecoveryHistory();
//   } catch (err) {
//     if (err.code !== "AUTH_REQUIRED") {
//       alert(err.message);
//     }
//   }
// }

async function executeRecovery(action) {
  if (!latestAnalysis) {
    alert("Run an analysis first so the dashboard has a target service.");
    return;
  }

  const context = currentRecoveryContext();
  try {
    const result = await fetchJson("/api/recovery/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        system_id: context.context.system_id,
        service_name: context.service_name,
        severity: context.severity,
        source: "dashboard_demo",
        context: context.context,
      }),
    });
    recoveryLog.innerHTML = `
      <strong>${result.action_label}</strong> accepted for <strong>${result.service_name}</strong><br/>
      <span>Status: ${result.status} | Severity: ${result.severity} | Mode: ${result.mode || "-"}</span><br/>
      <span>${result.notes}</span><br/>
      <small>${result.timestamp}</small>
    `;
    await refreshRecoveryHistory();
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      alert(err.message);
    }
  }
}

async function recordFeedback(feedback) {
  if (!latestAnalysis?.event_id) {
    alert("Run Live Analyze first so the dashboard has a monitoring event.");
    return;
  }
  try {
    const result = await fetchJson(`/api/monitoring-events/${latestAnalysis.event_id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback,
        actor: authState.user?.username || "operator",
        context: {
          system_id: latestAnalysis?.live_context?.system_id || systemSelect.value,
          anomaly_score: latestAnalysis?.anomaly?.anomaly_score ?? null,
          rca_top1_service: latestAnalysis?.rca?.top1?.service_name || null,
        },
      }),
    });
    feedbackLog.innerHTML = `
      <strong>${escapeHtml(result.feedback)}</strong> saved for event #${escapeHtml(result.event_id)}<br/>
      <small>${escapeHtml(result.created_at)}</small>
    `;
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      alert(err.message);
    }
  }
}

// async function refreshLogValidation(serviceName) {
//   const systemId = latestAnalysis?.live_context?.system_id || systemSelect.value;
//   logValidationBox.textContent = `Loading logs for ${serviceName}...`;
//   const params = new URLSearchParams({
//     system_id: systemId,
//     service_name: serviceName,
//     tail: "200",
//     since: "10m",
//   });
//   const data = await fetchJson(`/api/logs/recent?${params.toString()}`);
//   const lines = data.important_lines?.length ? data.important_lines : data.raw_tail || [];
//   const renderedLines = lines.length
//     ? lines.map((line) => `<div class="log-line">${escapeHtml(line)}</div>`).join("")
//     : `<div class="history-empty">No error/warn/exception lines found in the recent log window.</div>`;
//   logValidationBox.innerHTML = `
//     <div class="history-meta">
//       <span>service=${escapeHtml(data.service_name)}</span>
//       <span>namespace=${escapeHtml(data.namespace)}</span>
//       <span>important=${escapeHtml(data.important_count)}</span>
//       <span>lines=${escapeHtml(data.line_count)}</span>
//     </div>
//     <div class="log-lines">${renderedLines}</div>
//   `;
// }
async function refreshLogValidation(serviceName) {
  const systemId = latestAnalysis?.live_context?.system_id || systemSelect.value;
  logValidationBox.innerHTML = `
    <div class="history-meta">
      <span>service=${escapeHtml(serviceName)}</span>
      <span>system=${escapeHtml(systemId || "-")}</span>
      <span>status=loading</span>
    </div>
    <div class="history-empty">Loading recent logs...</div>
  `;

  try {
    const params = new URLSearchParams({
      system_id: systemId,
      service_name: serviceName,
      tail: "200",
      since: "10m",
    });
    const data = await fetchJson(`/api/logs/recent?${params.toString()}`);
    const lines = data.important_lines?.length ? data.important_lines : data.raw_tail || [];
    const renderedLines = lines.length
      ? lines.map((line) => `<div class="log-line">${escapeHtml(line)}</div>`).join("")
      : `<div class="history-empty">No error/warn/exception lines found in the recent log window.</div>`;

    logValidationBox.innerHTML = `
      <div class="history-meta">
        <span>service=${escapeHtml(data.service_name)}</span>
        <span>namespace=${escapeHtml(data.namespace)}</span>
        <span>important=${escapeHtml(data.important_count)}</span>
        <span>lines=${escapeHtml(data.line_count)}</span>
      </div>
      <div class="log-lines">${renderedLines}</div>
    `;
  } catch (err) {
    if (err.code === "AUTH_REQUIRED") {
      return;
    }
    logValidationBox.innerHTML = `
      <div class="history-meta">
        <span>service=${escapeHtml(serviceName)}</span>
        <span>system=${escapeHtml(systemId || "-")}</span>
        <span>status=failed</span>
      </div>
      <div class="history-empty">Failed to load logs: ${escapeHtml(err.message || "Unknown error")}</div>
    `;
  }
}



// async function runAnalysis({ preset, runRcaOnAnyInput }) {
//   if (!sampleSelect.value) {
//     alert("Please choose a graph sample first.");
//     setPipelineState("Sample required", "bad");
//     return;
//   }
//   setPipelineState("Analyzing", "warn");
//   try {
//     const result = await fetchJson("/api/demo/analyze", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         sample_name: sampleSelect.value,
//         preset,
//         run_rca_on_any_input: runRcaOnAnyInput,
//         model_key: modelSelect.value,
//       }),
//     });
//     renderAnalysis(result);
//   } catch (err) {
//     if (err.code !== "AUTH_REQUIRED") {
//       setPipelineState("Analysis failed", "bad");
//       alert(err.message);
//     }
//   }
// }

async function runAnalysis({ preset, runRcaOnAnyInput }) {
  if (!sampleSelect.value) {
    alert("Please choose a graph sample first.");
    setPipelineState("Sample required", "bad");
    return;
  }
  setPipelineState("Analyzing", "warn");
  try {
    const result = await fetchJson("/api/demo/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_id: selectedSystem()?.system_id || systemSelect.value,
        sample_name: sampleSelect.value,
        preset,
        run_rca_on_any_input: runRcaOnAnyInput,
        model_key: modelSelect.value,
      }),
    });
    renderAnalysis(result);
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      setPipelineState("Analysis failed", "bad");
      alert(err.message);
    }
  }
}


async function runLiveAnalysis() {
  setPipelineState("Collecting live data", "warn");
  try {
    const system = selectedSystem();
    const result = await fetchJson("/api/live/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_id: system?.system_id || systemSelect.value,
        source_service: "all",
        run_rca_on_any_input: false,
        model_key: modelSelect.value,
      }),
    });
    renderAnalysis(result);
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      setPipelineState("Live analyze failed", "bad");
      alert(err.message);
    }
  }
}

async function boot() {
  setPipelineState("Loading", "muted");
  const [health, metadata, samples, history, systemPayload] = await Promise.all([
    fetchJson("/api/health"),
    fetchJson("/api/metadata"),
    fetchJson("/api/samples"),
    fetchJson("/api/recovery/history"),
    fetchJson("/api/systems"),
  ]);
  renderHealth(health);
  renderMetadata(metadata);
  renderRcaModels(metadata.rca || {});
  renderSamples(samples.items || []);
  renderRecoveryHistory(history.items || []);
  renderSystems(systemPayload.items || [], systemPayload.default_system_id);
  renderAccessSummary();
  await refreshUsers();
  await refreshAuditLogs();
  await refreshModelManagement();
  setPipelineState("Ready", "ok");
}

async function ensureAuthenticated() {
  authState.config = await fetchAuthConfig();
  if (!authState.config.enabled) {
    authState.user = null;
    authState.permissions = new Set(ALL_PERMISSIONS);
    updateAuthChrome();
    renderAccessSummary();
    applyPermissionState();
    hideAuthOverlay();
    return true;
  }

  const me = authState.config.bootstrap_required ? null : await fetchCurrentUser();
  if (me) {
    completeAuthentication(me);
    return true;
  }

  authState.user = null;
  authState.permissions = new Set();
  updateAuthChrome();
  renderAccessSummary();
  applyPermissionState();
  showAuthOverlay(authState.config.bootstrap_required ? "bootstrap" : "login");
  return false;
}

async function handleLoginSubmit(event) {
  event.preventDefault();
  const username = loginUsernameInput.value.trim();
  const password = loginPasswordInput.value;
  if (!username || !password) {
    setAuthMessage("Username and password are required.");
    return;
  }

  loginSubmitBtn.disabled = true;
  setAuthMessage("");
  try {
    const result = await fetchJson("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }, { authRedirect: false });
    completeAuthentication(result);
    loginPasswordInput.value = "";
    await boot();
  } catch (err) {
    setAuthMessage(err.message || "Sign-in failed.");
  } finally {
    loginSubmitBtn.disabled = false;
  }
}

async function handleBootstrapSubmit(event) {
  event.preventDefault();
  const username = bootstrapUsernameInput.value.trim();
  const displayName = bootstrapDisplayNameInput.value.trim() || "Administrator";
  const password = bootstrapPasswordInput.value;
  if (!username || !password) {
    setAuthMessage("Admin username and password are required.");
    return;
  }

  bootstrapSubmitBtn.disabled = true;
  setAuthMessage("");
  try {
    await fetchJson("/api/auth/bootstrap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        display_name: displayName,
        password,
      }),
    }, { authRedirect: false });
    authState.config.bootstrap_required = false;
    setAuthMessage("Admin account created. Signing you in...", "ok");
    const result = await fetchJson("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }, { authRedirect: false });
    completeAuthentication(result);
    bootstrapPasswordInput.value = "";
    await boot();
  } catch (err) {
    setAuthMessage(err.message || "Bootstrap failed.");
  } finally {
    bootstrapSubmitBtn.disabled = false;
  }
}

async function handleLogout() {
  logoutBtn.disabled = true;
  try {
    await fetchJson("/api/auth/logout", { method: "POST" }, { authRedirect: false });
  } catch (_err) {
    // Best effort logout: clear UI state even if the server session is already gone.
  } finally {
    authState.user = null;
    authState.permissions = new Set();
    updateAuthChrome();
    renderAccessSummary();
    applyPermissionState();
    showAuthOverlay("login", "You have been signed out.", "ok");
    setPipelineState("Signed out", "muted");
    logoutBtn.disabled = false;
  }
}

async function handleCreateUser(event) {
  event.preventDefault();
  const username = createUserUsername.value.trim();
  const password = createUserPassword.value;
  const displayName = createUserDisplayName.value.trim();
  const role = createUserRole.value;
  const isActive = createUserActive.checked;

  if (!username || !password) {
    userAdminState.textContent = "Username and password are required to create a user.";
    return;
  }

  createUserSubmitBtn.disabled = true;
  userAdminState.textContent = `Creating ${username}...`;
  try {
    await fetchJson("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        password,
        display_name: displayName,
        role,
        is_active: isActive,
      }),
    });
    createUserForm.reset();
    createUserRole.value = "operator";
    createUserActive.checked = true;
    userAdminState.textContent = `Created user ${username}.`;
    await refreshUsers();
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      userAdminState.textContent = `Create user failed: ${err.message}`;
    }
  } finally {
    createUserSubmitBtn.disabled = false;
  }
}

async function handleUserListClick(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest("[data-username]");
  if (!card) return;
  const username = card.dataset.username;
  const action = button.dataset.action;

  try {
    if (action === "update-role") {
      const role = card.querySelector(".user-role-select")?.value;
      if (!role) return;
      button.disabled = true;
      userAdminState.textContent = `Updating role for ${username}...`;
      await fetchJson(`/api/users/${encodeURIComponent(username)}/role`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      userAdminState.textContent = `Updated role for ${username}.`;
      await refreshUsers();
      return;
    }

    if (action === "toggle-active") {
      const current = managedUsers.find((item) => item.username === username);
      if (!current) return;
      button.disabled = true;
      userAdminState.textContent = `${current.is_active ? "Deactivating" : "Activating"} ${username}...`;
      await fetchJson(`/api/users/${encodeURIComponent(username)}/active`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_active: !current.is_active }),
      });
      userAdminState.textContent = `${current.is_active ? "Deactivated" : "Activated"} ${username}.`;
      await refreshUsers();
      return;
    }

    if (action === "reset-password") {
      const password = window.prompt(`Set a new password for ${username}:`);
      if (!password) return;
      button.disabled = true;
      userAdminState.textContent = `Updating password for ${username}...`;
      await fetchJson(`/api/users/${encodeURIComponent(username)}/password`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      userAdminState.textContent = `Updated password for ${username}.`;
    }
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      userAdminState.textContent = `User action failed: ${err.message}`;
    }
  } finally {
    button.disabled = false;
  }
}

async function handleModelCardClick(event) {
  const button = event.target.closest(".promote-model-button");
  if (!button) return;

  const systemId = button.dataset.systemId;
  const modelType = button.dataset.modelType;
  const modelName = button.dataset.modelName;
  if (!systemId || !modelType || !modelName) {
    return;
  }

  try {
    button.disabled = true;
    modelManagementState.textContent = `Promoting ${modelName} for ${systemId}/${modelType}...`;
    await fetchJson("/api/models/promote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_id: systemId,
        model_type: modelType,
        model_name: modelName,
      }),
    });
    modelManagementState.textContent = `Promoted ${modelName} for ${systemId}/${modelType}.`;
    await refreshModelManagement();
    await refreshAuditLogs();
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      modelManagementState.textContent = `Promote failed: ${err.message}`;
    }
  } finally {
    button.disabled = false;
  }
}

async function initializeApp() {
  try {
    const ready = await ensureAuthenticated();
    if (!ready) {
      setPipelineState(authState.config.bootstrap_required ? "Bootstrap required" : "Sign in required", "warn");
      return;
    }
    await boot();
  } catch (err) {
    setPipelineState("Startup failed", "bad");
    metadataBox.textContent = err.message;
    if (authState.config?.enabled) {
      showAuthOverlay(authState.config.bootstrap_required ? "bootstrap" : "login", err.message);
    }
  }
}

refreshHealthBtn.addEventListener("click", async () => {
  setPipelineState("Refreshing health", "muted");
  try {
    const [health, metadata] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/metadata"),
    ]);
    renderHealth(health);
    const systemDetail = await getSelectedSystemDetail();
    renderMetadata(metadata, systemDetail);
    renderRcaModels(metadata.rca || {});
    setPipelineState("Ready", "ok");
  } catch (err) {
    if (err.code !== "AUTH_REQUIRED") {
      setPipelineState("Health error", "bad");
      alert(err.message);
    }
  }
});

runDemoBtn.addEventListener("click", async () => {
  runAnalysis({
    preset: presetSelect.value,
    runRcaOnAnyInput: forceRcaToggle.checked,
  });
});

runHealthyBtn.addEventListener("click", async () => {
  presetSelect.value = "healthy";
  forceRcaToggle.checked = false;
  await runAnalysis({ preset: "healthy", runRcaOnAnyInput: false });
});

runIncidentBtn.addEventListener("click", async () => {
  presetSelect.value = "suspicious";
  forceRcaToggle.checked = false;
  await runAnalysis({ preset: "suspicious", runRcaOnAnyInput: false });
});

runLiveBtn.addEventListener("click", async () => {
  await runLiveAnalysis();
});

restartPodBtn.addEventListener("click", async () => {
  await executeRecovery("restart_pod");
});

scaleServiceBtn.addEventListener("click", async () => {
  await executeRecovery("scale_service");
});

alertOnlyBtn.addEventListener("click", async () => {
  await executeRecovery("alert_only");
});

acceptIncidentBtn.addEventListener("click", async () => {
  await recordFeedback("accepted_incident");
});

rejectIncidentBtn.addEventListener("click", async () => {
  await recordFeedback("rejected_false_positive");
});

unknownIncidentBtn.addEventListener("click", async () => {
  await recordFeedback("unknown");
});

loginForm.addEventListener("submit", handleLoginSubmit);
bootstrapForm.addEventListener("submit", handleBootstrapSubmit);
logoutBtn.addEventListener("click", handleLogout);
createUserForm.addEventListener("submit", handleCreateUser);
userList.addEventListener("click", handleUserListClick);
auditRefreshButton.addEventListener("click", refreshAuditLogs);

systemSelect.addEventListener("change", async () => {
  if (currentView === "metadata") {
    await refreshMetadataPanel();
  }
  if (currentView === "audit-logs") {
    await refreshAuditLogs();
  }
  if (currentView === "model-management") {
    await refreshModelManagement();
  }
});

anomalyCandidates.addEventListener("click", handleModelCardClick);
rcaCandidates.addEventListener("click", handleModelCardClick);
for (const item of navItems) {
  item.addEventListener("click", async () => {
    setActiveNav(item);
    switchView(item.dataset.view || "dashboard", item.dataset.scrollTarget || "");
    if (item.dataset.view === "metadata") {
      await refreshMetadataPanel();
    }
    if (item.dataset.view === "audit-logs") {
      await refreshAuditLogs();
    }
    if (item.dataset.view === "model-management") {
      await refreshModelManagement();
    }
  });
}

switchView("dashboard");
initializeApp();
