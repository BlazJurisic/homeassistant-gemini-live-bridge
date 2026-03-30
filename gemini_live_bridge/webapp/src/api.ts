const BASE = '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error || res.statusText);
  }
  return res.json();
}

// ── Status ─────────────────────────────────────────────────────

export interface BridgeStatus {
  provider: string;
  uptime_seconds: number;
  ha_connected: boolean;
  tools_loaded: number;
}

export function getStatus() {
  return request<BridgeStatus>('/api/status');
}

// ── Config ─────────────────────────────────────────────────────

export interface BridgeConfig {
  provider?: string;
  gemini_api_key?: string;
  gemini_voice?: string;
  openai_api_key?: string;
  openai_voice?: string;
  server_port?: number;
  log_level?: string;
  croatian_personality?: boolean;
  session_timeout_seconds?: number;
  [key: string]: unknown;
}

export function getConfig() {
  return request<BridgeConfig>('/api/config');
}

export function saveConfig(config: Record<string, unknown>) {
  return request<{ ok: boolean }>('/api/config', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

// ── Prompt ─────────────────────────────────────────────────────

export function getPrompt() {
  return request<{ prompt: string }>('/api/prompt');
}

export function savePrompt(prompt: string) {
  return request<{ ok: boolean }>('/api/prompt', {
    method: 'POST',
    body: JSON.stringify({ prompt }),
  });
}

// ── Tools ──────────────────────────────────────────────────────

export interface ToolSummary {
  name: string;
  description: string;
  builtin: boolean;
}

export interface ToolDetail {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
  code: string | null;
  builtin: boolean;
}

export function getTools() {
  return request<ToolSummary[]>('/api/tools');
}

export function getTool(name: string) {
  return request<ToolDetail>(`/api/tools/${name}`);
}

export function createTool(name: string, code: string) {
  return request<{ ok: boolean; name: string }>('/api/tools', {
    method: 'POST',
    body: JSON.stringify({ name, code }),
  });
}

export function updateTool(name: string, code: string) {
  return request<{ ok: boolean }>(`/api/tools/${name}`, {
    method: 'PUT',
    body: JSON.stringify({ code }),
  });
}

export function deleteTool(name: string) {
  return request<{ ok: boolean }>(`/api/tools/${name}`, {
    method: 'DELETE',
  });
}

// ── Devices ────────────────────────────────────────────────────

export interface HADevice {
  entity_id: string;
  domain: string;
  state: string;
  friendly_name: string;
  attributes: Record<string, unknown>;
}

export function getDevices() {
  return request<HADevice[]>('/api/devices');
}
