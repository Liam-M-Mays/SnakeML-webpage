/**
 * API utility functions for communicating with the backend.
 */

const API_BASE = "http://127.0.0.1:5000/api";

export const api = {
  // Environments
  getEnvironments: () =>
    fetch(`${API_BASE}/environments`).then((r) => r.json()),

  getEnvironmentMetadata: (envName) =>
    fetch(`${API_BASE}/environments/${envName}/metadata`).then((r) => r.json()),

  // Configuration
  getDefaultConfig: (algo) =>
    fetch(`${API_BASE}/config/default/${algo}`).then((r) => r.json()),

  // Training
  startTraining: (config) =>
    fetch(`${API_BASE}/training/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }).then((r) => r.json()),

  stopTraining: () =>
    fetch(`${API_BASE}/training/stop`, {
      method: "POST",
    }).then((r) => r.json()),

  getTrainingStatus: () =>
    fetch(`${API_BASE}/training/status`).then((r) => r.json()),

  // Models
  listModels: (envName = null) => {
    const url = envName
      ? `${API_BASE}/models?env_name=${envName}`
      : `${API_BASE}/models`;
    return fetch(url).then((r) => r.json());
  },

  deleteModel: (envName, runId) =>
    fetch(`${API_BASE}/models/${envName}/${runId}`, {
      method: "DELETE",
    }).then((r) => r.json()),

  getModelConfig: (envName, runId) =>
    fetch(`${API_BASE}/models/${envName}/${runId}/config`).then((r) =>
      r.json()
    ),

  // Metrics
  getMetrics: (envName, runId, limit = null) => {
    const url = limit
      ? `${API_BASE}/metrics/${envName}/${runId}?limit=${limit}`
      : `${API_BASE}/metrics/${envName}/${runId}`;
    return fetch(url).then((r) => r.json());
  },

  getDeathStats: (envName, runId) =>
    fetch(`${API_BASE}/metrics/${envName}/${runId}/death_stats`).then((r) =>
      r.json()
    ),

  // Replays
  listReplays: (envName, runId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}`).then((r) => r.json()),

  getReplay: (envName, runId, replayId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}/${replayId}`).then((r) =>
      r.json()
    ),

  deleteReplay: (envName, runId, replayId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}/${replayId}`, {
      method: "DELETE",
    }).then((r) => r.json()),

  // System
  getDeviceInfo: () => fetch(`${API_BASE}/device`).then((r) => r.json()),

  healthCheck: () => fetch(`${API_BASE}/health`).then((r) => r.json()),
};
