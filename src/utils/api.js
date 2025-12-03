/**
 * API utility functions for communicating with the backend.
 */

const API_BASE = "http://127.0.0.1:5000/api";

async function handleResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `HTTP error ${response.status}`);
  }
  return data;
}

export const api = {
  // Environments
  getEnvironments: () =>
    fetch(`${API_BASE}/environments`).then(handleResponse),

  getEnvironmentMetadata: (envName) =>
    fetch(`${API_BASE}/environments/${envName}/metadata`).then(handleResponse),

  // Configuration
  getDefaultConfig: (algo) =>
    fetch(`${API_BASE}/config/default/${algo}`).then(handleResponse),

  // Training
  startTraining: (config) =>
    fetch(`${API_BASE}/training/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }).then(handleResponse),

  stopTraining: () =>
    fetch(`${API_BASE}/training/stop`, {
      method: "POST",
    }).then(handleResponse),

  resumeTraining: (envName, runId, continueTraining = true, maxSpeed = false) =>
    fetch(`${API_BASE}/training/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        env_name: envName,
        run_id: runId,
        continue_training: continueTraining,
        max_speed: maxSpeed,
      }),
    }).then(handleResponse),

  getTrainingStatus: () =>
    fetch(`${API_BASE}/training/status`).then(handleResponse),

  // Models
  listModels: (envName = null) => {
    const url = envName
      ? `${API_BASE}/models?env_name=${envName}`
      : `${API_BASE}/models`;
    return fetch(url).then(handleResponse);
  },

  deleteModel: (envName, runId) =>
    fetch(`${API_BASE}/models/${envName}/${runId}`, {
      method: "DELETE",
    }).then(handleResponse),

  getModelConfig: (envName, runId) =>
    fetch(`${API_BASE}/models/${envName}/${runId}/config`).then(handleResponse),

  // Metrics
  getMetrics: (envName, runId, limit = null) => {
    const url = limit
      ? `${API_BASE}/metrics/${envName}/${runId}?limit=${limit}`
      : `${API_BASE}/metrics/${envName}/${runId}`;
    return fetch(url).then(handleResponse);
  },

  getDeathStats: (envName, runId) =>
    fetch(`${API_BASE}/metrics/${envName}/${runId}/death_stats`).then(
      handleResponse
    ),

  // Replays
  listReplays: (envName, runId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}`).then(handleResponse),

  getReplay: (envName, runId, replayId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}/${replayId}`).then(
      handleResponse
    ),

  deleteReplay: (envName, runId, replayId) =>
    fetch(`${API_BASE}/replays/${envName}/${runId}/${replayId}`, {
      method: "DELETE",
    }).then(handleResponse),

  // Device
  getDeviceInfo: () =>
    fetch(`${API_BASE}/device`).then(handleResponse),

  setDevice: (preference) =>
    fetch(`${API_BASE}/device/set`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device: preference }),
    }).then(handleResponse),

  // System
  healthCheck: () =>
    fetch(`${API_BASE}/health`).then(handleResponse),
};
