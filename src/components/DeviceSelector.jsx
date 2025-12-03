/**
 * Device selection component for choosing compute device (CPU/GPU).
 */

import React from "react";

function DeviceSelector({ deviceInfo, onDeviceChange, disabled }) {
  if (!deviceInfo) {
    return (
      <div className="device-selector">
        <p className="info-text">Loading device info...</p>
      </div>
    );
  }

  const handleChange = (e) => {
    if (onDeviceChange) {
      onDeviceChange(e.target.value);
    }
  };

  const getDeviceIcon = (type) => {
    switch (type?.toLowerCase()) {
      case "cuda":
        return "GPU";
      case "mps":
        return "M1/M2";
      default:
        return "CPU";
    }
  };

  return (
    <div className="device-selector">
      <h3>Compute Device</h3>

      <div className="device-current">
        <div className="device-icon">{getDeviceIcon(deviceInfo.type)}</div>
        <div>
          <div className="device-name">{deviceInfo.name || deviceInfo.type}</div>
          <div className="device-type">
            {deviceInfo.type?.toUpperCase() || "CPU"}
          </div>
        </div>
      </div>

      <div className="device-select-group">
        <label htmlFor="device-select">Device Preference</label>
        <select
          id="device-select"
          value={deviceInfo.preference || "auto"}
          onChange={handleChange}
          disabled={disabled}
        >
          <option value="auto">Auto (recommended)</option>
          <option value="cpu">CPU</option>
          {deviceInfo.available?.includes("cuda") && (
            <option value="cuda">CUDA (NVIDIA GPU)</option>
          )}
          {deviceInfo.available?.includes("mps") && (
            <option value="mps">MPS (Apple Silicon)</option>
          )}
        </select>
      </div>

      <div className="info-text">
        Available devices:{" "}
        {deviceInfo.available?.map((d) => d.toUpperCase()).join(", ") || "CPU"}
      </div>

      {deviceInfo.fallback_reason && (
        <div className="device-warning">{deviceInfo.fallback_reason}</div>
      )}

      {deviceInfo.details?.cuda_version && (
        <div className="info-text">CUDA version: {deviceInfo.details.cuda_version}</div>
      )}

      {deviceInfo.details?.gpu_name && (
        <div className="info-text">GPU: {deviceInfo.details.gpu_name}</div>
      )}
    </div>
  );
}

export default DeviceSelector;
