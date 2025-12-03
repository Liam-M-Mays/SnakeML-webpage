import { useEffect, useState } from 'react';
import '../styles/RewardDesigner.css';

export default function RewardDesigner({ envName, onRewardChange }) {
  const [config, setConfig] = useState({});
  const [status, setStatus] = useState('');

  useEffect(() => {
    if (!envName) return;
    fetch(`/api/rewards/${envName}`)
      .then((r) => r.json())
      .then((data) => {
        setConfig(data || {});
        onRewardChange?.(data || {});
      })
      .catch(() => {});
  }, [envName]);

  const updateField = (key, value) => {
    const next = { ...config, [key]: Number(value) };
    setConfig(next);
    onRewardChange?.(next);
  };

  const handleSave = async () => {
    await fetch(`/api/rewards/${envName}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    setStatus('Saved');
    setTimeout(() => setStatus(''), 1500);
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Reward Designer</h2>
        {status && <span className="badge">{status}</span>}
      </div>
      <div className="reward-grid">
        {Object.entries(config).map(([key, val]) => (
          <label key={key} className="reward-field">
            <span>{key}</span>
            <input
              type="number"
              step="0.01"
              value={val}
              onChange={(e) => updateField(key, e.target.value)}
            />
          </label>
        ))}
      </div>
      <button className="primary" onClick={handleSave}>Save reward defaults</button>
    </div>
  );
}
