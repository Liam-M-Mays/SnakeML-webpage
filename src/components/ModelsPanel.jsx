import { useEffect, useState } from 'react';
import '../styles/ModelsPanel.css';

export default function ModelsPanel({ onSelect }) {
  const [runs, setRuns] = useState([]);

  const loadRuns = () => {
    fetch('/api/runs')
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => setRuns([]));
  };

  useEffect(() => {
    loadRuns();
  }, []);

  const handleDelete = async (env, run_id) => {
    await fetch(`/api/runs/${env}/${run_id}`, { method: 'DELETE' });
    loadRuns();
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Models</h2>
        <button className="ghost" onClick={loadRuns}>Refresh</button>
      </div>
      <div className="scroll">
        {runs.map((run) => (
          <div key={`${run.env}-${run.run_id}`} className="model-row">
            <div>
              <strong>{run.run_id}</strong>
              <div className="muted">Env: {run.env}</div>
            </div>
            <div className="muted">Best reward: {run.best_reward ?? 'n/a'}</div>
            <div className="model-actions">
              <button className="primary" onClick={() => onSelect(run)}>Load</button>
              <button className="ghost" onClick={() => handleDelete(run.env, run.run_id)}>Delete</button>
            </div>
          </div>
        ))}
        {runs.length === 0 && <p className="muted">No saved models yet.</p>}
      </div>
    </div>
  );
}
