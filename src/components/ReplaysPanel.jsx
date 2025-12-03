import { useEffect, useState } from 'react';
import '../styles/ReplaysPanel.css';

export default function ReplaysPanel({ env, runId }) {
  const [replays, setReplays] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (!env || !runId) return;
    fetch(`/api/runs/${env}/${runId}/replays`)
      .then((r) => r.json())
      .then(setReplays)
      .catch(() => setReplays([]));
  }, [env, runId]);

  useEffect(() => {
    if (!selected) return;
    fetch(`/api/runs/${env}/${runId}/replays/${selected}`)
      .then((r) => r.json())
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected, env, runId]);

  return (
    <div className="card">
      <div className="card-header">
        <h2>Replays</h2>
      </div>
      <div className="grid two">
        <div className="scroll">
          {replays.map((rp) => (
            <div key={rp.id} className={`replay-row ${selected === rp.id ? 'active' : ''}`} onClick={() => setSelected(rp.id)}>
              <div>Episode {rp.episode}</div>
              <div>Score {rp.score}</div>
              <div>{rp.death_reason}</div>
            </div>
          ))}
          {replays.length === 0 && <p className="muted">No replays yet.</p>}
        </div>
        <div>
          {detail ? (
            <div className="replay-detail">
              <p>Actions: {detail.actions?.length}</p>
              <p>Length: {detail.length}</p>
              <p>Death: {detail.death_reason}</p>
              <div className="actions-preview">
                {detail.actions?.slice(0, 50).map((a, idx) => (
                  <span key={idx} className="action-chip">{a}</span>
                ))}
              </div>
            </div>
          ) : (
            <p className="muted">Select a replay to inspect actions.</p>
          )}
        </div>
      </div>
    </div>
  );
}
