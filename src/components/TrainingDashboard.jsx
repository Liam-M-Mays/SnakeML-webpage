import { useEffect, useMemo, useState } from 'react';
import { useSocket } from '../hooks/useSocket';
import '../styles/TrainingDashboard.css';

export default function TrainingDashboard({ currentRun, env }) {
  const [metrics, setMetrics] = useState([]);
  const [progress, setProgress] = useState(null);

  const events = useMemo(
    () => [
      {
        name: 'training_metrics',
        handler: (data) => {
          if (!currentRun || data.run_id === currentRun) {
            setMetrics((prev) => [...prev.slice(-99), data]);
          }
        },
      },
      {
        name: 'training_progress',
        handler: (data) => {
          if (!currentRun || data.run_id === currentRun) setProgress(data);
        },
      },
      {
        name: 'episode_summary',
        handler: (data) => {
          if (!currentRun || data.run_id === currentRun) setMetrics((prev) => [...prev.slice(-99), data]);
        },
      },
    ],
    [currentRun]
  );

  const { connected } = useSocket(events);

  useEffect(() => {
    if (!currentRun || !env) return;
    fetch(`/api/runs/${env}/${currentRun}/metrics`)
      .then((r) => r.json())
      .then((data) => setMetrics(data || []))
      .catch(() => {});
  }, [currentRun, env]);

  const latest = metrics[metrics.length - 1];
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2>Training Dashboard</h2>
          <p className="muted">Socket: {connected ? 'connected' : 'disconnected'}</p>
        </div>
        {progress && (
          <div className="badge">Episode {progress.episode} · {progress.episodes_per_second?.toFixed(2)} eps/s</div>
        )}
      </div>
      <div className="grid two">
        <div>
          <h3>Latest Metrics</h3>
          {latest ? (
            <ul className="metric-list">
              <li>Episode: {latest.episode}</li>
              <li>Reward: {latest.reward?.toFixed?.(3) ?? latest.reward}</li>
              <li>Length: {latest.length}</li>
              {latest.loss !== undefined && <li>Loss: {Number(latest.loss).toFixed(4)}</li>}
              {latest.epsilon !== undefined && <li>Epsilon: {Number(latest.epsilon).toFixed(3)}</li>}
              {latest.death_reason && <li>Death: {latest.death_reason}</li>}
            </ul>
          ) : (
            <p className="muted">No metrics yet.</p>
          )}
        </div>
        <div>
          <h3>Recent Episodes</h3>
          <div className="scroll">
            {metrics.slice(-20).map((m) => (
              <div key={`${m.run_id}-${m.episode}-${m.timestamp}`} className="metric-row">
                <div>Ep {m.episode}</div>
                <div>R {m.reward?.toFixed?.(2) ?? m.reward}</div>
                <div>L {m.length}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
