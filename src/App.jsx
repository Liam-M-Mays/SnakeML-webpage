import { useEffect, useState } from 'react';
import TrainingDashboard from './components/TrainingDashboard';
import RewardDesigner from './components/RewardDesigner';
import NetworkBuilder from './components/NetworkBuilder';
import ReplaysPanel from './components/ReplaysPanel';
import ModelsPanel from './components/ModelsPanel';
import './styles/App.css';

export default function App() {
  const [envs, setEnvs] = useState([]);
  const [env, setEnv] = useState('snake');
  const [runId, setRunId] = useState('');
  const [hyper, setHyper] = useState({
    maxEpisodes: 10,
    gamma: 0.99,
    learningRate: 0.001,
    batchSize: 64,
    epsilonStart: 1,
    epsilonEnd: 0.1,
    epsilonDecay: 0.99,
    replaySize: 2000,
  });
  const [network, setNetwork] = useState({ type: 'dense', hidden: [] });
  const [reward, setReward] = useState({});

  useEffect(() => {
    fetch('/api/envs')
      .then((r) => r.json())
      .then((data) => {
        setEnvs(data || []);
        if (data?.length && !env) setEnv(data[0].name);
      })
      .catch(() => {});
  }, []);

  const startRun = async () => {
    const payload = {
      run_id: runId || `run-${Date.now()}`,
      env,
      algo: 'dqn',
      hyperparameters: hyper,
      reward_config: reward,
      network,
    };
    const res = await fetch('/api/runs/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setRunId(data.run_id);
  };

  const updateHyper = (key, value) => setHyper((h) => ({ ...h, [key]: Number(value) }));

  return (
    <div className="layout">
      <header className="app-header">
        <div>
          <h1>Snake RL Playground</h1>
          <p className="muted">Train, tweak rewards, and inspect replays.</p>
        </div>
        <div className="header-actions">
          <select value={env} onChange={(e) => setEnv(e.target.value)}>
            {envs.map((e) => (
              <option key={e.name} value={e.name}>{e.name}</option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Run ID"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
          />
          <button className="primary" onClick={startRun}>Start training</button>
        </div>
      </header>

      <main className="grid three">
        <section>
          <div className="card">
            <h2>Hyperparameters</h2>
            <div className="form-grid">
              {Object.entries(hyper).map(([k, v]) => (
                <label key={k}>
                  <span>{k}</span>
                  <input type="number" value={v} onChange={(e) => updateHyper(k, e.target.value)} />
                </label>
              ))}
            </div>
          </div>
          <RewardDesigner envName={env} onRewardChange={setReward} />
          <NetworkBuilder value={network} onChange={setNetwork} />
        </section>
        <section>
          <TrainingDashboard currentRun={runId} env={env} />
          <ReplaysPanel env={env} runId={runId} />
        </section>
        <section>
          <ModelsPanel onSelect={(run) => { setEnv(run.env); setRunId(run.run_id); }} />
          <div className="card">
            <h2>Notes</h2>
            <p className="muted">Training updates stream live via Socket.IO. Use the reward designer and network builder to craft custom runs.</p>
          </div>
        </section>
      </main>
    </div>
  );
}
