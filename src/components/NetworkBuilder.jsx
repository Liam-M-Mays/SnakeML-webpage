import { useEffect, useState } from 'react';
import '../styles/NetworkBuilder.css';

const ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'linear'];

export default function NetworkBuilder({ value, onChange }) {
  const [layers, setLayers] = useState(value?.hidden || []);

  useEffect(() => {
    onChange?.({ type: 'dense', hidden: layers });
  }, [layers]);

  const addLayer = () => {
    setLayers([...layers, { units: 64, activation: 'relu' }]);
  };

  const updateLayer = (index, key, val) => {
    const next = layers.map((layer, i) => (i === index ? { ...layer, [key]: val } : layer));
    setLayers(next);
  };

  const removeLayer = (index) => {
    setLayers(layers.filter((_, i) => i !== index));
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Network Builder</h2>
        <button className="ghost" onClick={addLayer}>Add layer</button>
      </div>
      {layers.length === 0 && <p className="muted">No layers configured.</p>}
      <div className="layer-list">
        {layers.map((layer, idx) => (
          <div key={idx} className="layer-row">
            <span>Layer {idx + 1}</span>
            <label>
              Units
              <input
                type="number"
                value={layer.units}
                onChange={(e) => updateLayer(idx, 'units', Number(e.target.value))}
              />
            </label>
            <label>
              Activation
              <select value={layer.activation} onChange={(e) => updateLayer(idx, 'activation', e.target.value)}>
                {ACTIVATIONS.map((act) => (
                  <option key={act} value={act}>{act}</option>
                ))}
              </select>
            </label>
            <button className="ghost" onClick={() => removeLayer(idx)}>Remove</button>
          </div>
        ))}
      </div>
      <div className="network-visual">
        {layers.map((layer, idx) => (
          <div key={idx} className="network-node">
            <div className="node-title">Dense {layer.units}</div>
            <div className="node-sub">{layer.activation}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
