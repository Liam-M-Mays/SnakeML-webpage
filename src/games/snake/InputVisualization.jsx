import React, { useState, useEffect } from 'react'

/**
 * Snake-specific network input visualization.
 * Shows what features the AI is receiving in real-time.
 *
 * This is game-specific because it knows about snake features:
 * food direction, path distance, hunger, body segments, danger zones.
 */
export default function InputVisualization({ socket, isTraining, gameSettings }) {
  const [inputState, setInputState] = useState(null)
  const [isExpanded, setIsExpanded] = useState(false)

  useEffect(() => {
    if (!socket) return

    const handleInputState = (data) => {
      setInputState(data)
    }

    socket.on('input_state', handleInputState)

    return () => {
      socket.off('input_state', handleInputState)
    }
  }, [socket])

  if (!isTraining || !inputState) {
    return null
  }

  const cfg = gameSettings?.inputs || {}
  const { features, labels } = inputState

  // Build feature groups based on config
  const featureGroups = []
  let idx = 0

  if (cfg.foodDirection !== false && features) {
    featureGroups.push({
      name: 'Food Direction',
      values: [
        { label: 'X', value: features[idx]?.toFixed(1) || '0', color: features[idx] > 0 ? '#4CAF50' : features[idx] < 0 ? '#f44336' : '#888' },
        { label: 'Y', value: features[idx + 1]?.toFixed(1) || '0', color: features[idx + 1] > 0 ? '#4CAF50' : features[idx + 1] < 0 ? '#f44336' : '#888' },
      ]
    })
    idx += 2
  }

  if (cfg.pathDistance !== false && features) {
    const dist = features[idx] || 0
    featureGroups.push({
      name: 'Path Distance',
      values: [
        { label: 'Dist', value: (dist * 100).toFixed(0) + '%', color: dist < 0.2 ? '#4CAF50' : dist < 0.5 ? '#FF9800' : '#f44336' },
      ]
    })
    idx += 1
  }

  if (cfg.currentDirection !== false && features) {
    featureGroups.push({
      name: 'Direction',
      values: [
        { label: 'X', value: features[idx]?.toFixed(0) || '0', color: '#2196F3' },
        { label: 'Y', value: features[idx + 1]?.toFixed(0) || '0', color: '#2196F3' },
      ]
    })
    idx += 2
  }

  if (cfg.hunger !== false && features) {
    const hunger = features[idx] || 0
    featureGroups.push({
      name: 'Hunger',
      values: [
        { label: '', value: (hunger * 100).toFixed(0) + '%', color: hunger < 0.3 ? '#4CAF50' : hunger < 0.7 ? '#FF9800' : '#f44336' },
      ]
    })
    idx += 1
  }

  // Segment features - show chained dx,dy pairs
  if (cfg.segments !== false && features) {
    const segCount = cfg.segmentCount ?? 10
    const segmentValues = []
    for (let i = 0; i < segCount && idx + 1 < features.length; i++) {
      const dx = features[idx] || 0
      const dy = features[idx + 1] || 0
      segmentValues.push({
        label: i === segCount - 1 ? 'T' : String(i + 1),
        value: `(${dx.toFixed(2)}, ${dy.toFixed(2)})`,
        color: '#9E9E9E'
      })
      idx += 2
    }
    if (segmentValues.length > 0) {
      featureGroups.push({
        name: `Segments (${segCount})`,
        values: segmentValues,
        isSegments: true
      })
    }
  }

  // Danger features
  if (cfg.danger !== false && features) {
    const visionRange = cfg.visionRange ?? 1
    let dangerValues = []

    if (visionRange === 0) {
      // 3 values: left, right, forward
      const labels = ['L', 'R', 'F']
      for (let i = 0; i < 3 && idx < features.length; i++) {
        const val = features[idx] || 0
        dangerValues.push({
          label: labels[i],
          value: val === 1 ? '!' : '-',
          color: val === 1 ? '#f44336' : '#4CAF50'
        })
        idx += 1
      }
    } else {
      // Window mode: (2*vision+1)^2 - 1 values
      const windowSize = (2 * visionRange + 1) ** 2 - 1
      const dangerCount = features.slice(idx, idx + windowSize).filter(d => d === 1).length
      dangerValues.push({
        label: 'Nearby',
        value: `${dangerCount}/${windowSize}`,
        color: dangerCount > 0 ? '#f44336' : '#4CAF50'
      })
      idx += windowSize
    }

    if (dangerValues.length > 0) {
      featureGroups.push({
        name: visionRange === 0 ? 'Danger (L/R/F)' : `Danger (${visionRange})`,
        values: dangerValues
      })
    }
  }

  // Snake length (if enabled)
  if (cfg.snakeLength === true && features && idx < features.length) {
    const length = features[idx] || 0
    featureGroups.push({
      name: 'Length',
      values: [
        { label: '', value: (length * 100).toFixed(0) + '%', color: '#9C27B0' }
      ]
    })
    idx += 1
  }

  return (
    <div className="input-visualization">
      <div
        className="input-viz-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span>Network Inputs</span>
        <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
      </div>

      {isExpanded && (
        <div className="input-viz-content">
          {featureGroups.map((group, i) => (
            <div key={i} className={`feature-group ${group.isSegments ? 'segments-group' : ''}`}>
              <span className="feature-name">{group.name}</span>
              <div className={`feature-values ${group.isSegments ? 'segments-values' : ''}`}>
                {group.values.map((v, j) => (
                  <span
                    key={j}
                    className={`feature-value ${group.isSegments ? 'segment-value' : ''}`}
                    style={{ color: v.color }}
                  >
                    {v.label && <span className="feature-label">{v.label}:</span>}
                    {v.value}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <div className="feature-group">
            <span className="feature-name">Total Features</span>
            <span className="feature-value">{features?.length || 0}</span>
          </div>
          <div className="feature-group">
            <span className="feature-name">Parsed Index</span>
            <span className="feature-value" style={{ color: idx === features?.length ? '#4CAF50' : '#f44336' }}>
              {idx} / {features?.length || 0} {idx !== features?.length && '⚠'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
