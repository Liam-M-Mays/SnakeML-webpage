import React, { useEffect, useRef, useState } from 'react'

/**
 * Simple canvas-based line chart component.
 * No external dependencies - renders directly to canvas.
 */
function LineChart({ data, label, color, height = 120, maxPoints = 100 }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    // Set canvas size based on container
    const rect = container.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    canvas.style.width = `${rect.width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const width = rect.width
    const h = height

    // Clear canvas
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, h)

    // Handle empty data
    if (!data || data.length === 0) {
      ctx.fillStyle = '#666'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('No data yet', width / 2, h / 2)
      return
    }

    // Limit data points
    const displayData = data.slice(-maxPoints)

    // Calculate min/max for scaling
    const minVal = Math.min(...displayData)
    const maxVal = Math.max(...displayData)
    const range = maxVal - minVal || 1
    const padding = { top: 25, bottom: 20, left: 50, right: 10 }

    const chartWidth = width - padding.left - padding.right
    const chartHeight = h - padding.top - padding.bottom

    // Draw grid lines
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartHeight / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.stroke()
    }

    // Draw Y-axis labels
    ctx.fillStyle = '#888'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const value = maxVal - (range / 4) * i
      const y = padding.top + (chartHeight / 4) * i
      ctx.fillText(value.toFixed(1), padding.left - 5, y + 3)
    }

    // Draw data line
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()

    displayData.forEach((value, index) => {
      const x = padding.left + (index / (displayData.length - 1 || 1)) * chartWidth
      const y = padding.top + ((maxVal - value) / range) * chartHeight

      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw label
    ctx.fillStyle = color
    ctx.font = 'bold 11px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(label, padding.left, 14)

    // Draw latest value
    if (displayData.length > 0) {
      const latest = displayData[displayData.length - 1]
      ctx.textAlign = 'right'
      ctx.fillText(latest.toFixed(2), width - padding.right, 14)
    }

  }, [data, label, color, height, maxPoints])

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: `${height}px`,
          borderRadius: '4px',
          background: '#1a1a2e'
        }}
      />
    </div>
  )
}

/**
 * Bar chart for expert weight visualization (MANN).
 */
function ExpertWeightsChart({ weights, height = 100, isRealtime = false }) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    // Set canvas size based on container
    const rect = container.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    canvas.style.width = `${rect.width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const width = rect.width
    const h = height

    // Clear canvas
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, h)

    if (!weights || weights.length === 0) {
      ctx.fillStyle = '#666'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('No expert data yet', width / 2, h / 2)
      return
    }

    const padding = { top: 25, bottom: 25, left: 20, right: 20 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = h - padding.top - padding.bottom

    const barWidth = Math.min(60, (chartWidth / weights.length) - 10)
    const totalBarsWidth = weights.length * barWidth + (weights.length - 1) * 10
    const startX = padding.left + (chartWidth - totalBarsWidth) / 2

    const colors = ['#4ecdc4', '#ff6b6b', '#ffd93d', '#6bcb77', '#845ec2', '#ff9671', '#00c9a7', '#c34a36']

    // Draw label with indicator for real-time vs averaged
    ctx.fillStyle = '#aaa'
    ctx.font = 'bold 11px monospace'
    ctx.textAlign = 'left'
    const label = isRealtime ? 'Expert Weights (Live)' : 'Expert Weights (Avg)'
    ctx.fillText(label, padding.left, 14)

    // Draw live indicator dot if real-time
    if (isRealtime) {
      ctx.fillStyle = '#4ecdc4'
      ctx.beginPath()
      ctx.arc(padding.left + ctx.measureText(label).width + 10, 10, 4, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw bars
    weights.forEach((weight, index) => {
      const x = startX + index * (barWidth + 10)
      const barHeight = Math.max(2, weight * chartHeight)
      const y = padding.top + chartHeight - barHeight

      // Bar
      ctx.fillStyle = colors[index % colors.length]
      ctx.fillRect(x, y, barWidth, barHeight)

      // Weight value on top
      ctx.fillStyle = '#fff'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText((weight * 100).toFixed(0) + '%', x + barWidth / 2, y - 5)

      // Expert label below
      ctx.fillStyle = '#888'
      ctx.fillText(`Expert ${index + 1}`, x + barWidth / 2, h - 8)
    })

  }, [weights, height, isRealtime])

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: `${height}px`,
          borderRadius: '4px',
          background: '#1a1a2e'
        }}
      />
    </div>
  )
}

/**
 * Training metrics dashboard component.
 * Displays real-time charts for rewards, scores, losses, and expert weights.
 */
function TrainingChart({ socket, isTraining, agentType, gameSpeed }) {
  const [metrics, setMetrics] = useState({
    rewards: [],
    scores: [],
    losses: [],
    expert_weights: [],
    episodes: 0
  })

  // Real-time expert weights (updated every step during normal speed training)
  const [realtimeWeights, setRealtimeWeights] = useState([])

  useEffect(() => {
    if (!socket) return

    const handleMetrics = (data) => {
      setMetrics({
        rewards: data.rewards || [],
        scores: data.scores || [],
        losses: data.losses || [],
        expert_weights: data.expert_weights || [],
        episodes: data.episodes || 0
      })
    }

    // Real-time expert weights from step-by-step training
    const handleRealtimeWeights = (data) => {
      if (data.weights && data.weights.length > 0) {
        setRealtimeWeights(data.weights)
      }
    }

    socket.on('training_metrics', handleMetrics)
    socket.on('expert_weights_realtime', handleRealtimeWeights)

    // Request initial metrics when training starts
    if (isTraining) {
      socket.emit('get_metrics')
    }

    return () => {
      socket.off('training_metrics', handleMetrics)
      socket.off('expert_weights_realtime', handleRealtimeWeights)
    }
  }, [socket, isTraining])

  // Reset metrics when training stops
  useEffect(() => {
    if (!isTraining) {
      setMetrics({
        rewards: [],
        scores: [],
        losses: [],
        expert_weights: [],
        episodes: 0
      })
      setRealtimeWeights([])
    }
  }, [isTraining])

  if (!isTraining) {
    return null
  }

  const isMANN = agentType === 'mann' || agentType === 'mapo'
  const isMaxSpeed = gameSpeed === 0

  // Determine which expert weights to display:
  // - During max speed (0ms): use averaged weights from training_metrics
  // - During normal speed: use real-time weights updated every step
  let displayWeights = []
  if (isMANN) {
    if (isMaxSpeed) {
      // Extract latest from averaged snapshots
      if (metrics.expert_weights && metrics.expert_weights.length > 0) {
        const lastSnapshot = metrics.expert_weights[metrics.expert_weights.length - 1]
        if (lastSnapshot && lastSnapshot.weights) {
          displayWeights = lastSnapshot.weights
        }
      }
    } else {
      // Use real-time weights
      displayWeights = realtimeWeights
    }
  }

  return (
    <div className="training-charts-container">
      <div className="training-charts-header">
        <h4>Training Metrics</h4>
        <span className="episode-badge">Episode: {metrics.episodes}</span>
      </div>

      <div className="charts-row">
        <div className="chart-panel">
          <LineChart
            data={metrics.rewards}
            label="Avg Reward"
            color="#4ecdc4"
            height={120}
          />
        </div>

        <div className="chart-panel">
          <LineChart
            data={metrics.scores}
            label="Score"
            color="#ffd93d"
            height={120}
          />
        </div>

        <div className="chart-panel">
          <LineChart
            data={metrics.losses}
            label="Loss"
            color="#ff6b6b"
            height={120}
          />
        </div>
      </div>

      {isMANN && (
        <div className="expert-weights-row">
          <ExpertWeightsChart
            weights={displayWeights}
            height={120}
            isRealtime={!isMaxSpeed}
          />
        </div>
      )}
    </div>
  )
}

export default TrainingChart
