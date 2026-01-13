import React from 'react'

/**
 * Speed control slider for AI training.
 * Only shown during AI training.
 */
export default function SpeedControl({ speed, onChange, isVisible }) {
  if (!isVisible) return null

  const speedLabel = speed === 0 ? 'Max Speed (no render)' : `${speed}ms`

  return (
    <div className="speed-control">
      <label>Training Speed: {speedLabel}</label>
      <input
        type="range"
        min="0"
        max="500"
        step="10"
        value={speed}
        onChange={(e) => onChange(parseInt(e.target.value))}
      />
      <div className="speed-hints">
        <span>Max</span>
        <span>Slow</span>
      </div>
    </div>
  )
}
