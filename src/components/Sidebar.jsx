import React from 'react'

/**
 * Collapsible sidebar component.
 * Can be positioned on left or right, with custom title.
 */
export default function Sidebar({ title, side, collapsed, onToggle, children }) {
  return (
    <div className={`sidebar sidebar-${side} ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header" onClick={onToggle}>
        <span className="sidebar-title">{collapsed ? '' : title}</span>
        <button className="sidebar-toggle">
          {side === 'left'
            ? (collapsed ? '▶' : '◀')
            : (collapsed ? '◀' : '▶')
          }
        </button>
      </div>
      {!collapsed && (
        <div className="sidebar-content">
          {children}
        </div>
      )}
    </div>
  )
}
