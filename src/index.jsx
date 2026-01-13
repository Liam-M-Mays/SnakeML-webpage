import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { SocketProvider, GameProvider, AgentProvider } from './context'

// This finds the div with id="root" and puts our React app inside it
const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(
  <SocketProvider>
    <GameProvider>
      <AgentProvider>
        <App />
      </AgentProvider>
    </GameProvider>
  </SocketProvider>
)
