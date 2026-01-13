import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { useSocket } from './SocketContext'

const AgentContext = createContext(null)

/**
 * AI Agent state and actions provider.
 * Manages agent creation, selection, training, and model save/load.
 */
export function AgentProvider({ children }) {
  const { socket } = useSocket()

  // Agent state
  const [agents, setAgents] = useState([])
  const [selectedAgentId, setSelectedAgentId] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [savedModels, setSavedModels] = useState([])

  // Derived values
  const selectedAgent = agents.find(a => a.id === selectedAgentId)

  // Socket listeners for model operations
  useEffect(() => {
    if (!socket) return

    // Request saved models on connect
    socket.emit('list_models')

    socket.on('models_list', (data) => {
      setSavedModels(data.models || [])
    })

    socket.on('save_model_result', (data) => {
      if (data.success) {
        console.log('Model saved:', data.filename)
        socket.emit('list_models')
      } else {
        console.error('Save failed:', data.error)
      }
    })

    socket.on('load_model_result', (data) => {
      if (data.success) {
        console.log('Model loaded:', data.filename)
      } else {
        console.error('Load failed:', data.error)
      }
    })

    socket.on('delete_model_result', (data) => {
      if (data.success) {
        socket.emit('list_models')
      }
    })

    return () => {
      socket.off('models_list')
      socket.off('save_model_result')
      socket.off('load_model_result')
      socket.off('delete_model_result')
    }
  }, [socket])

  // Agent CRUD operations
  const createAgent = useCallback((agent) => {
    setAgents(prev => [...prev, agent])
    setSelectedAgentId(agent.id)
  }, [])

  const updateAgentParams = useCallback((agentId, paramKey, value) => {
    setAgents(prev => prev.map(agent => {
      if (agent.id !== agentId) return agent
      return {
        ...agent,
        params: {
          ...agent.params,
          [paramKey]: { ...agent.params[paramKey], value }
        }
      }
    }))
  }, [])

  // Model operations
  const saveModel = useCallback((name, agentId, agentName, game) => {
    if (socket) {
      socket.emit('save_model', { name, agent_id: agentId, agent_name: agentName, game })
    }
  }, [socket])

  const deleteModel = useCallback((filename) => {
    if (socket) {
      socket.emit('delete_model', { filename })
    }
  }, [socket])

  const refreshModels = useCallback(() => {
    if (socket) {
      socket.emit('list_models')
    }
  }, [socket])

  // Training controls
  const startTraining = useCallback(() => {
    if (selectedAgent) {
      setIsTraining(true)
    }
  }, [selectedAgent])

  const stopTraining = useCallback(() => {
    setIsTraining(false)
    if (socket) {
      socket.emit('stop_loop')
    }
  }, [socket])

  const value = {
    // Agent state
    agents,
    selectedAgentId,
    selectedAgent,
    isTraining,
    savedModels,

    // Agent actions
    setSelectedAgentId,
    createAgent,
    updateAgentParams,

    // Training actions
    startTraining,
    stopTraining,
    setIsTraining,

    // Model actions
    saveModel,
    deleteModel,
    refreshModels,
  }

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  )
}

/**
 * Hook to access agent state and actions.
 */
export function useAgent() {
  const context = useContext(AgentContext)
  if (!context) {
    throw new Error('useAgent must be used within an AgentProvider')
  }
  return context
}

export default AgentContext
