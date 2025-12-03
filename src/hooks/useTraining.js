/**
 * Custom hook for managing training state and metrics.
 */

import { useState, useEffect, useCallback } from "react";
import { api } from "../utils/api";
import { useSocket } from "./useSocket";

export function useTraining() {
  const [isTraining, setIsTraining] = useState(false);
  const [runId, setRunId] = useState(null);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(null);
  const [episodes, setEpisodes] = useState([]);
  const [error, setError] = useState(null);

  const { on, off } = useSocket();

  // Fetch training status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await api.getTrainingStatus();
        setIsTraining(data.is_running || false);
        setStatus(data.is_running ? data : null);
        if (data.is_running) {
          setRunId(data.run_id);
        }
      } catch (err) {
        console.error("Failed to fetch training status:", err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);

    return () => clearInterval(interval);
  }, []);

  // Listen for training events
  useEffect(() => {
    const handleProgress = (data) => {
      setProgress(data);
    };

    const handleEpisode = (data) => {
      setEpisodes((prev) => [...prev.slice(-99), data]); // Keep last 100
    };

    on("training_progress", handleProgress);
    on("episode_summary", handleEpisode);

    return () => {
      off("training_progress", handleProgress);
      off("episode_summary", handleEpisode);
    };
  }, [on, off]);

  const startTraining = useCallback(async (config) => {
    try {
      setError(null);
      setEpisodes([]);
      const result = await api.startTraining(config);
      setIsTraining(true);
      setRunId(result.run_id);
      setStatus(result.status);
      return result;
    } catch (err) {
      setError(err.message || "Failed to start training");
      throw err;
    }
  }, []);

  const stopTraining = useCallback(async () => {
    try {
      setError(null);
      const result = await api.stopTraining();
      setIsTraining(false);
      setProgress(null);
      return result;
    } catch (err) {
      setError(err.message || "Failed to stop training");
      throw err;
    }
  }, []);

  return {
    isTraining,
    runId,
    status,
    progress,
    episodes,
    error,
    startTraining,
    stopTraining,
  };
}
