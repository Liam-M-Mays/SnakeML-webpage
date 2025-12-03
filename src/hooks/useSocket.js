/**
 * Custom hook for Socket.IO connection and event handling.
 */

import { useEffect, useState } from "react";
import { io } from "socket.io-client";

let socket = null;

export function useSocket() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Create socket connection if it doesn't exist
    if (!socket) {
      socket = io("http://127.0.0.1:5000", {
        transports: ["websocket", "polling"],
        withCredentials: false,
      });

      socket.on("connect", () => {
        console.log("Socket connected");
        setIsConnected(true);
      });

      socket.on("disconnect", () => {
        console.log("Socket disconnected");
        setIsConnected(false);
      });
    }

    return () => {
      // Don't disconnect on unmount - keep connection alive
    };
  }, []);

  const on = (event, handler) => {
    if (socket) {
      socket.on(event, handler);
    }
  };

  const off = (event, handler) => {
    if (socket) {
      socket.off(event, handler);
    }
  };

  const emit = (event, data) => {
    if (socket) {
      socket.emit(event, data);
    }
  };

  return {
    socket,
    isConnected,
    on,
    off,
    emit,
  };
}
