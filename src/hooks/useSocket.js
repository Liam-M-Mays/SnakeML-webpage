import { useEffect, useMemo, useState } from 'react';
import { io } from 'socket.io-client';

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://localhost:5000';

export function useSocket(events = []) {
  const socket = useMemo(() => io(SOCKET_URL, { transports: ['websocket', 'polling'] }), []);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    socket.on('connect', () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));
    events.forEach(({ name, handler }) => socket.on(name, handler));
    return () => {
      socket.off('connect');
      socket.off('disconnect');
      events.forEach(({ name, handler }) => socket.off(name, handler));
      socket.close();
    };
  }, [socket, events]);

  return { socket, connected };
}
