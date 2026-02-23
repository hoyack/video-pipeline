import { useEffect, useRef, useState, useCallback } from "react";
import type { WsEvent } from "../api/wsTypes.ts";

interface UseSceneWebSocketOptions {
  sceneId: string | undefined;
  enabled: boolean;
  onEvent: (event: WsEvent) => void;
}

interface UseSceneWebSocketResult {
  connected: boolean;
  error: string | null;
}

const MAX_RECONNECT_ATTEMPTS = 5;
const MAX_RECONNECT_DELAY = 30_000;
const CLIENT_PING_INTERVAL = 25_000;

export function useSceneWebSocket({
  sceneId,
  enabled,
  onEvent,
}: UseSceneWebSocketOptions): UseSceneWebSocketResult {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const cleanup = useCallback(() => {
    if (connectTimerRef.current) {
      clearTimeout(connectTimerRef.current);
      connectTimerRef.current = null;
    }
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const connect = useCallback(() => {
    if (!sceneId || !enabled) return;

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}/api/scenes/${sceneId}/ws`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectCountRef.current = 0;
      setConnected(true);
      setError(null);

      // Client-side ping to keep connection alive
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, CLIENT_PING_INTERVAL);
    };

    ws.onmessage = (msg) => {
      try {
        const event: WsEvent = JSON.parse(msg.data);
        if (event.type === "heartbeat") return;
        onEventRef.current(event);
      } catch {
        // Ignore unparseable messages
      }
    };

    ws.onclose = () => {
      cleanup();
      // Auto-reconnect with exponential backoff
      if (enabled && reconnectCountRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          1000 * Math.pow(2, reconnectCountRef.current),
          MAX_RECONNECT_DELAY,
        );
        reconnectCountRef.current += 1;
        reconnectTimerRef.current = setTimeout(connect, delay);
      } else if (reconnectCountRef.current >= MAX_RECONNECT_ATTEMPTS) {
        setError("WebSocket connection lost");
      }
    };

    ws.onerror = () => {
      // onerror is always followed by onclose; no special handling needed
    };
  }, [sceneId, enabled, cleanup]);

  useEffect(() => {
    if (enabled && sceneId) {
      reconnectCountRef.current = 0;
      // Defer connection to next tick so React Strict Mode's mount→unmount→mount
      // cycle cancels the first attempt before the WebSocket is actually created.
      connectTimerRef.current = setTimeout(connect, 0);
    } else {
      cleanup();
    }
    return cleanup;
  }, [enabled, sceneId, connect, cleanup]);

  return { connected, error };
}
