"use client";

import { useEffect, useRef, useState } from "react";
import { FARM_API } from "@/lib/api";

/**
 * Subscribe to an SSE endpoint and call onEvent for every payload.
 * Returns the EventSource instance for callers that need to close early.
 */
export function subscribeSSE(
  path: string,
  onEvent: (event: unknown) => void,
): EventSource {
  const url = path.startsWith("http") ? path : `${FARM_API}${path}`;
  const es = new EventSource(url);
  es.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data));
    } catch {
      // ignore malformed frame
    }
  };
  return es;
}

/**
 * useSSE — accumulates events from an SSE endpoint into a single rolling
 * state buffer. The reducer collapses every event into the latest snapshot
 * by `type`, so consumers always see the freshest of each kind.
 */
export function useLatestByType<T extends { type: string }>(
  path: string,
): Record<string, T> {
  const [byType, setByType] = useState<Record<string, T>>({});
  useEffect(() => {
    const es = subscribeSSE(path, (raw) => {
      const ev = raw as T;
      if (!ev || typeof ev.type !== "string") return;
      setByType((prev) => ({ ...prev, [ev.type]: ev }));
    });
    return () => es.close();
  }, [path]);
  return byType;
}

/**
 * useEventStream — accumulates every event into a ref-backed array for
 * components that need the full history (e.g., a run timeline).
 */
export function useEventStream<T extends { type: string }>(
  path: string,
  cap = 2000,
): T[] {
  const [events, setEvents] = useState<T[]>([]);
  const buf = useRef<T[]>([]);
  useEffect(() => {
    buf.current = [];
    setEvents([]);
    const es = subscribeSSE(path, (raw) => {
      const ev = raw as T;
      if (!ev || typeof ev.type !== "string") return;
      buf.current = [...buf.current.slice(-cap + 1), ev];
      setEvents(buf.current);
    });
    return () => es.close();
  }, [path, cap]);
  return events;
}
