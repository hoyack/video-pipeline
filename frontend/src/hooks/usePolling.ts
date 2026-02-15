import { useEffect, useRef } from "react";

/**
 * Generic polling hook. Calls `fn` immediately, then every `intervalMs`.
 * Stops when `enabled` is false. Cleans up on unmount.
 */
export function usePolling(
  fn: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean,
) {
  const savedFn = useRef(fn);
  savedFn.current = fn;

  useEffect(() => {
    if (!enabled) return;

    // Fire immediately
    savedFn.current();

    const id = setInterval(() => {
      savedFn.current();
    }, intervalMs);

    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}
