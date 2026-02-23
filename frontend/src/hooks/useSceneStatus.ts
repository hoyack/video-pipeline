import { useCallback, useState } from "react";
import { getProjectStatus } from "../api/client.ts";
import type { StatusResponse } from "../api/types.ts";
import { TERMINAL_STATUSES, SLOW_STAGES } from "../lib/constants.ts";
import { usePolling } from "./usePolling.ts";

const FAST_INTERVAL = 2000;
const SLOW_INTERVAL = 5000;

/**
 * Poll a project's status while it's running.
 * Slows down during video_gen, stops on terminal status.
 */
export function useProjectStatus(projectId: string | null) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isTerminal = status ? TERMINAL_STATUSES.has(status.status) : false;
  const isSlow = status ? SLOW_STAGES.has(status.status) : false;
  const interval = isSlow ? SLOW_INTERVAL : FAST_INTERVAL;

  const poll = useCallback(async () => {
    if (!projectId) return;
    try {
      const data = await getProjectStatus(projectId);
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Polling failed");
    }
  }, [projectId]);

  usePolling(poll, interval, !!projectId && !isTerminal);

  return { status, error, isTerminal };
}
