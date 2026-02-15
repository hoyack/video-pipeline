import { useCallback, useEffect, useState } from "react";
import { getMetrics } from "../api/client.ts";
import type { MetricsResponse } from "../api/types.ts";
import { TEXT_MODELS, IMAGE_MODELS, VIDEO_MODELS } from "../lib/constants.ts";

// ---------------------------------------------------------------------------
// Label lookups
// ---------------------------------------------------------------------------

const MODEL_LABELS: Record<string, string> = {};
for (const m of TEXT_MODELS) MODEL_LABELS[m.id] = m.label;
for (const m of IMAGE_MODELS) MODEL_LABELS[m.id] = m.label;
for (const m of VIDEO_MODELS) MODEL_LABELS[m.id] = m.label;

function displayLabel(key: string): string {
  return MODEL_LABELS[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Color palette for distribution bars
// ---------------------------------------------------------------------------

const PALETTE = [
  "bg-blue-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-violet-500",
  "bg-cyan-500",
  "bg-orange-500",
  "bg-pink-500",
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
        {label}
      </p>
      <p className="mt-1 text-2xl font-bold text-white">{value}</p>
    </div>
  );
}

function DistributionSection({
  title,
  data,
  labelFn = displayLabel,
}: {
  title: string;
  data: Record<string, number>;
  labelFn?: (key: string) => string;
}) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return null;

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-3 text-sm font-medium text-gray-400">{title}</h3>

      {/* Stacked bar */}
      <div className="mb-3 flex h-3 overflow-hidden rounded-full bg-gray-800">
        {entries.map(([key, count], i) => (
          <div
            key={key}
            className={`${PALETTE[i % PALETTE.length]} transition-all`}
            style={{ width: `${(count / total) * 100}%` }}
            title={`${labelFn(key)}: ${count}`}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {entries.map(([key, count], i) => (
          <div key={key} className="flex items-center gap-1.5 text-xs">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-sm ${PALETTE[i % PALETTE.length]}`}
            />
            <span className="text-gray-300">{labelFn(key)}</span>
            <span className="text-gray-500">
              {count} ({Math.round((count / total) * 100)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const data = await getMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  if (loading) {
    return (
      <p className="text-center text-sm text-gray-500">Loading metrics...</p>
    );
  }

  if (error) {
    return (
      <div className="text-center">
        <p className="text-sm text-red-400">{error}</p>
        <button
          onClick={fetchMetrics}
          className="mt-2 text-sm text-blue-400 hover:text-blue-300"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!metrics || metrics.total_projects === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No project data yet.</p>
        <p className="mt-1 text-sm text-gray-600">
          Generate some videos to see metrics here.
        </p>
      </div>
    );
  }

  const successRate =
    metrics.total_projects > 0
      ? Math.round(
          ((metrics.status_counts["complete"] ?? 0) / metrics.total_projects) *
            100,
        )
      : 0;

  return (
    <div>
      <h1 className="mb-4 text-2xl font-bold text-white">Dashboard</h1>

      {/* Summary cards */}
      <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <SummaryCard
          label="Total Projects"
          value={String(metrics.total_projects)}
        />
        <SummaryCard
          label="Video Seconds"
          value={String(metrics.total_video_seconds)}
        />
        <SummaryCard
          label="Est. Cost"
          value={`$${metrics.total_estimated_cost.toFixed(2)}`}
        />
        <SummaryCard label="Success Rate" value={`${successRate}%`} />
      </div>

      {/* Distribution sections */}
      <div className="grid gap-4 sm:grid-cols-2">
        <DistributionSection
          title="Status"
          data={metrics.status_counts}
          labelFn={(k) => k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ")}
        />
        <DistributionSection title="Style" data={metrics.style_counts} />
        <DistributionSection
          title="Text Model"
          data={metrics.text_model_counts}
        />
        <DistributionSection
          title="Image Model"
          data={metrics.image_model_counts}
        />
        <DistributionSection
          title="Video Model"
          data={metrics.video_model_counts}
        />
        <DistributionSection
          title="Audio"
          data={metrics.audio_counts}
          labelFn={(k) => k.charAt(0).toUpperCase() + k.slice(1)}
        />
        <DistributionSection
          title="Aspect Ratio"
          data={metrics.aspect_ratio_counts}
          labelFn={(k) => k}
        />
        <DistributionSection
          title="Scenes per Project"
          data={metrics.scene_count_counts}
          labelFn={(k) => `${k} scene${k === "1" ? "" : "s"}`}
        />
      </div>
    </div>
  );
}
