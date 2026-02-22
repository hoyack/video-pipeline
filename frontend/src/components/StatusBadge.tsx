import clsx from "clsx";

const STATUS_COLORS: Record<string, string> = {
  pending: "bg-gray-600 text-gray-200",
  storyboarding: "bg-blue-600 text-blue-100",
  keyframing: "bg-indigo-600 text-indigo-100",
  video_gen: "bg-purple-600 text-purple-100",
  stitching: "bg-amber-600 text-amber-100",
  complete: "bg-green-600 text-green-100",
  failed: "bg-red-600 text-red-100",
  stopped: "bg-amber-600 text-amber-100",
  staged: "bg-cyan-600 text-cyan-100",
};

const LABELS: Record<string, string> = {
  pending: "Pending",
  storyboarding: "Storyboarding",
  keyframing: "Keyframing",
  video_gen: "Video Gen",
  stitching: "Stitching",
  complete: "Complete",
  failed: "Failed",
  stopped: "Stopped",
  staged: "Staged",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        "inline-block rounded-full px-2.5 py-0.5 text-xs font-medium",
        STATUS_COLORS[status] ?? "bg-gray-600 text-gray-200",
      )}
    >
      {LABELS[status] ?? status}
    </span>
  );
}
