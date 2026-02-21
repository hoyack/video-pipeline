import clsx from "clsx";

const STAGE_LABELS = ["Storyboard", "Keyframes", "Video", "Stitch", "All"];
const RUN_THROUGH_MAP: (string | null)[] = ["storyboard", "keyframes", "video", null, null];

interface GenerateThroughSliderProps {
  value: number; // 0-4
  onChange: (v: number) => void;
  stageCompletion?: boolean[]; // which stages have completed content
  disabled?: boolean;
}

export function GenerateThroughSlider({
  value,
  onChange,
  stageCompletion,
  disabled,
}: GenerateThroughSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <label className="text-xs font-medium text-gray-400">Generate Through</label>
        <span className="text-xs font-semibold text-cyan-400">{STAGE_LABELS[value]}</span>
      </div>
      <input
        type="range"
        min={0}
        max={4}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full accent-cyan-500 disabled:opacity-50"
      />
      <div className="flex justify-between">
        {STAGE_LABELS.map((label, i) => (
          <span
            key={label}
            className={clsx(
              "text-[10px] font-medium",
              stageCompletion?.[i]
                ? "text-green-400"
                : i === value
                  ? "text-cyan-400"
                  : "text-gray-600",
            )}
          >
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Convert slider index (0-4) to run_through API value */
export function sliderToRunThrough(index: number): string | null {
  return RUN_THROUGH_MAP[index] ?? null;
}
