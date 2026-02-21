import { useState } from "react";
import { CopyButton } from "./CopyButton.tsx";

interface PromptChainViewerProps {
  /** Base user-written prompt */
  basePrompt: string | null | undefined;
  /** Rewritten prompt (manifest-enhanced or LLM-refined) */
  rewrittenPrompt: string | null | undefined;
  /** Actual prompt sent to the model */
  sentPrompt: string | null | undefined;
  /** Label for the chain (e.g., "Start Keyframe", "Video Clip") */
  label: string;
  /** Compact mode — collapsed by default */
  defaultOpen?: boolean;
}

function PromptRow({
  tier,
  label,
  text,
}: {
  tier: "base" | "rewritten" | "sent";
  label: string;
  text: string;
}) {
  const tierColors = {
    base: "text-gray-500 bg-gray-800/50",
    rewritten: "text-blue-500 bg-blue-900/20",
    sent: "text-green-500 bg-green-900/20",
  };

  return (
    <div className="flex gap-2">
      <span
        className={`shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${tierColors[tier]}`}
      >
        {label}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-start gap-1">
          <p className="text-[11px] leading-relaxed text-gray-400 break-words">
            {text}
          </p>
          <CopyButton text={text} />
        </div>
      </div>
    </div>
  );
}

export function PromptChainViewer({
  basePrompt,
  rewrittenPrompt,
  sentPrompt,
  label,
  defaultOpen = false,
}: PromptChainViewerProps) {
  const [open, setOpen] = useState(defaultOpen);

  // Count available tiers
  const tiers = [basePrompt, rewrittenPrompt, sentPrompt].filter(Boolean);
  if (tiers.length === 0) return null;

  // If only one tier, just show it inline
  if (tiers.length === 1) {
    return (
      <div className="text-[11px]">
        <span className="font-medium text-gray-500">{label}:</span>{" "}
        <span className="text-gray-400">{tiers[0]}</span>
      </div>
    );
  }

  return (
    <div>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
        className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
      >
        <svg
          className={`h-3 w-3 transition-transform ${open ? "rotate-90" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        {label} prompt chain ({tiers.length} tiers)
      </button>
      {open && (
        <div className="mt-1 space-y-1.5 rounded border border-gray-800 bg-gray-950 p-2">
          {basePrompt && (
            <PromptRow tier="base" label="Base" text={basePrompt} />
          )}
          {rewrittenPrompt && (
            <PromptRow tier="rewritten" label="Rewritten" text={rewrittenPrompt} />
          )}
          {sentPrompt && (
            <PromptRow tier="sent" label="Sent" text={sentPrompt} />
          )}
        </div>
      )}
    </div>
  );
}
