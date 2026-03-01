import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ScreenplayResponse,
  CharacterBreakdownEntry,
  SceneBreakdownEntry,
  ShotListEntry,
} from "../api/types.ts";
import {
  getScreenplay,
  updateScreenplay,
  generateScreenplayFull,
  generateScreenplayStep,
  updateScreenplayStatus,
  generateScenesFromScreenplay,
} from "../api/client.ts";

interface ScreenplayEditorProps {
  productionId: string;
  onScenesGenerated?: () => void;
}

type TabId = "logline" | "treatment" | "characters" | "breakdown" | "script" | "shotlist";

const TABS: { id: TabId; label: string }[] = [
  { id: "logline", label: "Logline" },
  { id: "treatment", label: "Treatment" },
  { id: "characters", label: "Character Breakdowns" },
  { id: "breakdown", label: "Scene Breakdown" },
  { id: "script", label: "Script" },
  { id: "shotlist", label: "Shot List" },
];

// Map tab IDs to the API step name for generation
const TAB_TO_STEP: Record<TabId, string> = {
  logline: "logline",
  treatment: "treatment",
  characters: "character-breakdowns",
  breakdown: "scene-breakdown",
  script: "script",
  shotlist: "shot-list",
};

export function ScreenplayEditor({ productionId, onScenesGenerated }: ScreenplayEditorProps) {
  const [screenplay, setScreenplay] = useState<ScreenplayResponse | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("logline");
  const [loading, setLoading] = useState(true);
  const [regenerating, setRegenerating] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatingFull, setGeneratingFull] = useState(false);
  const [generateScenesMsg, setGenerateScenesMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Debounce timer refs for text fields
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadScreenplay = useCallback(async () => {
    try {
      setLoading(true);
      const sp = await getScreenplay(productionId);
      setScreenplay(sp);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load screenplay");
    } finally {
      setLoading(false);
    }
  }, [productionId]);

  useEffect(() => {
    loadScreenplay();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [loadScreenplay]);

  const isLocked = screenplay?.status === "LOCKED";

  // ----- Save helpers -----

  async function saveField(field: string, value: unknown) {
    if (isLocked) return;
    setSaving(true);
    try {
      const updated = await updateScreenplay(productionId, { [field]: value });
      setScreenplay(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  function debouncedSave(field: string, value: string) {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => saveField(field, value), 2000);
  }

  // ----- Regeneration -----

  async function handleRegenerate(tab: TabId) {
    if (isLocked) return;
    const step = TAB_TO_STEP[tab];
    setRegenerating(step);
    setError(null);
    try {
      const updated = await generateScreenplayStep(productionId, step);
      setScreenplay(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to regenerate ${step}`);
    } finally {
      setRegenerating(null);
    }
  }

  // ----- Generate Full -----

  async function handleGenerateFull() {
    setGeneratingFull(true);
    setError(null);
    try {
      await generateScreenplayFull(productionId);
      // Start polling for progress
      pollRef.current = setInterval(async () => {
        try {
          const sp = await getScreenplay(productionId);
          setScreenplay(sp);
          if (!sp.generating_step) {
            // Generation complete
            if (pollRef.current) {
              clearInterval(pollRef.current);
              pollRef.current = null;
            }
            setGeneratingFull(false);
          }
        } catch {
          // Ignore poll errors
        }
      }, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start generation");
      setGeneratingFull(false);
    }
  }

  // ----- Status transitions -----

  async function handleStatusChange(newStatus: string) {
    setError(null);
    try {
      const updated = await updateScreenplayStatus(productionId, newStatus);
      setScreenplay(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update status");
    }
  }

  // ----- Generate Scenes -----

  async function handleGenerateScenes() {
    setError(null);
    setGenerateScenesMsg(null);
    try {
      const results = await generateScenesFromScreenplay(productionId);
      setGenerateScenesMsg(`Created ${results.length} scene${results.length !== 1 ? "s" : ""} from screenplay`);
      onScenesGenerated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate scenes");
    }
  }

  // ----- Render helpers -----

  if (loading) {
    return <div className="text-center py-8 text-gray-400 text-sm">Loading screenplay...</div>;
  }

  if (!screenplay) {
    return <div className="text-center py-8 text-red-400 text-sm">Failed to load screenplay</div>;
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded-md bg-red-900/50 p-3 text-sm text-red-300 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200 ml-3">Dismiss</button>
        </div>
      )}

      {generateScenesMsg && (
        <div className="rounded-md bg-green-900/50 p-3 text-sm text-green-300 flex justify-between">
          <span>{generateScenesMsg}</span>
          <button onClick={() => setGenerateScenesMsg(null)} className="text-green-400 hover:text-green-200 ml-3">Dismiss</button>
        </div>
      )}

      {/* Status bar */}
      <div className="flex items-center justify-between rounded-lg border border-gray-700 bg-gray-800/50 p-3">
        <div className="flex items-center gap-3">
          <StatusBadge status={screenplay.status} />
          {screenplay.title && (
            <span className="text-sm text-gray-300 font-medium">{screenplay.title}</span>
          )}
          {saving && <span className="text-xs text-gray-500">Saving...</span>}
          {generatingFull && screenplay.generating_step && (
            <span className="text-xs text-blue-400 flex items-center gap-1">
              <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              Generating: {screenplay.generating_step}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Generate Full button */}
          <button
            onClick={handleGenerateFull}
            disabled={generatingFull || isLocked}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              generatingFull || isLocked
                ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                : "bg-purple-600 text-white hover:bg-purple-500"
            }`}
          >
            {generatingFull ? "Generating..." : "Generate Full Screenplay"}
          </button>

          {/* Status transition buttons */}
          {screenplay.status === "DRAFT" && (
            <button
              onClick={() => handleStatusChange("IN_REVIEW")}
              className="rounded-md bg-yellow-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-yellow-500"
            >
              Mark for Review
            </button>
          )}
          {screenplay.status === "IN_REVIEW" && (
            <>
              <button
                onClick={() => handleStatusChange("DRAFT")}
                className="rounded-md bg-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-600"
              >
                Back to Draft
              </button>
              <button
                onClick={() => handleStatusChange("LOCKED")}
                className="rounded-md bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-500"
              >
                Lock Screenplay
              </button>
            </>
          )}
          {screenplay.status === "LOCKED" && (
            <button
              onClick={() => handleStatusChange("DRAFT")}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-600"
            >
              Unlock
            </button>
          )}
        </div>
      </div>

      {/* Generate Scenes button (only when LOCKED + scene_breakdown exists) */}
      {isLocked && screenplay.scene_breakdown && screenplay.scene_breakdown.length > 0 && (
        <div className="flex justify-end">
          <button
            onClick={handleGenerateScenes}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
          >
            Generate Scenes from Screenplay
          </button>
        </div>
      )}

      {/* Tab navigation */}
      <div className="flex gap-1 overflow-x-auto">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors whitespace-nowrap ${
              activeTab === tab.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
        {activeTab === "logline" && (
          <LoglineTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("logline")}
            onChange={(field, value) => {
              setScreenplay((prev) => prev ? { ...prev, [field]: value } : prev);
              debouncedSave(field, value);
            }}
          />
        )}
        {activeTab === "treatment" && (
          <TreatmentTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("treatment")}
            onChange={(value) => {
              setScreenplay((prev) => prev ? { ...prev, treatment: value } : prev);
              debouncedSave("treatment", value);
            }}
          />
        )}
        {activeTab === "characters" && (
          <CharactersTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("characters")}
            onSave={(data) => saveField("character_breakdowns", data)}
          />
        )}
        {activeTab === "breakdown" && (
          <BreakdownTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("breakdown")}
            onSave={(data) => saveField("scene_breakdown", data)}
          />
        )}
        {activeTab === "script" && (
          <ScriptTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("script")}
            onChange={(value) => {
              setScreenplay((prev) => prev ? { ...prev, script: value } : prev);
              debouncedSave("script", value);
            }}
          />
        )}
        {activeTab === "shotlist" && (
          <ShotListTab
            screenplay={screenplay}
            isLocked={isLocked}
            regenerating={regenerating}
            onRegenerate={() => handleRegenerate("shotlist")}
            onSave={(data) => saveField("shot_list", data)}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Status Badge (local, simple)
// ============================================================================

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    DRAFT: { bg: "bg-gray-700", text: "text-gray-300", label: "Draft" },
    IN_REVIEW: { bg: "bg-yellow-900", text: "text-yellow-300", label: "In Review" },
    LOCKED: { bg: "bg-green-900", text: "text-green-300", label: "Locked" },
  };
  const c = config[status] ?? config.DRAFT;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${c.bg} ${c.text}`}>
      {status === "LOCKED" && (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      )}
      {c.label}
    </span>
  );
}

// ============================================================================
// Regenerate Button
// ============================================================================

function RegenerateButton({
  step,
  isLocked,
  regenerating,
  onRegenerate,
}: {
  step: string;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
}) {
  const isActive = regenerating === step;
  const disabled = isLocked || regenerating !== null;
  return (
    <button
      onClick={onRegenerate}
      disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
        disabled
          ? "bg-gray-700 text-gray-500 opacity-50 cursor-not-allowed"
          : "bg-blue-600 text-white hover:bg-blue-500"
      }`}
    >
      {isActive ? (
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
          Regenerating...
        </span>
      ) : (
        "Regenerate"
      )}
    </button>
  );
}

// ============================================================================
// Tab: Logline
// ============================================================================

function LoglineTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onChange,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onChange: (field: string, value: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Logline</h3>
        <RegenerateButton step="logline" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
      </div>
      <textarea
        value={screenplay.logline ?? ""}
        onChange={(e) => onChange("logline", e.target.value)}
        readOnly={isLocked}
        rows={3}
        placeholder="A one-sentence summary of the story..."
        className={`w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600 outline-none focus:border-blue-500 resize-none ${
          isLocked ? "opacity-70 cursor-not-allowed" : ""
        }`}
      />
      <div>
        <label className="block text-xs text-gray-500 mb-1">Genre</label>
        <input
          type="text"
          value={screenplay.genre ?? ""}
          onChange={(e) => onChange("genre", e.target.value)}
          readOnly={isLocked}
          placeholder="e.g. Sci-fi thriller, Romantic comedy"
          className={`w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600 outline-none focus:border-blue-500 ${
            isLocked ? "opacity-70 cursor-not-allowed" : ""
          }`}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Tab: Treatment
// ============================================================================

function TreatmentTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onChange,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Treatment</h3>
        <RegenerateButton step="treatment" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
      </div>
      <textarea
        value={screenplay.treatment ?? ""}
        onChange={(e) => onChange(e.target.value)}
        readOnly={isLocked}
        rows={10}
        placeholder="A prose narrative describing the story, characters, and visual approach..."
        className={`w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 placeholder:text-gray-600 outline-none focus:border-blue-500 resize-none ${
          isLocked ? "opacity-70 cursor-not-allowed" : ""
        }`}
      />
    </div>
  );
}

// ============================================================================
// Tab: Character Breakdowns
// ============================================================================

function CharactersTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onSave,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onSave: (data: CharacterBreakdownEntry[]) => void;
}) {
  const data = screenplay.character_breakdowns;
  const [editMode, setEditMode] = useState(false);
  const [jsonText, setJsonText] = useState("");

  function startEdit() {
    setJsonText(JSON.stringify(data ?? [], null, 2));
    setEditMode(true);
  }

  function handleSaveJson() {
    try {
      const parsed = JSON.parse(jsonText);
      onSave(parsed);
      setEditMode(false);
    } catch {
      // JSON parse error -- keep editing
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Character Breakdowns</h3>
        <div className="flex gap-2">
          {!isLocked && data && data.length > 0 && !editMode && (
            <button
              onClick={startEdit}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-600"
            >
              Edit JSON
            </button>
          )}
          <RegenerateButton step="character-breakdowns" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
        </div>
      </div>

      {editMode ? (
        <div className="space-y-2">
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            rows={16}
            className="w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs font-mono text-gray-200 outline-none focus:border-blue-500 resize-none"
          />
          <div className="flex gap-2">
            <button onClick={handleSaveJson} className="rounded-md bg-blue-600 px-3 py-1.5 text-xs text-white hover:bg-blue-500">
              Save
            </button>
            <button onClick={() => setEditMode(false)} className="rounded-md bg-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-600">
              Cancel
            </button>
          </div>
        </div>
      ) : data && data.length > 0 ? (
        <div className="space-y-3">
          {data.map((char: CharacterBreakdownEntry, idx: number) => (
            <div key={idx} className="rounded-lg border border-gray-700 bg-gray-900/50 p-3 space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">{char.name}</span>
                <span className="rounded-full bg-blue-900 px-2 py-0.5 text-xs text-blue-300">{char.role}</span>
                {char.manifest_tag && (
                  <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs text-gray-400">{char.manifest_tag}</span>
                )}
              </div>
              {char.arc && <p className="text-xs text-gray-400"><span className="text-gray-500">Arc:</span> {char.arc}</p>}
              {char.description && <p className="text-xs text-gray-400">{char.description}</p>}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500 py-4">No character breakdowns yet. Click Regenerate to generate them.</p>
      )}
    </div>
  );
}

// ============================================================================
// Tab: Scene Breakdown
// ============================================================================

function BreakdownTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onSave,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onSave: (data: SceneBreakdownEntry[]) => void;
}) {
  const data = screenplay.scene_breakdown;
  const [editMode, setEditMode] = useState(false);
  const [jsonText, setJsonText] = useState("");

  function startEdit() {
    setJsonText(JSON.stringify(data ?? [], null, 2));
    setEditMode(true);
  }

  function handleSaveJson() {
    try {
      const parsed = JSON.parse(jsonText);
      onSave(parsed);
      setEditMode(false);
    } catch {
      // JSON parse error
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Scene Breakdown</h3>
        <div className="flex gap-2">
          {!isLocked && data && data.length > 0 && !editMode && (
            <button
              onClick={startEdit}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-600"
            >
              Edit JSON
            </button>
          )}
          <RegenerateButton step="scene-breakdown" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
        </div>
      </div>

      {editMode ? (
        <div className="space-y-2">
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            rows={20}
            className="w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs font-mono text-gray-200 outline-none focus:border-blue-500 resize-none"
          />
          <div className="flex gap-2">
            <button onClick={handleSaveJson} className="rounded-md bg-blue-600 px-3 py-1.5 text-xs text-white hover:bg-blue-500">
              Save
            </button>
            <button onClick={() => setEditMode(false)} className="rounded-md bg-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-600">
              Cancel
            </button>
          </div>
        </div>
      ) : data && data.length > 0 ? (
        <div className="space-y-3">
          {data.map((entry: SceneBreakdownEntry, idx: number) => (
            <div key={idx} className="rounded-lg border border-gray-700 bg-gray-900/50 p-3 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-gray-500">#{entry.scene_number}</span>
                <span className="text-sm font-bold text-white">{entry.slugline}</span>
              </div>
              <p className="text-xs text-gray-300">{entry.intent}</p>
              {entry.emotional_beat && (
                <p className="text-xs text-gray-400">
                  <span className="text-gray-500">Emotional beat:</span> {entry.emotional_beat}
                </p>
              )}
              <div className="flex flex-wrap gap-2 text-xs">
                {entry.characters_present && entry.characters_present.length > 0 && (
                  <div className="flex items-center gap-1">
                    <span className="text-gray-500">Characters:</span>
                    {entry.characters_present.map((c, ci) => (
                      <span key={ci} className="rounded-full bg-blue-900/50 px-2 py-0.5 text-blue-300">{c}</span>
                    ))}
                  </div>
                )}
                {entry.set_ref && (
                  <span className="rounded-full bg-green-900/50 px-2 py-0.5 text-green-300">Set: {entry.set_ref}</span>
                )}
                {entry.props_required && entry.props_required.length > 0 && (
                  <div className="flex items-center gap-1">
                    <span className="text-gray-500">Props:</span>
                    {entry.props_required.map((p, pi) => (
                      <span key={pi} className="rounded-full bg-yellow-900/50 px-2 py-0.5 text-yellow-300">{p}</span>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex gap-4 text-xs text-gray-500">
                <span>In: {entry.story_state_in}</span>
                <span>Out: {entry.story_state_out}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500 py-4">No scene breakdown yet. Click Regenerate to generate one.</p>
      )}
    </div>
  );
}

// ============================================================================
// Tab: Script
// ============================================================================

function ScriptTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onChange,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Script</h3>
        <RegenerateButton step="script" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
      </div>
      <textarea
        value={screenplay.script ?? ""}
        onChange={(e) => onChange(e.target.value)}
        readOnly={isLocked}
        rows={20}
        placeholder="The full screenplay script..."
        className={`w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-sm font-mono text-gray-200 placeholder:text-gray-600 outline-none focus:border-blue-500 resize-none ${
          isLocked ? "opacity-70 cursor-not-allowed" : ""
        }`}
      />
    </div>
  );
}

// ============================================================================
// Tab: Shot List
// ============================================================================

function ShotListTab({
  screenplay,
  isLocked,
  regenerating,
  onRegenerate,
  onSave,
}: {
  screenplay: ScreenplayResponse;
  isLocked: boolean;
  regenerating: string | null;
  onRegenerate: () => void;
  onSave: (data: ShotListEntry[]) => void;
}) {
  const data = screenplay.shot_list;
  const [editMode, setEditMode] = useState(false);
  const [jsonText, setJsonText] = useState("");

  function startEdit() {
    setJsonText(JSON.stringify(data ?? [], null, 2));
    setEditMode(true);
  }

  function handleSaveJson() {
    try {
      const parsed = JSON.parse(jsonText);
      onSave(parsed);
      setEditMode(false);
    } catch {
      // JSON parse error
    }
  }

  // Group shots by scene_number
  const grouped: Record<number, ShotListEntry[]> = {};
  if (data) {
    for (const shot of data) {
      const sn = shot.scene_number;
      if (!grouped[sn]) grouped[sn] = [];
      grouped[sn].push(shot);
    }
  }
  const sceneNumbers = Object.keys(grouped).map(Number).sort((a, b) => a - b);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Shot List</h3>
        <div className="flex gap-2">
          {!isLocked && data && data.length > 0 && !editMode && (
            <button
              onClick={startEdit}
              className="rounded-md bg-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-600"
            >
              Edit JSON
            </button>
          )}
          <RegenerateButton step="shot-list" isLocked={isLocked} regenerating={regenerating} onRegenerate={onRegenerate} />
        </div>
      </div>

      {editMode ? (
        <div className="space-y-2">
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            rows={20}
            className="w-full rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs font-mono text-gray-200 outline-none focus:border-blue-500 resize-none"
          />
          <div className="flex gap-2">
            <button onClick={handleSaveJson} className="rounded-md bg-blue-600 px-3 py-1.5 text-xs text-white hover:bg-blue-500">
              Save
            </button>
            <button onClick={() => setEditMode(false)} className="rounded-md bg-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-600">
              Cancel
            </button>
          </div>
        </div>
      ) : data && data.length > 0 ? (
        <div className="space-y-4">
          {sceneNumbers.map((sceneNum) => (
            <div key={sceneNum}>
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Scene {sceneNum}</h4>
              <div className="space-y-2">
                {grouped[sceneNum].map((shot: ShotListEntry, idx: number) => (
                  <div key={idx} className="rounded-lg border border-gray-700 bg-gray-900/50 p-3 flex items-start gap-3">
                    <span className="text-xs font-mono text-gray-500 mt-0.5">
                      {shot.shot_number}
                    </span>
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="rounded bg-gray-700 px-2 py-0.5 text-xs font-medium text-gray-200">
                          {shot.shot_type}
                        </span>
                        {shot.camera_movement && (
                          <span className="text-xs text-gray-400">{shot.camera_movement}</span>
                        )}
                        {shot.duration_hint && (
                          <span className="text-xs text-gray-500">{shot.duration_hint}</span>
                        )}
                      </div>
                      <p className="text-xs text-gray-300">{shot.action_notes}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500 py-4">
          Generate Scene Breakdown first, then generate Shot List.
        </p>
      )}
    </div>
  );
}
