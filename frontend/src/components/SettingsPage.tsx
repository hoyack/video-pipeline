import { useState, useEffect } from "react";
import clsx from "clsx";
import { getSettings, updateSettings } from "../api/client.ts";
import {
  TEXT_MODELS,
  IMAGE_MODELS,
  VIDEO_MODELS,
  VERTEX_TEXT_MODELS,
  VERTEX_IMAGE_MODELS,
  VERTEX_VIDEO_MODELS,
  COMFYUI_VIDEO_MODELS_LIST,
  COMFYUI_IMAGE_MODELS,
} from "../lib/constants.ts";
import type { UserSettingsResponse, OllamaModelEntry } from "../api/types.ts";
import { useShowCost } from "../hooks/useShowCost.tsx";

// ---------------------------------------------------------------------------
// Collapsible provider card
// ---------------------------------------------------------------------------

function CollapsibleCard({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="rounded-lg border border-gray-700 bg-gray-800/50">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-5 py-3.5"
      >
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <svg
          className={clsx(
            "h-5 w-5 text-gray-400 transition-transform",
            open && "rotate-180",
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="space-y-5 px-5 pb-5">{children}</div>}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Sub-container inside a provider card (e.g. "Text Models", "API Settings")
// ---------------------------------------------------------------------------

function SettingsContainer({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-md border border-gray-700/60 bg-gray-900/30 p-4">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
        {title}
      </h3>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model toggle row with optional Vision badge
// ---------------------------------------------------------------------------

function ModelToggleRow({
  id,
  label,
  isEnabled,
  onToggle,
  showVision,
}: {
  id: string;
  label: string;
  isEnabled: boolean;
  onToggle: () => void;
  showVision?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-gray-800 bg-gray-900/50 px-4 py-2.5">
      <div className="flex items-center gap-2">
        <span
          className={clsx(
            "text-sm font-medium",
            isEnabled ? "text-gray-200" : "text-gray-500",
          )}
        >
          {label}
        </span>
        <span className="text-xs text-gray-600">{id}</span>
        {showVision && (
          <span className="rounded bg-purple-600/30 px-2 py-0.5 text-xs text-purple-300 border border-purple-600/50">
            Vision
          </span>
        )}
      </div>
      <button
        type="button"
        onClick={onToggle}
        className={clsx(
          "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
          isEnabled ? "bg-blue-600" : "bg-gray-700",
        )}
      >
        <span
          className={clsx(
            "inline-block h-4 w-4 rounded-full bg-white transition-transform",
            isEnabled ? "translate-x-6" : "translate-x-1",
          )}
        />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Default model dropdown
// ---------------------------------------------------------------------------

function DefaultModelSelect({
  allModels,
  enabled,
  defaultModel,
  onDefaultChange,
}: {
  allModels: { id: string; label: string }[];
  enabled: string[];
  defaultModel: string | null;
  onDefaultChange: (id: string | null) => void;
}) {
  const enabledModels = allModels.filter((m) => enabled.includes(m.id));
  return (
    <div className="flex items-center gap-3 pt-1">
      <label className="text-sm text-gray-400">Default:</label>
      <select
        value={defaultModel ?? ""}
        onChange={(e) => onDefaultChange(e.target.value || null)}
        className="rounded-md border border-gray-700 bg-gray-900 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
      >
        <option value="">None</option>
        {enabledModels.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Input field helper
// ---------------------------------------------------------------------------

const inputClass =
  "w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500";

// ===========================================================================
// Main component
// ===========================================================================

export function SettingsPage() {
  const { setShowCost: setShowCostCtx } = useShowCost();
  const [showCostLocal, setShowCostLocal] = useState(true);
  const [, setSettings] = useState<UserSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{
    type: "success" | "error";
    msg: string;
  } | null>(null);

  // Model state (flat lists — same data model as before)
  const [enabledText, setEnabledText] = useState<string[]>([]);
  const [enabledImage, setEnabledImage] = useState<string[]>([]);
  const [enabledVideo, setEnabledVideo] = useState<string[]>([]);
  const [defaultText, setDefaultText] = useState<string | null>(null);
  const [defaultImage, setDefaultImage] = useState<string | null>(null);
  const [defaultVideo, setDefaultVideo] = useState<string | null>(null);

  // Vertex Studio / GCP
  const [gcpProject, setGcpProject] = useState("");
  const [gcpLocation, setGcpLocation] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [hasApiKey, setHasApiKey] = useState(false);

  // ComfyUI
  const [comfyuiHost, setComfyuiHost] = useState("");
  const [comfyuiApiKey, setComfyuiApiKey] = useState("");
  const [hasComfyuiKey, setHasComfyuiKey] = useState(false);
  const [comfyuiCostPerSecond, setComfyuiCostPerSecond] = useState("");

  // Ollama
  const [ollamaUseCloud, setOllamaUseCloud] = useState(false);
  const [ollamaApiKey, setOllamaApiKey] = useState("");
  const [hasOllamaKey, setHasOllamaKey] = useState(false);
  const [ollamaEndpoint, setOllamaEndpoint] = useState("");
  const [ollamaModels, setOllamaModels] = useState<OllamaModelEntry[]>([]);
  const [newModelName, setNewModelName] = useState("");

  // GCP Service Account
  const [hasGcpServiceAccount, setHasGcpServiceAccount] = useState(false);
  const [serviceAccountJson, setServiceAccountJson] = useState("");
  const [serviceAccountFileName, setServiceAccountFileName] = useState("");

  // ElevenLabs
  const [elevenlabsApiKey, setElevenlabsApiKey] = useState("");
  const [hasElevenlabsKey, setHasElevenlabsKey] = useState(false);

  // ---- Load settings ----
  useEffect(() => {
    getSettings()
      .then((s) => {
        setSettings(s);
        setEnabledText(s.enabled_text_models ?? TEXT_MODELS.map((m) => m.id));
        setEnabledImage(s.enabled_image_models ?? IMAGE_MODELS.map((m) => m.id));
        setEnabledVideo(s.enabled_video_models ?? VIDEO_MODELS.map((m) => m.id));
        setDefaultText(s.default_text_model);
        setDefaultImage(s.default_image_model);
        setDefaultVideo(s.default_video_model);
        setGcpProject(s.gcp_project_id ?? "");
        setGcpLocation(s.gcp_location ?? "");
        setHasApiKey(s.has_api_key);
        setHasGcpServiceAccount(s.has_gcp_service_account);
        setComfyuiHost(s.comfyui_host ?? "");
        setHasComfyuiKey(s.has_comfyui_key);
        setComfyuiCostPerSecond(
          s.comfyui_cost_per_second != null
            ? String(s.comfyui_cost_per_second)
            : "",
        );
        setOllamaUseCloud(s.ollama_use_cloud);
        setHasOllamaKey(s.has_ollama_key);
        setOllamaEndpoint(s.ollama_endpoint ?? "");
        setOllamaModels(s.ollama_models ?? []);
        setShowCostLocal(s.show_cost);
        setHasElevenlabsKey(s.has_elevenlabs_key);
      })
      .catch(() =>
        setFeedback({ type: "error", msg: "Failed to load settings" }),
      )
      .finally(() => setLoading(false));
  }, []);

  // ---- Helpers ----
  function toggleModel(
    list: string[],
    setList: (v: string[]) => void,
    id: string,
    defaultVal: string | null,
    setDefault: (v: string | null) => void,
  ) {
    if (list.includes(id)) {
      const next = list.filter((m) => m !== id);
      setList(next);
      if (defaultVal === id) setDefault(null);
    } else {
      setList([...list, id]);
    }
  }

  // ---- Save ----
  async function handleSave() {
    setSaving(true);
    setFeedback(null);
    try {
      const costVal = comfyuiCostPerSecond.trim();
      const res = await updateSettings({
        show_cost: showCostLocal,
        enabled_text_models: enabledText,
        enabled_image_models: enabledImage,
        enabled_video_models: enabledVideo,
        default_text_model: defaultText,
        default_image_model: defaultImage,
        default_video_model: defaultVideo,
        gcp_project_id: gcpProject || null,
        gcp_location: gcpLocation || null,
        vertex_api_key: apiKey || null,
        ...(serviceAccountJson ? { gcp_service_account_json: serviceAccountJson } : {}),
        comfyui_host: comfyuiHost || null,
        comfyui_api_key: comfyuiApiKey || null,
        comfyui_cost_per_second: costVal ? parseFloat(costVal) : null,
        ollama_use_cloud: ollamaUseCloud,
        ollama_api_key: ollamaApiKey || null,
        ollama_endpoint: ollamaEndpoint || null,
        ollama_models: ollamaModels,
        elevenlabs_api_key: elevenlabsApiKey || null,
      });
      setSettings(res);
      setHasApiKey(res.has_api_key);
      setHasGcpServiceAccount(res.has_gcp_service_account);
      setServiceAccountJson("");
      setServiceAccountFileName("");
      setApiKey("");
      setHasComfyuiKey(res.has_comfyui_key);
      setComfyuiApiKey("");
      setHasOllamaKey(res.has_ollama_key);
      setOllamaApiKey("");
      setHasElevenlabsKey(res.has_elevenlabs_key);
      setElevenlabsApiKey("");
      setShowCostCtx(res.show_cost);
      setFeedback({ type: "success", msg: "Settings saved" });
    } catch (err) {
      setFeedback({
        type: "error",
        msg: err instanceof Error ? err.message : "Save failed",
      });
    } finally {
      setSaving(false);
    }
  }

  // ---- Clear key helpers ----
  async function handleClearKey() {
    setSaving(true);
    setFeedback(null);
    try {
      const res = await updateSettings({
        enabled_text_models: enabledText,
        enabled_image_models: enabledImage,
        enabled_video_models: enabledVideo,
        default_text_model: defaultText,
        default_image_model: defaultImage,
        default_video_model: defaultVideo,
        gcp_project_id: gcpProject || null,
        gcp_location: gcpLocation || null,
        clear_api_key: true,
      });
      setSettings(res);
      setHasApiKey(false);
      setApiKey("");
      setFeedback({ type: "success", msg: "API key cleared" });
    } catch (err) {
      setFeedback({
        type: "error",
        msg: err instanceof Error ? err.message : "Failed to clear key",
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleClearComfyuiKey() {
    setSaving(true);
    setFeedback(null);
    try {
      const res = await updateSettings({
        enabled_text_models: enabledText,
        enabled_image_models: enabledImage,
        enabled_video_models: enabledVideo,
        default_text_model: defaultText,
        default_image_model: defaultImage,
        default_video_model: defaultVideo,
        gcp_project_id: gcpProject || null,
        gcp_location: gcpLocation || null,
        clear_comfyui_key: true,
      });
      setSettings(res);
      setHasComfyuiKey(false);
      setComfyuiApiKey("");
      setFeedback({ type: "success", msg: "ComfyUI API key cleared" });
    } catch (err) {
      setFeedback({
        type: "error",
        msg: err instanceof Error ? err.message : "Failed to clear key",
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleClearElevenlabsKey() {
    setSaving(true);
    setFeedback(null);
    try {
      const res = await updateSettings({
        enabled_text_models: enabledText,
        enabled_image_models: enabledImage,
        enabled_video_models: enabledVideo,
        default_text_model: defaultText,
        default_image_model: defaultImage,
        default_video_model: defaultVideo,
        gcp_project_id: gcpProject || null,
        gcp_location: gcpLocation || null,
        clear_elevenlabs_key: true,
      });
      setSettings(res);
      setHasElevenlabsKey(false);
      setElevenlabsApiKey("");
      setFeedback({ type: "success", msg: "ElevenLabs API key cleared" });
    } catch (err) {
      setFeedback({
        type: "error",
        msg: err instanceof Error ? err.message : "Failed to clear key",
      });
    } finally {
      setSaving(false);
    }
  }

  // ---- Ollama model management ----
  function handleAddOllamaModel() {
    const name = newModelName.trim();
    if (!name) return;
    const id = `ollama/${name}`;
    if (ollamaModels.some((m) => m.id === id)) return;
    const label = name.split("/").pop() ?? name;
    const vision = /vision|llava/i.test(name);
    setOllamaModels([...ollamaModels, { id, label, enabled: true, vision }]);
    setNewModelName("");
  }

  function toggleOllamaModelEnabled(id: string) {
    setOllamaModels(
      ollamaModels.map((m) =>
        m.id === id ? { ...m, enabled: !m.enabled } : m,
      ),
    );
  }

  function toggleOllamaModelVision(id: string) {
    setOllamaModels(
      ollamaModels.map((m) =>
        m.id === id ? { ...m, vision: !m.vision } : m,
      ),
    );
  }

  function removeOllamaModel(id: string) {
    setOllamaModels(ollamaModels.filter((m) => m.id !== id));
  }

  // ---- Loading state ----
  if (loading) {
    return (
      <div className="mx-auto max-w-2xl py-12 text-center text-gray-400">
        Loading settings...
      </div>
    );
  }

  // ---- Render ----
  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="mb-1 text-2xl font-bold text-white">Settings</h1>
        <p className="text-sm text-gray-400">
          Configure enabled models, defaults, and API credentials.
        </p>
      </div>

      {/* ================================================================
          Display Preferences
          ================================================================ */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Display</h2>
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-medium text-gray-300">
              Show Cost Estimates
            </span>
            <p className="text-xs text-gray-500">
              Display estimated costs on scenes, forms, and dashboard
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowCostLocal((v) => !v)}
            className={clsx(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
              showCostLocal ? "bg-cyan-600" : "bg-gray-700",
            )}
          >
            <span
              className={clsx(
                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                showCostLocal ? "translate-x-6" : "translate-x-1",
              )}
            />
          </button>
        </div>
      </section>

      {/* ================================================================
          Vertex Studio
          ================================================================ */}
      <CollapsibleCard title="Vertex Studio">
        {/* --- API Settings --- */}
        <SettingsContainer title="API Settings">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-300">
                Project ID
              </label>
              <input
                type="text"
                value={gcpProject}
                onChange={(e) => setGcpProject(e.target.value)}
                placeholder="my-gcp-project"
                className={inputClass}
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-300">
                Location
              </label>
              <input
                type="text"
                value={gcpLocation}
                onChange={(e) => setGcpLocation(e.target.value)}
                placeholder="us-central1"
                className={inputClass}
              />
            </div>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                hasApiKey
                  ? "API key is set (leave blank to keep)"
                  : "Enter API key"
              }
              className={inputClass}
            />
            {hasApiKey && (
              <div className="mt-2 flex items-center gap-3">
                <span className="text-xs text-green-400">API key is set</span>
                <button
                  type="button"
                  onClick={handleClearKey}
                  disabled={saving}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Clear key
                </button>
              </div>
            )}
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              Service Account JSON
            </label>
            <p className="mb-2 text-xs text-gray-500">
              Upload your GCP service account key file (.json). Project ID and Location will be auto-populated if empty.
            </p>
            <div className="flex items-center gap-3">
              <label className="cursor-pointer rounded-lg border border-gray-700 bg-gray-900 px-4 py-2 text-sm text-gray-300 hover:border-blue-500 hover:text-white transition-colors">
                {serviceAccountFileName || "Choose file..."}
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = () => {
                      const text = reader.result as string;
                      try {
                        const parsed = JSON.parse(text);
                        if (!parsed.type || !parsed.project_id || !parsed.private_key) {
                          setFeedback({ type: "error", msg: "Invalid service account JSON: missing required fields" });
                          return;
                        }
                        setServiceAccountJson(text);
                        setServiceAccountFileName(file.name);
                        // Auto-populate project/location if empty
                        if (!gcpProject && parsed.project_id) {
                          setGcpProject(parsed.project_id);
                        }
                      } catch {
                        setFeedback({ type: "error", msg: "Invalid JSON file" });
                      }
                    };
                    reader.readAsText(file);
                    e.target.value = "";
                  }}
                />
              </label>
              {(hasGcpServiceAccount || serviceAccountJson) && (
                <span className="text-xs text-green-400">
                  {serviceAccountJson ? "Ready to upload" : "Service account is set"}
                </span>
              )}
            </div>
            {hasGcpServiceAccount && !serviceAccountJson && (
              <div className="mt-2">
                <button
                  type="button"
                  onClick={async () => {
                    setSaving(true);
                    setFeedback(null);
                    try {
                      const res = await updateSettings({
                        enabled_text_models: enabledText,
                        enabled_image_models: enabledImage,
                        enabled_video_models: enabledVideo,
                        default_text_model: defaultText,
                        default_image_model: defaultImage,
                        default_video_model: defaultVideo,
                        gcp_project_id: gcpProject || null,
                        gcp_location: gcpLocation || null,
                        clear_gcp_service_account: true,
                      });
                      setSettings(res);
                      setHasGcpServiceAccount(false);
                      setFeedback({ type: "success", msg: "Service account cleared" });
                    } catch (err) {
                      setFeedback({ type: "error", msg: err instanceof Error ? err.message : "Failed to clear" });
                    } finally {
                      setSaving(false);
                    }
                  }}
                  disabled={saving}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Clear service account
                </button>
              </div>
            )}
          </div>
        </SettingsContainer>

        {/* --- Text Models --- */}
        <SettingsContainer title="Text Models">
          <div className="space-y-2">
            {VERTEX_TEXT_MODELS.map((m) => (
              <ModelToggleRow
                key={m.id}
                id={m.id}
                label={m.label}
                isEnabled={enabledText.includes(m.id)}
                onToggle={() =>
                  toggleModel(
                    enabledText,
                    setEnabledText,
                    m.id,
                    defaultText,
                    setDefaultText,
                  )
                }
                showVision
              />
            ))}
          </div>
          <DefaultModelSelect
            allModels={TEXT_MODELS.map((m) => ({ id: m.id, label: m.label }))}
            enabled={enabledText}
            defaultModel={defaultText}
            onDefaultChange={setDefaultText}
          />
        </SettingsContainer>

        {/* --- Image Models --- */}
        <SettingsContainer title="Image Models">
          <div className="space-y-2">
            {VERTEX_IMAGE_MODELS.map((m) => (
              <ModelToggleRow
                key={m.id}
                id={m.id}
                label={m.label}
                isEnabled={enabledImage.includes(m.id)}
                onToggle={() =>
                  toggleModel(
                    enabledImage,
                    setEnabledImage,
                    m.id,
                    defaultImage,
                    setDefaultImage,
                  )
                }
              />
            ))}
          </div>
          <DefaultModelSelect
            allModels={IMAGE_MODELS.map((m) => ({ id: m.id, label: m.label }))}
            enabled={enabledImage}
            defaultModel={defaultImage}
            onDefaultChange={setDefaultImage}
          />
        </SettingsContainer>

        {/* --- Video Models --- */}
        <SettingsContainer title="Video Models">
          <div className="space-y-2">
            {VERTEX_VIDEO_MODELS.map((m) => (
              <ModelToggleRow
                key={m.id}
                id={m.id}
                label={m.label}
                isEnabled={enabledVideo.includes(m.id)}
                onToggle={() =>
                  toggleModel(
                    enabledVideo,
                    setEnabledVideo,
                    m.id,
                    defaultVideo,
                    setDefaultVideo,
                  )
                }
              />
            ))}
          </div>
          <DefaultModelSelect
            allModels={VIDEO_MODELS.map((m) => ({ id: m.id, label: m.label }))}
            enabled={enabledVideo}
            defaultModel={defaultVideo}
            onDefaultChange={setDefaultVideo}
          />
        </SettingsContainer>
      </CollapsibleCard>

      {/* ================================================================
          ComfyUI
          ================================================================ */}
      <CollapsibleCard title="ComfyUI">
        {/* --- API Settings --- */}
        <SettingsContainer title="API Settings">
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              Host URL
            </label>
            <input
              type="text"
              value={comfyuiHost}
              onChange={(e) => setComfyuiHost(e.target.value)}
              placeholder="https://api.comfy.org"
              className={inputClass}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              API Key
            </label>
            <input
              type="password"
              value={comfyuiApiKey}
              onChange={(e) => setComfyuiApiKey(e.target.value)}
              placeholder={
                hasComfyuiKey
                  ? "API key is set (leave blank to keep)"
                  : "Enter ComfyUI API key"
              }
              className={inputClass}
            />
            {hasComfyuiKey && (
              <div className="mt-2 flex items-center gap-3">
                <span className="text-xs text-green-400">API key is set</span>
                <button
                  type="button"
                  onClick={handleClearComfyuiKey}
                  disabled={saving}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Clear key
                </button>
              </div>
            )}
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              Cost per Second ($)
            </label>
            <input
              type={showCostLocal ? "number" : "password"}
              step="0.01"
              min="0"
              value={comfyuiCostPerSecond}
              onChange={(e) => setComfyuiCostPerSecond(e.target.value)}
              placeholder="0.00"
              className={inputClass}
            />
            <p className="mt-1 text-xs text-gray-500">
              Leave empty for $0.00. Used for cost estimation in the Generate
              form.
            </p>
          </div>
        </SettingsContainer>

        {/* --- Video Models --- */}
        <SettingsContainer title="Video Models">
          <div className="space-y-2">
            {COMFYUI_VIDEO_MODELS_LIST.map((m) => (
              <ModelToggleRow
                key={m.id}
                id={m.id}
                label={m.label}
                isEnabled={enabledVideo.includes(m.id)}
                onToggle={() =>
                  toggleModel(
                    enabledVideo,
                    setEnabledVideo,
                    m.id,
                    defaultVideo,
                    setDefaultVideo,
                  )
                }
              />
            ))}
          </div>
        </SettingsContainer>

        {/* --- Image Models --- */}
        <SettingsContainer title="Image Models">
          <div className="space-y-2">
            {COMFYUI_IMAGE_MODELS.map((m) => (
              <ModelToggleRow
                key={m.id}
                id={m.id}
                label={m.label}
                isEnabled={enabledImage.includes(m.id)}
                onToggle={() =>
                  toggleModel(
                    enabledImage,
                    setEnabledImage,
                    m.id,
                    defaultImage,
                    setDefaultImage,
                  )
                }
              />
            ))}
          </div>
        </SettingsContainer>
      </CollapsibleCard>

      {/* ================================================================
          Ollama
          ================================================================ */}
      <CollapsibleCard title="Ollama">
        {/* --- API Settings --- */}
        <SettingsContainer title="API Settings">
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-300">Mode:</span>
            <button
              onClick={() => setOllamaUseCloud(false)}
              className={clsx(
                "rounded px-3 py-1 text-sm",
                !ollamaUseCloud
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300",
              )}
            >
              Local
            </button>
            <button
              onClick={() => setOllamaUseCloud(true)}
              className={clsx(
                "rounded px-3 py-1 text-sm",
                ollamaUseCloud
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300",
              )}
            >
              Cloud
            </button>
          </div>

          {ollamaUseCloud && (
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-300">
                API Key
              </label>
              <input
                type="password"
                value={ollamaApiKey}
                onChange={(e) => setOllamaApiKey(e.target.value)}
                placeholder={
                  hasOllamaKey
                    ? "Key is set (leave blank to keep)"
                    : "Enter Ollama API key"
                }
                className={inputClass}
              />
              {hasOllamaKey && (
                <span className="mt-1 text-xs text-green-400">
                  API key is set
                </span>
              )}
            </div>
          )}

          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              Endpoint URL
            </label>
            <input
              type="text"
              value={ollamaEndpoint}
              onChange={(e) => setOllamaEndpoint(e.target.value)}
              placeholder={
                ollamaUseCloud
                  ? "https://ollama.com"
                  : "http://localhost:11434"
              }
              className={inputClass}
            />
            <p className="mt-1 text-xs text-gray-500">
              Leave empty for default. Local: http://localhost:11434, Cloud:
              https://ollama.com
            </p>
          </div>
        </SettingsContainer>

        {/* --- Models --- */}
        <SettingsContainer title="Models">
          <div className="mb-3 flex gap-2">
            <input
              type="text"
              value={newModelName}
              onChange={(e) => setNewModelName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddOllamaModel()}
              placeholder="e.g. llama3.2-vision"
              className={clsx(inputClass, "flex-1")}
            />
            <button
              type="button"
              onClick={handleAddOllamaModel}
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
            >
              Add
            </button>
          </div>

          <div className="space-y-2">
            {ollamaModels.map((m) => (
              <div
                key={m.id}
                className="flex items-center justify-between rounded-lg border border-gray-800 bg-gray-900/50 px-4 py-2.5"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={clsx(
                      "text-sm font-medium",
                      m.enabled ? "text-gray-200" : "text-gray-500",
                    )}
                  >
                    {m.label}
                  </span>
                  <span className="text-xs text-gray-600">{m.id}</span>
                  <button
                    type="button"
                    onClick={() => toggleOllamaModelVision(m.id)}
                    className={clsx(
                      "rounded px-2 py-0.5 text-xs",
                      m.vision
                        ? "border border-purple-600/50 bg-purple-600/30 text-purple-300"
                        : "bg-gray-800 text-gray-500",
                    )}
                  >
                    Vision
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => toggleOllamaModelEnabled(m.id)}
                    className={clsx(
                      "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
                      m.enabled ? "bg-blue-600" : "bg-gray-700",
                    )}
                  >
                    <span
                      className={clsx(
                        "inline-block h-3 w-3 rounded-full bg-white transition-transform",
                        m.enabled ? "translate-x-5" : "translate-x-1",
                      )}
                    />
                  </button>
                  <button
                    type="button"
                    onClick={() => removeOllamaModel(m.id)}
                    className="text-xs text-red-400 hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
            {ollamaModels.length === 0 && (
              <p className="py-2 text-xs text-gray-600">
                No Ollama models added. Type a model name above and click Add.
              </p>
            )}
          </div>
        </SettingsContainer>
      </CollapsibleCard>

      {/* ================================================================
          ElevenLabs
          ================================================================ */}
      <CollapsibleCard title="ElevenLabs" defaultOpen={false}>
        <SettingsContainer title="API Settings">
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-300">
              API Key
            </label>
            <input
              type="password"
              value={elevenlabsApiKey}
              onChange={(e) => setElevenlabsApiKey(e.target.value)}
              placeholder={
                hasElevenlabsKey
                  ? "API key is set (leave blank to keep)"
                  : "Enter ElevenLabs API key"
              }
              className={inputClass}
            />
            {hasElevenlabsKey && (
              <div className="mt-2 flex items-center gap-3">
                <span className="text-xs text-green-400">API key is set</span>
                <button
                  type="button"
                  onClick={handleClearElevenlabsKey}
                  disabled={saving}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  Clear key
                </button>
              </div>
            )}
          </div>
        </SettingsContainer>
      </CollapsibleCard>

      {/* ================================================================
          Feedback & Save
          ================================================================ */}
      {feedback && (
        <div
          className={clsx(
            "rounded-md border px-3 py-2 text-sm",
            feedback.type === "success"
              ? "border-green-800 bg-green-900/50 text-green-300"
              : "border-red-800 bg-red-900/50 text-red-300",
          )}
        >
          {feedback.msg}
        </div>
      )}

      <button
        type="button"
        onClick={handleSave}
        disabled={saving}
        className={clsx(
          "w-full rounded-lg py-2.5 text-sm font-semibold transition-colors",
          saving
            ? "cursor-not-allowed bg-gray-800 text-gray-500"
            : "bg-blue-600 text-white hover:bg-blue-500",
        )}
      >
        {saving ? "Saving..." : "Save Settings"}
      </button>
    </div>
  );
}
