import { useState, useEffect } from "react";
import clsx from "clsx";
import { getSettings, updateSettings } from "../api/client.ts";
import { TEXT_MODELS, IMAGE_MODELS, VIDEO_MODELS } from "../lib/constants.ts";
import type { UserSettingsResponse } from "../api/types.ts";

export function SettingsPage() {
  const [settings, setSettings] = useState<UserSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  // Local form state
  const [enabledText, setEnabledText] = useState<string[]>([]);
  const [enabledImage, setEnabledImage] = useState<string[]>([]);
  const [enabledVideo, setEnabledVideo] = useState<string[]>([]);
  const [defaultText, setDefaultText] = useState<string | null>(null);
  const [defaultImage, setDefaultImage] = useState<string | null>(null);
  const [defaultVideo, setDefaultVideo] = useState<string | null>(null);
  const [gcpProject, setGcpProject] = useState("");
  const [gcpLocation, setGcpLocation] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [hasApiKey, setHasApiKey] = useState(false);

  // ComfyUI form state
  const [comfyuiHost, setComfyuiHost] = useState("");
  const [comfyuiApiKey, setComfyuiApiKey] = useState("");
  const [hasComfyuiKey, setHasComfyuiKey] = useState(false);
  const [comfyuiCostPerSecond, setComfyuiCostPerSecond] = useState("");

  useEffect(() => {
    getSettings()
      .then((s) => {
        setSettings(s);
        // null = all enabled
        setEnabledText(s.enabled_text_models ?? TEXT_MODELS.map((m) => m.id));
        setEnabledImage(s.enabled_image_models ?? IMAGE_MODELS.map((m) => m.id));
        setEnabledVideo(s.enabled_video_models ?? VIDEO_MODELS.map((m) => m.id));
        setDefaultText(s.default_text_model);
        setDefaultImage(s.default_image_model);
        setDefaultVideo(s.default_video_model);
        setGcpProject(s.gcp_project_id ?? "");
        setGcpLocation(s.gcp_location ?? "");
        setHasApiKey(s.has_api_key);
        setComfyuiHost(s.comfyui_host ?? "");
        setHasComfyuiKey(s.has_comfyui_key);
        setComfyuiCostPerSecond(
          s.comfyui_cost_per_second != null ? String(s.comfyui_cost_per_second) : ""
        );
      })
      .catch(() => setFeedback({ type: "error", msg: "Failed to load settings" }))
      .finally(() => setLoading(false));
  }, []);

  function toggleModel(list: string[], setList: (v: string[]) => void, id: string,
    defaultVal: string | null, setDefault: (v: string | null) => void) {
    if (list.includes(id)) {
      const next = list.filter((m) => m !== id);
      setList(next);
      if (defaultVal === id) setDefault(null);
    } else {
      setList([...list, id]);
    }
  }

  async function handleSave() {
    setSaving(true);
    setFeedback(null);
    try {
      const costVal = comfyuiCostPerSecond.trim();
      const res = await updateSettings({
        enabled_text_models: enabledText,
        enabled_image_models: enabledImage,
        enabled_video_models: enabledVideo,
        default_text_model: defaultText,
        default_image_model: defaultImage,
        default_video_model: defaultVideo,
        gcp_project_id: gcpProject || null,
        gcp_location: gcpLocation || null,
        vertex_api_key: apiKey || null,
        comfyui_host: comfyuiHost || null,
        comfyui_api_key: comfyuiApiKey || null,
        comfyui_cost_per_second: costVal ? parseFloat(costVal) : null,
      });
      setSettings(res);
      setHasApiKey(res.has_api_key);
      setApiKey("");
      setHasComfyuiKey(res.has_comfyui_key);
      setComfyuiApiKey("");
      setFeedback({ type: "success", msg: "Settings saved" });
    } catch (err) {
      setFeedback({ type: "error", msg: err instanceof Error ? err.message : "Save failed" });
    } finally {
      setSaving(false);
    }
  }

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
      setFeedback({ type: "error", msg: err instanceof Error ? err.message : "Failed to clear key" });
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
      setFeedback({ type: "error", msg: err instanceof Error ? err.message : "Failed to clear key" });
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-2xl py-12 text-center text-gray-400">
        Loading settings...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div>
        <h1 className="mb-1 text-2xl font-bold text-white">Settings</h1>
        <p className="text-sm text-gray-400">
          Configure enabled models, defaults, and API credentials.
        </p>
      </div>

      {/* Text Models */}
      <ModelSection
        title="Text Models"
        models={TEXT_MODELS.map((m) => ({ id: m.id, label: m.label }))}
        enabled={enabledText}
        onToggle={(id) => toggleModel(enabledText, setEnabledText, id, defaultText, setDefaultText)}
        defaultModel={defaultText}
        onDefaultChange={setDefaultText}
      />

      {/* Image Models */}
      <ModelSection
        title="Image Models"
        models={IMAGE_MODELS.map((m) => ({ id: m.id, label: m.label }))}
        enabled={enabledImage}
        onToggle={(id) => toggleModel(enabledImage, setEnabledImage, id, defaultImage, setDefaultImage)}
        defaultModel={defaultImage}
        onDefaultChange={setDefaultImage}
      />

      {/* Video Models */}
      <ModelSection
        title="Video Models"
        models={VIDEO_MODELS.map((m) => ({ id: m.id, label: m.label }))}
        enabled={enabledVideo}
        onToggle={(id) => toggleModel(enabledVideo, setEnabledVideo, id, defaultVideo, setDefaultVideo)}
        defaultModel={defaultVideo}
        onDefaultChange={setDefaultVideo}
      />

      {/* GCP Configuration */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">GCP Configuration</h2>
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
              className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
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
              className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>
      </section>

      {/* Vertex AI API Key */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-white">Vertex AI API Key</h2>
        <div>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={hasApiKey ? "API key is set (leave blank to keep)" : "Enter API key"}
            className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
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
      </section>

      {/* ComfyUI Configuration */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-white">ComfyUI Configuration</h2>
        <div>
          <label className="mb-1 block text-sm font-medium text-gray-300">
            Host URL
          </label>
          <input
            type="text"
            value={comfyuiHost}
            onChange={(e) => setComfyuiHost(e.target.value)}
            placeholder="https://api.comfy.org"
            className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
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
            placeholder={hasComfyuiKey ? "API key is set (leave blank to keep)" : "Enter ComfyUI API key"}
            className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
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
            type="number"
            step="0.01"
            min="0"
            value={comfyuiCostPerSecond}
            onChange={(e) => setComfyuiCostPerSecond(e.target.value)}
            placeholder="0.00"
            className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-100 placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <p className="mt-1 text-xs text-gray-500">
            Leave empty for $0.00. Used for cost estimation in the Generate form.
          </p>
        </div>
      </section>

      {/* Feedback */}
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

      {/* Save */}
      <button
        type="button"
        onClick={handleSave}
        disabled={saving}
        className={clsx(
          "w-full rounded-lg py-2.5 text-sm font-semibold transition-colors",
          saving
            ? "bg-gray-800 text-gray-500 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-500",
        )}
      >
        {saving ? "Saving..." : "Save Settings"}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model toggle section sub-component
// ---------------------------------------------------------------------------

interface ModelSectionProps {
  title: string;
  models: { id: string; label: string }[];
  enabled: string[];
  onToggle: (id: string) => void;
  defaultModel: string | null;
  onDefaultChange: (id: string | null) => void;
}

function ModelSection({ title, models, enabled, onToggle, defaultModel, onDefaultChange }: ModelSectionProps) {
  const enabledModels = models.filter((m) => enabled.includes(m.id));

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-white">{title}</h2>
      <div className="space-y-2">
        {models.map((m) => {
          const isEnabled = enabled.includes(m.id);
          return (
            <div key={m.id} className="flex items-center justify-between rounded-lg border border-gray-800 bg-gray-900/50 px-4 py-2.5">
              <span className={clsx("text-sm font-medium", isEnabled ? "text-gray-200" : "text-gray-500")}>
                {m.label}
                <span className="ml-2 text-xs text-gray-600">{m.id}</span>
              </span>
              <button
                type="button"
                onClick={() => onToggle(m.id)}
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
        })}
      </div>

      {/* Default model dropdown */}
      <div className="flex items-center gap-3">
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
    </section>
  );
}
