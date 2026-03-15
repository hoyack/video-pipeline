import { useState, useEffect } from "react";
import type {
  ProductionBibleDetail,
  AssetResponse,
  UpdateAssetRequest,
  ProcessingProgress,
} from "../api/types.ts";
import {
  createProductionBible,
  getProductionBibleDetail,
  updateProductionBible,
  createAsset,
  updateAsset,
  deleteAsset,
  uploadAssetImage,
  processProductionBible,
  getProcessingProgress,
  reprocessAsset,
  uploadVideoForProductionBible,
  getExtractionProgress,
  importSceneToProductionBible,
} from "../api/client.ts";
import type {
  SetBinding,
  PropBinding,
  SoundBinding,
} from "../api/types.ts";
import { AssetUploader } from "./AssetUploader.tsx";
import { AssetEditor } from "./AssetEditor.tsx";
import { CharacterDetail } from "./CharacterDetail.tsx";
import { SetDetail } from "./SetDetail.tsx";
import { SoundDepartment } from "./SoundDepartment.tsx";
import { CastingSection } from "./CastingSection.tsx";
import { AssetPicker } from "./AssetPicker.tsx";
import { TagReferenceSheet } from "./TagReferenceSheet.tsx";
import {
  listCharacters,
  listSets,
  listProps,
  listScoreThemes,
  listSFXItems,
  listSetBindings,
  createSetBinding,
  deleteSetBinding,
  listPropBindings,
  createPropBinding,
  deletePropBinding,
  listSoundBindings,
  createSoundBinding,
  deleteSoundBinding,
} from "../api/client.ts";

interface ProductionBibleCreatorProps {
  productionBibleId?: string | null;
  onSaved: (productionBibleId: string) => void;
  onCancel: () => void;
}

const CATEGORIES = [
  "CUSTOM",
  "CHARACTERS",
  "ENVIRONMENTS",
  "OBJECTS",
  "PROPS",
  "STYLES",
];

const ASSET_TYPES = ["CHARACTER", "OBJECT", "VEHICLE", "ENVIRONMENT", "PROP", "STYLE", "OTHER"];

// Department tab configuration
const DEPARTMENT_TABS = [
  {
    id: "casting",
    label: "Casting",
    assetTypes: ["CHARACTER"],
  },
  {
    id: "art",
    label: "Art Department",
    assetTypes: ["ENVIRONMENT", "PROP", "OBJECT", "VEHICLE", "STYLE"],
  },
  {
    id: "sound",
    label: "Sound",
    assetTypes: [] as string[], // placeholder
  },
];

// Human-readable step labels
const STEP_LABELS: Record<string, string> = {
  contact_sheet: "Assembling contact sheet...",
  yolo_detection: "Detecting objects and faces...",
  face_matching: "Cross-matching faces...",
  reverse_prompting: "Generating asset descriptions...",
  finalizing: "Assigning tags and populating registry...",
  done: "Processing complete!",
};

// Extraction step labels
const EXTRACTION_STEP_LABELS: Record<string, string> = {
  initializing: "Initializing...",
  analyzing: "Analyzing video...",
  sampling: "Sampling frames...",
  deduplicating: "Deduplicating with CLIP...",
  saving: "Saving assets...",
  complete: "Extraction complete!",
};

export function ProductionBibleCreator({
  productionBibleId,
  onSaved,
  onCancel,
}: ProductionBibleCreatorProps) {
  const [manifest, setManifest] = useState<ProductionBibleDetail | null>(null);
  const [assets, setAssets] = useState<AssetResponse[]>([]);
  const [pendingFiles, setPendingFiles] = useState<Map<string, File>>(
    new Map()
  );
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("CUSTOM");
  const [tags, setTags] = useState("");
  const [saving, setSaving] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadingAssets, setUploadingAssets] = useState<Set<string>>(
    new Set()
  );
  const [progress, setProgress] = useState<ProcessingProgress | null>(null);
  const [reprocessingAssets, setReprocessingAssets] = useState<Set<string>>(
    new Set()
  );
  const [editingFields, setEditingFields] = useState<
    Record<string, Record<string, boolean>>
  >({});

  // Video extraction state
  const [extracting, setExtracting] = useState(false);
  const [extractionProgress, setExtractionProgress] =
    useState<ProcessingProgress | null>(null);

  // Scene import state
  const [sceneImportId, setSceneImportId] = useState("");
  const [importing, setImporting] = useState(false);

  // Department tab state (used in Stage 3)
  const [activeTab, setActiveTab] = useState("casting");

  // Entity counts for department tab badges (Stage 3)
  const [entityCounts, setEntityCounts] = useState<Record<string, number>>({
    casting: 0,
    art: 0,
    sound: 0,
  });
  const [showRawAssets, setShowRawAssets] = useState(false);

  // Binding state for Art Dept (sets, props) and Sound tabs
  const [setBindings, setSetBindings] = useState<SetBinding[]>([]);
  const [propBindings, setPropBindings] = useState<PropBinding[]>([]);
  const [soundBindings, setSoundBindings] = useState<SoundBinding[]>([]);
  const [setPickerOpen, setSetPickerOpen] = useState(false);
  const [propPickerOpen, setPropPickerOpen] = useState(false);
  const [soundPickerOpen, setSoundPickerOpen] = useState(false);

  // Binding create form state
  const [bindingFormType, setBindingFormType] = useState<"set" | "prop" | "sound" | null>(null);
  const [bindingFormAsset, setBindingFormAsset] = useState<{ id: string; name: string; primary_ref_url?: string | null } | null>(null);
  const [bindingFormName, setBindingFormName] = useState("");
  const [bindingFormTag, setBindingFormTag] = useState("");
  const [bindingFormNotes, setBindingFormNotes] = useState("");
  const [bindingFormSaving, setBindingFormSaving] = useState(false);

  // Collapsible legacy sections
  const [showLegacyCasting, setShowLegacyCasting] = useState(false);
  const [showLegacyArt, setShowLegacyArt] = useState(false);
  const [showLegacySound, setShowLegacySound] = useState(false);

  // Lightbox state for Stage 3 image preview
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);
  const [lightboxName, setLightboxName] = useState("");
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const CopyIcon = ({ size = 14 }: { size?: number }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" width={size} height={size}>
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );

  const SaveIcon = ({ size = 14 }: { size?: number }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" width={size} height={size}>
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  );

  const CheckIcon = ({ size = 14 }: { size?: number }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" width={size} height={size}>
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );

  const copyText = (text: string, fieldKey: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(fieldKey);
    setTimeout(() => setCopiedField(null), 1500);
  };

  const copyImage = async (imageUrl: string, fieldKey: string) => {
    try {
      const res = await fetch(imageUrl);
      const blob = await res.blob();
      // Convert to PNG for clipboard compatibility
      const pngBlob = blob.type === "image/png"
        ? blob
        : await new Promise<Blob>((resolve) => {
            const img = new Image();
            img.onload = () => {
              const canvas = document.createElement("canvas");
              canvas.width = img.naturalWidth;
              canvas.height = img.naturalHeight;
              canvas.getContext("2d")!.drawImage(img, 0, 0);
              canvas.toBlob((b) => resolve(b!), "image/png");
            };
            img.crossOrigin = "anonymous";
            img.src = URL.createObjectURL(blob);
          });
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": pngBlob }),
      ]);
      setCopiedField(fieldKey);
      setTimeout(() => setCopiedField(null), 1500);
    } catch {
      // Fallback: open in new tab if clipboard write fails
      window.open(imageUrl, "_blank");
    }
  };

  // Binding form helpers
  const generateBindingTag = (name: string): string =>
    name.toUpperCase().replace(/[^A-Z0-9\s]/g, "").trim().replace(/\s+/g, "_");

  const handleBindingPickerSelect = (
    type: "set" | "prop" | "sound",
    asset: { id: string; name: string; primary_ref_url?: string | null },
  ) => {
    setBindingFormType(type);
    setBindingFormAsset(asset);
    setBindingFormName(asset.name);
    setBindingFormTag(type === "sound" ? "" : generateBindingTag(asset.name));
    setBindingFormNotes("");
  };

  const handleSaveBinding = async () => {
    if (!bindingFormAsset || !bindingFormType) return;
    const bibleId = productionBibleId || manifest?.production_bible_id;
    if (!bibleId) return;

    setBindingFormSaving(true);
    try {
      if (bindingFormType === "set") {
        const created = await createSetBinding(bibleId, {
          library_set_id: bindingFormAsset.id,
          tag: bindingFormTag.trim() || generateBindingTag(bindingFormAsset.name),
          production_name: bindingFormName.trim() || undefined,
          lighting_override: bindingFormNotes.trim() || undefined,
        });
        setSetBindings((prev) => [...prev, created]);
      } else if (bindingFormType === "prop") {
        const created = await createPropBinding(bibleId, {
          library_prop_id: bindingFormAsset.id,
          tag: bindingFormTag.trim() || generateBindingTag(bindingFormAsset.name),
          production_name: bindingFormName.trim() || undefined,
          notes: bindingFormNotes.trim() || undefined,
        });
        setPropBindings((prev) => [...prev, created]);
      } else if (bindingFormType === "sound") {
        const created = await createSoundBinding(bibleId, {
          sound_asset_id: bindingFormAsset.id,
          tag: bindingFormTag.trim() || undefined,
          usage_notes: bindingFormNotes.trim() || undefined,
        });
        setSoundBindings((prev) => [...prev, created]);
      }
      setBindingFormType(null);
      setBindingFormAsset(null);
    } catch {
      // silently fail
    } finally {
      setBindingFormSaving(false);
    }
  };

  const handleDeleteSetBinding = async (id: string) => {
    try {
      await deleteSetBinding(id);
      setSetBindings((prev) => prev.filter((b) => b.id !== id));
    } catch { /* ignore */ }
  };

  const handleDeletePropBinding = async (id: string) => {
    try {
      await deletePropBinding(id);
      setPropBindings((prev) => prev.filter((b) => b.id !== id));
    } catch { /* ignore */ }
  };

  const handleDeleteSoundBinding = async (id: string) => {
    try {
      await deleteSoundBinding(id);
      setSoundBindings((prev) => prev.filter((b) => b.id !== id));
    } catch { /* ignore */ }
  };

  const isNewBible = !productionBibleId;

  // Determine current stage based on manifest status
  const currentStage = (() => {
    if (extracting) return 1; // Stay on stage 1 but show extraction UI
    if (!manifest || manifest.status === "DRAFT") return 1;
    if (manifest.status === "EXTRACTING") return 1;
    if (manifest.status === "PROCESSING") return 2;
    if (manifest.status === "READY" || manifest.status === "ERROR") return 3;
    return 1;
  })();

  // Load existing production bible in edit mode
  useEffect(() => {
    if (productionBibleId) {
      getProductionBibleDetail(productionBibleId)
        .then((data) => {
          setManifest(data);
          setName(data.name);
          setDescription(data.description || "");
          setCategory(data.category);
          setTags(data.tags ? data.tags.join(", ") : "");
          setAssets(data.assets);

          // If bible is already PROCESSING, start polling
          if (data.status === "PROCESSING") {
            setProcessing(true);
          }
          // If bible is EXTRACTING, start extraction polling
          if (data.status === "EXTRACTING") {
            setExtracting(true);
          }
        })
        .catch((err) => {
          setError(`Failed to load production bible: ${err.message}`);
        });
    }
  }, [productionBibleId]);

  // Fetch entity counts for Stage 3 tab badges
  useEffect(() => {
    const bibleId = productionBibleId || manifest?.production_bible_id;
    if (!bibleId || currentStage !== 3) return;

    const fetchCounts = async () => {
      try {
        const [chars, sets, props, themes, sfx] = await Promise.all([
          listCharacters(bibleId),
          listSets(bibleId),
          listProps(bibleId),
          listScoreThemes(bibleId),
          listSFXItems(bibleId),
        ]);
        setEntityCounts({
          casting: chars.length,
          art: sets.length + props.length,
          sound: themes.length + sfx.length,
        });
      } catch {
        // Counts are cosmetic; silently ignore errors
      }
    };
    fetchCounts();
  }, [productionBibleId, manifest?.production_bible_id, currentStage]);

  // Fetch bindings for Art Dept and Sound tabs
  useEffect(() => {
    const bibleId = productionBibleId || manifest?.production_bible_id;
    if (!bibleId || currentStage !== 3) return;

    const fetchBindings = async () => {
      try {
        const [sets, props, sounds] = await Promise.all([
          listSetBindings(bibleId),
          listPropBindings(bibleId),
          listSoundBindings(bibleId),
        ]);
        setSetBindings(sets);
        setPropBindings(props);
        setSoundBindings(sounds);
      } catch {
        // silently ignore
      }
    };
    fetchBindings();
  }, [productionBibleId, manifest?.production_bible_id, currentStage]);

  // Polling for Stage 2 (PROCESSING)
  useEffect(() => {
    if (!processing || !manifest?.production_bible_id) return;

    const pollInterval = setInterval(async () => {
      try {
        const progressData = await getProcessingProgress(manifest.production_bible_id);
        setProgress(progressData);

        if (progressData.status === "complete") {
          clearInterval(pollInterval);
          setProcessing(false);
          // Reload bible to get updated status and assets
          const updated = await getProductionBibleDetail(manifest.production_bible_id);
          setManifest(updated);
          setAssets(updated.assets);
        } else if (progressData.status === "error") {
          clearInterval(pollInterval);
          setProcessing(false);
          setError(progressData.error || "Processing failed");
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        console.error("Failed to fetch progress:", errorMessage);
        // Don't clear interval on network errors - keep polling
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [processing, manifest?.production_bible_id]);

  // Polling for video frame extraction
  useEffect(() => {
    if (!extracting || !manifest?.production_bible_id) return;

    const pollInterval = setInterval(async () => {
      try {
        const progressData = await getExtractionProgress(manifest.production_bible_id);
        setExtractionProgress(progressData);

        if (progressData.status === "complete") {
          clearInterval(pollInterval);
          setExtracting(false);
          setExtractionProgress(null);
          // Reload bible to get extracted frame assets
          const updated = await getProductionBibleDetail(manifest.production_bible_id);
          setManifest(updated);
          setAssets(updated.assets);
        } else if (progressData.status === "error") {
          clearInterval(pollInterval);
          setExtracting(false);
          setExtractionProgress(null);
          setError(progressData.error || "Video extraction failed");
          // Reload bible to get current state
          const updated = await getProductionBibleDetail(manifest.production_bible_id);
          setManifest(updated);
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        console.error("Failed to fetch extraction progress:", errorMessage);
      }
    }, 1500);

    return () => clearInterval(pollInterval);
  }, [extracting, manifest?.production_bible_id]);

  // Lazy-create production bible helper (shared between image and video upload)
  const ensureBibleExists = async (): Promise<string | null> => {
    const currentId = productionBibleId || manifest?.production_bible_id;
    if (currentId) return currentId;

    try {
      const newBible = await createProductionBible({
        name: name || "Untitled Production Bible",
        description: description || undefined,
        category,
        tags: tags
          ? tags
              .split(",")
              .map((t) => t.trim())
              .filter(Boolean)
          : undefined,
      });
      setManifest({
        ...newBible,
        assets: [],
        processing_progress: null,
        contact_sheet_url: null,
        total_processing_cost: 0,
        parent_production_bible_id: null,
        source_video_duration: null,
      });
      return newBible.production_bible_id;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to create production bible: ${errorMessage}`);
      return null;
    }
  };

  const handleFilesSelected = async (files: File[]) => {
    if (files.length === 0) return;

    const currentId = await ensureBibleExists();
    if (!currentId) return;

    // Process files sequentially to avoid overwhelming API
    for (const file of files) {
      try {
        // Create asset
        const asset = await createAsset(currentId, {
          name: file.name.replace(/\.[^/.]+$/, ""), // Remove extension
          asset_type: "CHARACTER", // Default type
        });

        // Store file for preview
        setPendingFiles((prev) => new Map(prev).set(asset.asset_id, file));

        // Mark as uploading
        setUploadingAssets((prev) => new Set(prev).add(asset.asset_id));

        // Upload image
        const updatedAsset = await uploadAssetImage(asset.asset_id, file);

        // Add to assets list
        setAssets((prev) => [...prev, updatedAsset]);

        // Clear uploading state
        setUploadingAssets((prev) => {
          const next = new Set(prev);
          next.delete(asset.asset_id);
          return next;
        });
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        setError(`Failed to upload ${file.name}: ${errorMessage}`);
      }
    }
  };

  const handleVideoSelected = async (file: File) => {
    const currentId = await ensureBibleExists();
    if (!currentId) return;

    setError(null);
    setExtracting(true);

    try {
      await uploadVideoForProductionBible(currentId, file);
      // Update local manifest status
      setManifest((prev) =>
        prev ? { ...prev, status: "EXTRACTING" } : null
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to upload video: ${errorMessage}`);
      setExtracting(false);
    }
  };

  const handleSceneImport = async () => {
    const trimmedId = sceneImportId.trim();
    if (!trimmedId) return;

    setError(null);
    setImporting(true);

    try {
      const result = await importSceneToProductionBible(
        trimmedId,
        name || undefined,
      );
      setManifest(result);
      setName(result.name);
      setDescription(result.description || "");
      setCategory(result.category);
      setTags(result.tags ? result.tags.join(", ") : "");
      setAssets(result.assets);
      setSceneImportId("");
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to import scene: ${errorMessage}`);
    } finally {
      setImporting(false);
    }
  };

  const handleAssetUpdate = async (
    assetId: string,
    updates: UpdateAssetRequest
  ) => {
    try {
      const updatedAsset = await updateAsset(assetId, updates);
      setAssets((prev) =>
        prev.map((a) => (a.asset_id === assetId ? updatedAsset : a))
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to update asset: ${errorMessage}`);
    }
  };

  const handleAssetDelete = async (assetId: string) => {
    if (!confirm("Remove this asset from the production bible?")) return;

    try {
      await deleteAsset(assetId);
      setAssets((prev) => prev.filter((a) => a.asset_id !== assetId));
      setPendingFiles((prev) => {
        const next = new Map(prev);
        next.delete(assetId);
        return next;
      });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to delete asset: ${errorMessage}`);
    }
  };

  const handleProcess = async () => {
    if (!manifest?.production_bible_id) return;

    setProcessing(true);
    setError(null);

    try {
      await processProductionBible(manifest.production_bible_id);
      // Update local status to transition to Stage 2
      setManifest((prev) => (prev ? { ...prev, status: "PROCESSING" } : null));
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to start processing: ${errorMessage}`);
      setProcessing(false);
    }
  };

  const handleReprocess = async (assetId: string) => {
    setReprocessingAssets((prev) => new Set(prev).add(assetId));
    setError(null);

    try {
      const updatedAsset = await reprocessAsset(assetId);
      setAssets((prev) =>
        prev.map((a) => (a.asset_id === assetId ? updatedAsset : a))
      );
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to reprocess asset: ${errorMessage}`);
    } finally {
      setReprocessingAssets((prev) => {
        const next = new Set(prev);
        next.delete(assetId);
        return next;
      });
    }
  };

  const handleReprocessAll = async () => {
    if (!manifest?.production_bible_id) return;
    if (
      !confirm(
        "Re-run full processing pipeline? This will regenerate all asset descriptions."
      )
    )
      return;

    setProcessing(true);
    setError(null);

    try {
      await processProductionBible(manifest.production_bible_id);
      setManifest((prev) => (prev ? { ...prev, status: "PROCESSING" } : null));
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to start processing: ${errorMessage}`);
      setProcessing(false);
    }
  };

  const handleSave = async () => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const currentId = productionBibleId || manifest?.production_bible_id;

      if (currentId) {
        // Update existing production bible
        await updateProductionBible(currentId, {
          name,
          description: description || undefined,
          category,
          tags: tags
            ? tags
                .split(",")
                .map((t) => t.trim())
                .filter(Boolean)
            : undefined,
        });
        onSaved(currentId);
      } else {
        // Create new production bible (no assets uploaded yet)
        const newBible = await createProductionBible({
          name,
          description: description || undefined,
          category,
          tags: tags
            ? tags
                .split(",")
                .map((t) => t.trim())
                .filter(Boolean)
            : undefined,
        });
        onSaved(newBible.production_bible_id);
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Failed to save production bible: ${errorMessage}`);
    } finally {
      setSaving(false);
    }
  };

  const toggleFieldEdit = (assetId: string, field: string) => {
    setEditingFields((prev) => ({
      ...prev,
      [assetId]: {
        ...prev[assetId],
        [field]: !prev[assetId]?.[field],
      },
    }));
  };

  const isFieldEditing = (assetId: string, field: string) => {
    return editingFields[assetId]?.[field] || false;
  };

  const getQualityBadgeColor = (score: number | null) => {
    if (score === null) return "bg-gray-700 text-gray-400";
    if (score >= 7) return "bg-green-700 text-green-300";
    if (score >= 4) return "bg-yellow-700 text-yellow-300";
    return "bg-red-700 text-red-300";
  };

  // Extraction progress UI (shown during video frame extraction)
  const renderExtractionProgress = () => {
    const stepLabel =
      EXTRACTION_STEP_LABELS[extractionProgress?.current_step || ""] ||
      "Extracting frames...";
    const candidateFrames =
      (extractionProgress?.progress as Record<string, number> | null)
        ?.candidate_frames || 0;
    const uniqueFrames =
      (extractionProgress?.progress as Record<string, number> | null)
        ?.unique_frames || 0;

    return (
      <div className="mb-8 rounded-lg border border-purple-800 bg-purple-900/20 p-8">
        <div className="flex flex-col items-center gap-4">
          {/* Spinner */}
          <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>

          {/* Step label */}
          <p className="text-lg text-purple-300">{stepLabel}</p>

          {/* Frame counts */}
          <div className="flex gap-6 text-sm">
            {candidateFrames > 0 && (
              <div>
                <span className="text-gray-400">Sampled:</span>{" "}
                <span className="text-white font-medium">
                  {candidateFrames} frames
                </span>
              </div>
            )}
            {uniqueFrames > 0 && (
              <div>
                <span className="text-gray-400">Unique:</span>{" "}
                <span className="text-purple-300 font-medium">
                  {uniqueFrames} frames
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Video info banner (shown in Stage 1 after extraction)
  const renderVideoInfoBanner = () => {
    if (!manifest?.source_video_duration) return null;

    const videoFrameAssets = assets.filter((a) => a.source === "video_frame");
    if (videoFrameAssets.length === 0) return null;

    return (
      <div className="mb-4 rounded-lg border border-purple-800 bg-purple-900/20 px-4 py-3">
        <p className="text-sm text-purple-300">
          Video uploaded ({manifest.source_video_duration.toFixed(1)}s) —{" "}
          {videoFrameAssets.length} frames extracted
        </p>
      </div>
    );
  };

  // Shared department tabs + binding UI — used in both Stage 1 (edit mode) and Stage 3
  const renderDepartmentTabs = () => {
    if (!manifest?.production_bible_id) return null;

    return (
      <>
        {/* Department tabs */}
        <div className="mb-4 flex gap-2">
          {DEPARTMENT_TABS.map((tab) => {
            const count = entityCounts[tab.id] || 0;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "bg-blue-600 text-white"
                    : "bg-gray-800 text-gray-400 hover:text-gray-200"
                }`}
              >
                {tab.label}
                {count > 0 && (
                  <span className="ml-1.5 rounded-full bg-gray-700 px-1.5 py-0.5 text-xs">
                    {count}
                  </span>
                )}
              </button>
            );
          })}
          <button
            key="tag-reference"
            onClick={() => setActiveTab("tag-reference")}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              activeTab === "tag-reference"
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Tag Reference
          </button>
        </div>

        {/* Casting tab */}
        {activeTab === "casting" && (
          <div className="mb-8 space-y-6">
            <CastingSection bibleId={manifest.production_bible_id} />
            <div>
              <button
                onClick={() => setShowLegacyCasting(!showLegacyCasting)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 mb-2"
              >
                <span className={`transition-transform ${showLegacyCasting ? "rotate-90" : ""}`}>&#9654;</span>
                Legacy Characters
              </button>
              {showLegacyCasting && (
                <CharacterDetail productionBibleId={manifest.production_bible_id} />
              )}
            </div>
          </div>
        )}

        {/* Art Department tab */}
        {activeTab === "art" && (
          <div className="mb-8 space-y-6">
            {/* Bound Sets */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Sets</h3>
                <button
                  onClick={() => setSetPickerOpen(true)}
                  className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
                >
                  + Add Set
                </button>
              </div>
              {bindingFormType === "set" && bindingFormAsset && (
                <div className="mb-3 rounded-lg border border-blue-800 bg-gray-900/80 p-4">
                  <h4 className="text-sm font-medium text-white mb-3">
                    Bind Set: <span className="text-blue-400">{bindingFormAsset.name}</span>
                  </h4>
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Production Name</label>
                      <input type="text" value={bindingFormName} onChange={(e) => setBindingFormName(e.target.value)}
                        className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500" />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Tag</label>
                      <input type="text" value={bindingFormTag} onChange={(e) => setBindingFormTag(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                        className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-blue-500" />
                    </div>
                  </div>
                  <div className="mb-3">
                    <label className="block text-xs text-gray-400 mb-1">Lighting Override (optional)</label>
                    <textarea value={bindingFormNotes} onChange={(e) => setBindingFormNotes(e.target.value)} rows={2}
                      className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none" />
                  </div>
                  <div className="flex gap-2">
                    <button onClick={handleSaveBinding} disabled={bindingFormSaving}
                      className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors disabled:opacity-50">
                      {bindingFormSaving ? "Saving..." : "Add Set"}
                    </button>
                    <button onClick={() => { setBindingFormType(null); setBindingFormAsset(null); }}
                      className="rounded bg-gray-700 px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors">
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {setBindings.length === 0 && bindingFormType !== "set" ? (
                <div className="flex min-h-[80px] items-center justify-center rounded-lg border border-dashed border-gray-700">
                  <p className="text-sm text-gray-500">No bound sets yet.</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {setBindings.map((b) => (
                    <div key={b.id} className="flex items-center gap-3 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
                      {b.primary_ref_url ? (
                        <img src={b.primary_ref_url} alt={b.production_name || b.set_name || ""} className="w-12 h-12 rounded object-cover border border-gray-700 shrink-0" />
                      ) : (
                        <div className="w-12 h-12 rounded bg-gray-800 border border-gray-700 shrink-0 flex items-center justify-center text-gray-600 text-xs">No ref</div>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-white">{b.production_name || b.set_name}</span>
                          <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs font-mono text-gray-300">{b.tag}</span>
                        </div>
                        {b.lighting_override && <div className="text-xs text-gray-500 mt-0.5 truncate">{b.lighting_override}</div>}
                      </div>
                      <button onClick={() => handleDeleteSetBinding(b.id)} className="rounded px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-gray-800 transition-colors shrink-0">Remove</button>
                    </div>
                  ))}
                </div>
              )}
              <AssetPicker isOpen={setPickerOpen} onClose={() => setSetPickerOpen(false)} assetType="set"
                onSelect={(a) => handleBindingPickerSelect("set", a)} excludeIds={setBindings.map((b) => b.library_set_id)} />
            </div>

            {/* Bound Props */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Props</h3>
                <button
                  onClick={() => setPropPickerOpen(true)}
                  className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
                >
                  + Add Prop
                </button>
              </div>
              {bindingFormType === "prop" && bindingFormAsset && (
                <div className="mb-3 rounded-lg border border-blue-800 bg-gray-900/80 p-4">
                  <h4 className="text-sm font-medium text-white mb-3">
                    Bind Prop: <span className="text-blue-400">{bindingFormAsset.name}</span>
                  </h4>
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Production Name</label>
                      <input type="text" value={bindingFormName} onChange={(e) => setBindingFormName(e.target.value)}
                        className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500" />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Tag</label>
                      <input type="text" value={bindingFormTag} onChange={(e) => setBindingFormTag(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                        className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-blue-500" />
                    </div>
                  </div>
                  <div className="mb-3">
                    <label className="block text-xs text-gray-400 mb-1">Notes (optional)</label>
                    <textarea value={bindingFormNotes} onChange={(e) => setBindingFormNotes(e.target.value)} rows={2}
                      className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none" />
                  </div>
                  <div className="flex gap-2">
                    <button onClick={handleSaveBinding} disabled={bindingFormSaving}
                      className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors disabled:opacity-50">
                      {bindingFormSaving ? "Saving..." : "Add Prop"}
                    </button>
                    <button onClick={() => { setBindingFormType(null); setBindingFormAsset(null); }}
                      className="rounded bg-gray-700 px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors">
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {propBindings.length === 0 && bindingFormType !== "prop" ? (
                <div className="flex min-h-[80px] items-center justify-center rounded-lg border border-dashed border-gray-700">
                  <p className="text-sm text-gray-500">No bound props yet.</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {propBindings.map((b) => (
                    <div key={b.id} className="flex items-center gap-3 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
                      {b.primary_ref_url ? (
                        <img src={b.primary_ref_url} alt={b.production_name || b.prop_name || ""} className="w-12 h-12 rounded object-cover border border-gray-700 shrink-0" />
                      ) : (
                        <div className="w-12 h-12 rounded bg-gray-800 border border-gray-700 shrink-0 flex items-center justify-center text-gray-600 text-xs">No ref</div>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-white">{b.production_name || b.prop_name}</span>
                          <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs font-mono text-gray-300">{b.tag}</span>
                        </div>
                        {b.notes && <div className="text-xs text-gray-500 mt-0.5 truncate">{b.notes}</div>}
                      </div>
                      <button onClick={() => handleDeletePropBinding(b.id)} className="rounded px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-gray-800 transition-colors shrink-0">Remove</button>
                    </div>
                  ))}
                </div>
              )}
              <AssetPicker isOpen={propPickerOpen} onClose={() => setPropPickerOpen(false)} assetType="prop"
                onSelect={(a) => handleBindingPickerSelect("prop", a)} excludeIds={propBindings.map((b) => b.library_prop_id)} />
            </div>

            {/* Legacy art dept entities */}
            <div>
              <button
                onClick={() => setShowLegacyArt(!showLegacyArt)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 mb-2"
              >
                <span className={`transition-transform ${showLegacyArt ? "rotate-90" : ""}`}>&#9654;</span>
                Legacy Sets & Props
              </button>
              {showLegacyArt && (
                <SetDetail productionBibleId={manifest.production_bible_id} />
              )}
            </div>
          </div>
        )}

        {/* Sound tab */}
        {activeTab === "sound" && (
          <div className="mb-8 space-y-6">
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Sound Assets</h3>
                <button
                  onClick={() => setSoundPickerOpen(true)}
                  className="rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors"
                >
                  + Add Sound
                </button>
              </div>
              {bindingFormType === "sound" && bindingFormAsset && (
                <div className="mb-3 rounded-lg border border-blue-800 bg-gray-900/80 p-4">
                  <h4 className="text-sm font-medium text-white mb-3">
                    Bind Sound: <span className="text-blue-400">{bindingFormAsset.name}</span>
                  </h4>
                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Tag (optional)</label>
                      <input type="text" value={bindingFormTag} onChange={(e) => setBindingFormTag(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ""))}
                        className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white font-mono outline-none focus:border-blue-500" />
                    </div>
                  </div>
                  <div className="mb-3">
                    <label className="block text-xs text-gray-400 mb-1">Usage Notes (optional)</label>
                    <textarea value={bindingFormNotes} onChange={(e) => setBindingFormNotes(e.target.value)} rows={2}
                      className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white outline-none focus:border-blue-500 resize-none" />
                  </div>
                  <div className="flex gap-2">
                    <button onClick={handleSaveBinding} disabled={bindingFormSaving}
                      className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-500 transition-colors disabled:opacity-50">
                      {bindingFormSaving ? "Saving..." : "Add Sound"}
                    </button>
                    <button onClick={() => { setBindingFormType(null); setBindingFormAsset(null); }}
                      className="rounded bg-gray-700 px-4 py-1.5 text-sm text-gray-300 hover:text-white transition-colors">
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {soundBindings.length === 0 && bindingFormType !== "sound" ? (
                <div className="flex min-h-[80px] items-center justify-center rounded-lg border border-dashed border-gray-700">
                  <p className="text-sm text-gray-500">No bound sound assets yet.</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {soundBindings.map((b) => (
                    <div key={b.id} className="flex items-center gap-3 rounded-lg border border-gray-800 bg-gray-900/50 p-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-white">{b.sound_name}</span>
                          {b.sound_category && (
                            <span className="rounded-full bg-purple-900/50 px-2 py-0.5 text-xs text-purple-300">{b.sound_category}</span>
                          )}
                          {b.tag && (
                            <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs font-mono text-gray-300">{b.tag}</span>
                          )}
                        </div>
                        {b.usage_notes && <div className="text-xs text-gray-500 mt-0.5 truncate">{b.usage_notes}</div>}
                      </div>
                      <button onClick={() => handleDeleteSoundBinding(b.id)} className="rounded px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-gray-800 transition-colors shrink-0">Remove</button>
                    </div>
                  ))}
                </div>
              )}
              <AssetPicker isOpen={soundPickerOpen} onClose={() => setSoundPickerOpen(false)} assetType="sound"
                onSelect={(a) => handleBindingPickerSelect("sound", a)} excludeIds={soundBindings.map((b) => b.sound_asset_id)} />
            </div>
            <div>
              <button
                onClick={() => setShowLegacySound(!showLegacySound)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 mb-2"
              >
                <span className={`transition-transform ${showLegacySound ? "rotate-90" : ""}`}>&#9654;</span>
                Legacy Score Themes & SFX
              </button>
              {showLegacySound && (
                <SoundDepartment productionBibleId={manifest.production_bible_id} />
              )}
            </div>
          </div>
        )}

        {/* Tag Reference tab */}
        {activeTab === "tag-reference" && (
          <TagReferenceSheet bibleId={manifest.production_bible_id} />
        )}
      </>
    );
  };

  // STAGE 1: Upload + tag UI (DRAFT)
  const renderStage1 = () => (
    <>
      {/* Header form */}
      <div className="mb-8 rounded-lg border border-gray-800 bg-gray-900/50 p-6">
        <h2 className="text-xl font-bold text-white mb-4">
          {isNewBible ? "New Production Bible" : "Edit Production Bible"}
        </h2>

        {/* Name */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
            placeholder="My Production Bible"
          />
        </div>

        {/* Description */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none resize-none"
            placeholder="Describe your production bible..."
          />
        </div>

        {/* Category + Tags */}
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-1">
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-1">Tags</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 w-full focus:border-blue-500 outline-none"
              placeholder="tag1, tag2, tag3"
            />
          </div>
        </div>
      </div>

      {/* Extraction progress (shown during video extraction) */}
      {extracting && renderExtractionProgress()}

      {/* Upload zone (hidden during extraction) */}
      {!extracting && (
        <div className="mb-8">
          <AssetUploader
            onFilesSelected={handleFilesSelected}
            onVideoSelected={handleVideoSelected}
            disabled={saving}
          />
        </div>
      )}

      {/* Import from Scene */}
      {!extracting && (
        <div className="mb-8 rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-1">
            Import from Scene
          </h3>
          <p className="text-xs text-gray-500 mb-3">
            Paste a scene ID to auto-create assets from its storyboard
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={sceneImportId}
              onChange={(e) => setSceneImportId(e.target.value)}
              placeholder="Scene ID (UUID)"
              disabled={importing || saving}
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-purple-500 outline-none disabled:opacity-50"
            />
            <button
              onClick={handleSceneImport}
              disabled={importing || saving || !sceneImportId.trim()}
              className="bg-purple-600 hover:bg-purple-500 text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
            >
              {importing ? "Importing..." : "Import"}
            </button>
          </div>
        </div>
      )}

      {/* Video info banner */}
      {!extracting && renderVideoInfoBanner()}

      {/* Asset list */}
      <div className="mb-8">
        <h3 className="text-lg font-medium text-gray-200 mb-4">
          Assets ({assets.length})
        </h3>
        {assets.length === 0 && !extracting ? (
          <p className="text-sm text-gray-500 text-center py-8">
            No assets yet. Drop images or a video above to add assets.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {assets.map((asset) => (
              <AssetEditor
                key={asset.asset_id}
                asset={asset}
                imageFile={pendingFiles.get(asset.asset_id)}
                onUpdate={handleAssetUpdate}
                onDelete={handleAssetDelete}
                isUploading={uploadingAssets.has(asset.asset_id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Asset Library Bindings — available in Stage 1 for existing bibles */}
      {manifest?.production_bible_id && !extracting && (
        <div className="mb-8">
          <div className="mb-4 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
            <h3 className="text-sm font-semibold text-gray-200 mb-1">
              Asset Library
            </h3>
            <p className="text-xs text-gray-500">
              Bind reusable assets from the Asset Library to this Production Bible. Use @tags in scene prompts to reference them.
            </p>
          </div>
          {renderDepartmentTabs()}
        </div>
      )}

      {/* Footer actions */}
      <div className="flex justify-end gap-3 pt-6 border-t border-gray-800">
        <button
          onClick={onCancel}
          className="text-gray-400 hover:text-gray-300 px-4 py-2 text-sm"
        >
          Cancel
        </button>
        {assets.length > 0 && !extracting && (
          <button
            onClick={handleProcess}
            disabled={processing}
            className="bg-green-600 hover:bg-green-500 text-white rounded-lg px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? "Starting..." : "Process"}
          </button>
        )}
        <button
          onClick={handleSave}
          disabled={saving || extracting || !name.trim()}
          className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {saving ? "Saving..." : "Save Draft"}
        </button>
      </div>
    </>
  );

  // STAGE 2: Processing progress UI
  const renderStage2 = () => {
    const stepLabel =
      STEP_LABELS[progress?.current_step || ""] || "Processing...";
    const uploadsTotal = progress?.progress?.uploads_total || 0;
    const uploadsProcessed = progress?.progress?.uploads_processed || 0;
    const cropsTotal = progress?.progress?.crops_total || 0;
    const cropsReversePrompted =
      progress?.progress?.crops_reverse_prompted || 0;

    const uploadsProgress =
      uploadsTotal > 0 ? (uploadsProcessed / uploadsTotal) * 100 : 0;
    const cropsProgress =
      cropsTotal > 0 ? (cropsReversePrompted / cropsTotal) * 100 : 0;

    return (
      <div className="flex flex-col items-center justify-center py-16">
        <div className="w-full max-w-lg">
          {/* Status header */}
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-white mb-2">
              Processing Production Bible
            </h2>
            <p className="text-sm text-gray-400">{manifest?.name}</p>
          </div>

          {/* Spinner */}
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>

          {/* Current step */}
          <div className="text-center mb-8">
            <p className="text-lg text-gray-300">{stepLabel}</p>
          </div>

          {/* Progress bars */}
          <div className="space-y-6">
            {/* Images processed */}
            <div>
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Images processed</span>
                <span>
                  {uploadsProcessed} / {uploadsTotal}
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${uploadsProgress}%` }}
                ></div>
              </div>
            </div>

            {/* Assets described */}
            {cropsTotal > 0 && (
              <div>
                <div className="flex justify-between text-sm text-gray-400 mb-2">
                  <span>Assets described</span>
                  <span>
                    {cropsReversePrompted} / {cropsTotal}
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-green-500 h-full transition-all duration-300"
                    style={{ width: `${cropsProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          {/* Face merges info */}
          {progress?.progress?.face_merges !== undefined &&
            progress.progress.face_merges > 0 && (
              <div className="mt-6 text-center text-sm text-gray-400">
                <p>
                  Merged {progress.progress.face_merges} duplicate face
                  {progress.progress.face_merges === 1 ? "" : "s"}
                </p>
              </div>
            )}
        </div>
      </div>
    );
  };

  // STAGE 3: Review/refine UI with department tabs
  const renderStage3 = () => {
    const uploadedAssets = assets.filter((a) => a.source === "uploaded");
    const extractedAssets = assets.filter((a) => a.source === "extracted");
    const videoFrameAssets = assets.filter((a) => a.source === "video_frame");

    const assetsByType = assets.reduce(
      (acc, asset) => {
        acc[asset.asset_type] = (acc[asset.asset_type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    // Filtered assets based on active department tab
    const activeTabConfig = DEPARTMENT_TABS.find((t) => t.id === activeTab);
    const filteredAssets = activeTab === "sound"
      ? []
      : assets.filter((a) => activeTabConfig?.assetTypes.includes(a.asset_type));

    return (
      <>
        {/* Header */}
        <div className="mb-8 rounded-lg border border-gray-800 bg-gray-900/50 p-6">
          <h2 className="text-xl font-bold text-white mb-2">
            Review & Refine Assets
          </h2>
          <p className="text-sm text-gray-400 mb-4">{manifest?.name}</p>

          {/* Summary stats */}
          <div className="flex gap-6 text-sm flex-wrap">
            <div>
              <span className="text-gray-400">Total assets:</span>{" "}
              <span className="text-white font-medium">{assets.length}</span>
            </div>
            {uploadedAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Uploaded:</span>{" "}
                <span className="text-white font-medium">
                  {uploadedAssets.length}
                </span>
              </div>
            )}
            {videoFrameAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Video frames:</span>{" "}
                <span className="text-purple-300 font-medium">
                  {videoFrameAssets.length}
                </span>
              </div>
            )}
            {extractedAssets.length > 0 && (
              <div>
                <span className="text-gray-400">Extracted:</span>{" "}
                <span className="text-white font-medium">
                  {extractedAssets.length}
                </span>
              </div>
            )}
            {Object.entries(assetsByType).map(([type, count]) => (
              <div key={type}>
                <span className="text-gray-400">{type}:</span>{" "}
                <span className="text-white font-medium">{count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Add more assets */}
        {!extracting && (
          <div className="mb-8">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Add more references
            </h3>
            <AssetUploader
              onFilesSelected={handleFilesSelected}
              onVideoSelected={handleVideoSelected}
              disabled={processing}
            />
          </div>
        )}

        {extracting && extractionProgress && renderExtractionProgress()}

        {renderDepartmentTabs()}

        {/* Collapsible Raw Assets section */}
        <div className="mb-8">
          <button
            onClick={() => setShowRawAssets(!showRawAssets)}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 mb-3"
          >
            <span className={`transition-transform ${showRawAssets ? "rotate-90" : ""}`}>&#9654;</span>
            Raw Assets ({filteredAssets.length})
          </button>

          {showRawAssets && (
            <div className="space-y-4">
              {filteredAssets.length === 0 ? (
                <div className="flex min-h-[120px] items-center justify-center rounded-lg border border-dashed border-gray-700">
                  <p className="text-sm text-gray-500">
                    No {activeTabConfig?.label} assets yet.
                  </p>
                </div>
              ) : (
                filteredAssets.map((asset) => (
                  <div
                    key={asset.asset_id}
                    className="rounded-lg border border-gray-800 bg-gray-900/50 p-4"
                  >
                    <div className="flex gap-4">
                      {/* Thumbnail */}
                      <div className="flex-shrink-0 space-y-1.5">
                        {asset.reference_image_url ? (
                          <img
                            src={asset.reference_image_url}
                            alt={asset.name}
                            onClick={() => {
                              setLightboxUrl(asset.reference_image_url);
                              setLightboxName(asset.name);
                            }}
                            className="w-32 h-32 object-cover rounded border border-gray-700 cursor-pointer transition-opacity hover:opacity-80"
                            title="Click to enlarge"
                          />
                        ) : (
                          <div className="w-32 h-32 bg-gray-800 rounded border border-gray-700 flex items-center justify-center text-gray-600 text-xs">
                            No image
                          </div>
                        )}
                        {/* Image action buttons */}
                        {asset.reference_image_url && (
                          <div className="flex gap-1.5 justify-center">
                            <button
                              onClick={() => copyImage(asset.reference_image_url!, `img-${asset.asset_id}`)}
                              className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
                              title="Copy image to clipboard"
                            >
                              {copiedField === `img-${asset.asset_id}` ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                            </button>
                            <a
                              href={asset.reference_image_url}
                              download={`${asset.name}.png`}
                              onClick={(e) => e.stopPropagation()}
                              className="flex items-center px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
                              title="Download image"
                            >
                              <SaveIcon size={12} />
                            </a>
                          </div>
                        )}
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        {/* Name + badges row */}
                        <div className="flex items-start gap-2 mb-2">
                          <input
                            type="text"
                            value={asset.name}
                            onChange={(e) =>
                              handleAssetUpdate(asset.asset_id, {
                                name: e.target.value,
                              })
                            }
                            className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-white text-sm font-medium flex-1 min-w-0 focus:border-blue-500 outline-none"
                          />
                          <span className="px-2 py-1 text-xs font-medium bg-blue-900 text-blue-300 rounded">
                            {asset.manifest_tag}
                          </span>
                          <select
                            value={asset.asset_type}
                            onChange={(e) =>
                              handleAssetUpdate(asset.asset_id, {
                                asset_type: e.target.value,
                              })
                            }
                            className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:border-blue-500 outline-none"
                          >
                            {ASSET_TYPES.map((type) => (
                              <option key={type} value={type}>
                                {type}
                              </option>
                            ))}
                          </select>
                        </div>

                        {/* Quality score + detection info */}
                        <div className="flex items-center gap-3 mb-3 text-xs">
                          {asset.quality_score !== null && (
                            <span
                              className={`px-2 py-1 rounded font-medium ${getQualityBadgeColor(asset.quality_score)}`}
                            >
                              Quality: {asset.quality_score.toFixed(1)}
                            </span>
                          )}
                          {asset.detection_class && (
                            <span className="text-gray-500">
                              {asset.detection_class}{" "}
                              {asset.detection_confidence !== null &&
                                `(${(asset.detection_confidence * 100).toFixed(0)}%)`}
                            </span>
                          )}
                          <span
                            className={`px-2 py-1 rounded ${
                              asset.source === "uploaded"
                                ? "bg-purple-900 text-purple-300"
                                : asset.source === "video_frame"
                                  ? "bg-purple-900 text-purple-300"
                                  : "bg-gray-800 text-gray-400"
                            }`}
                          >
                            {asset.source === "video_frame" ? "video frame" : asset.source}
                          </span>
                          {asset.is_face_crop && (
                            <span className="px-2 py-1 rounded bg-indigo-900 text-indigo-300">
                              face
                            </span>
                          )}
                        </div>

                        {/* Reverse prompt */}
                        <div className="mb-3">
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs font-medium text-gray-400">
                              Reverse Prompt
                            </label>
                            <div className="flex items-center gap-2">
                              {asset.reverse_prompt && (
                                <button
                                  onClick={() => copyText(asset.reverse_prompt!, `rp-${asset.asset_id}`)}
                                  className="text-gray-500 hover:text-gray-300 transition-colors"
                                  title="Copy reverse prompt"
                                >
                                  {copiedField === `rp-${asset.asset_id}` ? <CheckIcon /> : <CopyIcon />}
                                </button>
                              )}
                              <button
                                onClick={() =>
                                  toggleFieldEdit(asset.asset_id, "reverse_prompt")
                                }
                                className="text-xs text-blue-400 hover:text-blue-300"
                              >
                                {isFieldEditing(asset.asset_id, "reverse_prompt")
                                  ? "Save"
                                  : "Edit"}
                              </button>
                            </div>
                          </div>
                          {isFieldEditing(asset.asset_id, "reverse_prompt") ? (
                            <textarea
                              value={asset.reverse_prompt || ""}
                              onChange={(e) =>
                                handleAssetUpdate(asset.asset_id, {
                                  reverse_prompt: e.target.value,
                                })
                              }
                              rows={3}
                              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
                            />
                          ) : (
                            <p className="text-sm text-gray-300">
                              {asset.reverse_prompt || (
                                <span className="text-gray-600 italic">
                                  No reverse prompt
                                </span>
                              )}
                            </p>
                          )}
                        </div>

                        {/* Visual description */}
                        <div className="mb-3">
                          <div className="flex items-center justify-between mb-1">
                            <label className="text-xs font-medium text-gray-400">
                              Visual Description
                            </label>
                            <div className="flex items-center gap-2">
                              {asset.visual_description && (
                                <button
                                  onClick={() => copyText(asset.visual_description!, `vd-${asset.asset_id}`)}
                                  className="text-gray-500 hover:text-gray-300 transition-colors"
                                  title="Copy visual description"
                                >
                                  {copiedField === `vd-${asset.asset_id}` ? <CheckIcon /> : <CopyIcon />}
                                </button>
                              )}
                              <button
                                onClick={() =>
                                  toggleFieldEdit(asset.asset_id, "visual_description")
                                }
                                className="text-xs text-blue-400 hover:text-blue-300"
                              >
                                {isFieldEditing(asset.asset_id, "visual_description")
                                  ? "Save"
                                  : "Edit"}
                              </button>
                            </div>
                          </div>
                          {isFieldEditing(asset.asset_id, "visual_description") ? (
                            <textarea
                              value={asset.visual_description || ""}
                              onChange={(e) =>
                                handleAssetUpdate(asset.asset_id, {
                                  visual_description: e.target.value,
                                })
                              }
                              rows={2}
                              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:border-blue-500 outline-none resize-none"
                            />
                          ) : (
                            <p className="text-sm text-gray-300">
                              {asset.visual_description || (
                                <span className="text-gray-600 italic">
                                  No visual description
                                </span>
                              )}
                            </p>
                          )}
                        </div>

                        {/* Action buttons */}
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleReprocess(asset.asset_id)}
                            disabled={reprocessingAssets.has(asset.asset_id)}
                            className="text-xs px-3 py-1 rounded bg-yellow-900 text-yellow-300 hover:bg-yellow-800 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {reprocessingAssets.has(asset.asset_id)
                              ? "Processing..."
                              : "Re-process"}
                          </button>
                          <button
                            onClick={() => handleAssetDelete(asset.asset_id)}
                            className="text-xs px-3 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="flex justify-between pt-6 border-t border-gray-800">
          <button
            onClick={handleReprocessAll}
            disabled={processing}
            className="text-sm px-4 py-2 rounded bg-yellow-900 text-yellow-300 hover:bg-yellow-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? "Processing..." : "Reprocess All"}
          </button>
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="text-gray-400 hover:text-gray-300 px-4 py-2 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => manifest?.production_bible_id && onSaved(manifest.production_bible_id)}
              className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-2 text-sm font-medium"
            >
              Done
            </button>
          </div>
        </div>
      </>
    );
  };

  return (
    <div>
      {/* Back link */}
      <button
        onClick={onCancel}
        className="text-sm text-gray-400 hover:text-gray-300 mb-6"
      >
        ← Back to Library
      </button>

      {/* Error banner */}
      {error && (
        <div className="mb-6 rounded-lg bg-red-900/30 border border-red-700 px-4 py-3 flex items-start justify-between">
          <p className="text-sm text-red-300">{error}</p>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-300 ml-4"
          >
            ✕
          </button>
        </div>
      )}

      {/* Render current stage */}
      {currentStage === 1 && renderStage1()}
      {currentStage === 2 && renderStage2()}
      {currentStage === 3 && renderStage3()}

      {/* Image lightbox modal */}
      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/80"
          onClick={() => setLightboxUrl(null)}
        >
          {/* Close button - top right of viewport */}
          <button
            onClick={() => setLightboxUrl(null)}
            className="absolute top-4 right-4 rounded-full bg-gray-900/80 p-2 text-gray-300 backdrop-blur transition-colors hover:bg-gray-800 hover:text-white"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>

          {/* Image */}
          <div
            className="flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={lightboxUrl}
              alt={lightboxName}
              className="max-h-[80vh] max-w-[90vw] rounded-lg object-contain"
            />
          </div>

          {/* Toolbar below image */}
          <div
            className="mt-3 flex items-center gap-3"
            onClick={(e) => e.stopPropagation()}
          >
            <span className="text-sm text-gray-400">{lightboxName}</span>
            <button
              onClick={() => copyImage(lightboxUrl!, "lightbox-img")}
              className="flex items-center gap-1.5 rounded-lg bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-200 transition-colors hover:bg-gray-700"
            >
              {copiedField === "lightbox-img" ? <><CheckIcon /> Copied</> : <><CopyIcon /> Copy Image</>}
            </button>
            <a
              href={lightboxUrl}
              download={`${lightboxName}.png`}
              className="flex items-center gap-1.5 rounded-lg bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-200 transition-colors hover:bg-gray-700"
            >
              <SaveIcon /> Download
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
