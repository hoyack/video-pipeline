# Camera Pan Head-Cutoff Issue — Investigation & Recommendations

## Problem Statement

When generating videos using the WAN 2.2 (ComfyUI) pipeline, characters frequently appear with their heads cut off at the start of the clip. The camera then pans/tilts upward to reveal the full character, but by that point the model has lost the character's likeness — the face and features no longer match the reference image.

## Root Cause Analysis

Three independent issues compound to produce this behavior:

### 1. Storyboard Prompts Allow Incompatible Framing + Motion Combinations

The storyboard LLM generates a `camera_movement` field (e.g., `pan_up`, `tilt_up`) in the shot manifest, and separately generates `start_frame_prompt` with a camera/framing description. These two outputs are **not coordinated**.

**Failure scenario:**
- Start keyframe prompt: character framed center-to-upper in a medium shot
- Camera movement: `pan_up` or `tilt_up`
- Result: camera moves upward from an already high-framed subject → head exits frame

The relevant prompt sections (`storyboard.py` lines 230-235, 1159-1164) instruct the LLM to describe motion and camera movement but include **no rules** about ensuring the subject stays in frame throughout the movement.

```
VIDEO MOTION PROMPT FORMAT (video_motion_prompt):
Describe ONLY motion and camera movement...
Focus on: camera movement (pan, dolly, track, crane), subject animation...
```

No constraint like: *"If camera pans up, the subject must be positioned in the lower portion of the start keyframe."*

### 2. WAN 2.2 I2V Has No Character Reference Image Support

The pipeline loads character reference images (`video_gen.py` lines 1043-1051) and passes them to the ComfyUI adapter, but **the adapter ignores them entirely**:

| Layer | What happens |
|-------|-------------|
| `video_gen.py` | Calls `_load_char_ref_images()`, passes `char_ref_bytes` to adapter |
| `comfyui_adapter.py` | Accepts `char_ref_bytes` parameter but never uses it |
| `comfyui_client.py` | `build_wan22_i2v_workflow()` has no reference image parameters |
| WAN 2.2 I2V workflow JSON | Only uses start keyframe image + motion prompt text |

**Impact:** The WAN 2.2 I2V model generates video guided solely by the start keyframe image and the text motion prompt. It has **zero identity/face guidance** to maintain character likeness across frames. When camera motion moves away from the initial framing, the model hallucinates facial features with no anchor.

Compare with the FLF2V workflow (`build_wan22_flf2v_workflow`) which **does** accept `char_ref_01_filename` and `char_ref_02_filename` — but this workflow is not used for standard I2V generation.

### 3. Motion Prompt Guidelines Are Veo-Oriented

The prompt rewriter (`prompt_rewriter.py`) contains guidance like:

> "Do not re-describe visual appearance (reference images handle that)"

This is correct for **Veo 3.1** (which receives reference images via `reference_images` in the API), but misleading for **WAN 2.2** where no reference images are injected. For WAN, the motion prompt is the *only* text guidance the model receives — it should include more visual context, not less.

## Impact Matrix

| Issue | Severity | Frequency | Fix Complexity |
|-------|----------|-----------|----------------|
| No framing-motion coordination | High | Very common | Low (prompt change) |
| No reference images in WAN I2V | High | Always | Medium (workflow change) |
| Veo-oriented prompt guidance | Medium | Always for WAN | Low (conditional prompt) |

## Recommended Actions

### Action 1: Add Framing-Motion Safety Rules to Storyboard Prompts

**Files:** `backend/vidpipe/pipeline/storyboard.py` — both `STORYBOARD_SYSTEM_PROMPT` and `ENHANCED_STORYBOARD_PROMPT`

Add to the `VIDEO MOTION PROMPT FORMAT` section:

```
FRAMING-MOTION SAFETY:
Camera movement must be compatible with subject positioning in the start keyframe.
- If camera PANS UP or TILTS UP: position the subject in the LOWER half of the start keyframe
- If camera PANS DOWN or TILTS DOWN: position the subject in the UPPER half of the start keyframe
- If camera DOLLIES IN: ensure subject face is clearly visible and centered in start keyframe
- NEVER generate motion that would push the primary subject's head out of frame
- PREFER subtle camera movements (slow dolly, gentle tracking) over dramatic pans/tilts
  when the subject must remain recognizable throughout
- When in doubt, use a STATIC or SLOW DOLLY camera — these preserve character identity best
```

Also add to the `KEYFRAME PROMPT FORMAT` section (item 6, CAMERA):

```
6. CAMERA: Shot type (wide/medium/close-up), angle, lens.
   CRITICAL: The subject's framing must be compatible with the video_motion_prompt's camera movement.
   If the motion prompt calls for upward camera movement, frame the subject LOW in the composition.
```

**Effort:** ~30 minutes. Immediate impact on all new generations.

### Action 2: Add WAN-Specific Motion Prompt Enrichment

**Files:** `backend/vidpipe/pipeline/video_gen.py` or `backend/vidpipe/services/prompt_rewriter.py`

Before submitting to the ComfyUI adapter for WAN models, prepend framing guidance to the motion prompt:

```python
if video_model in COMFYUI_VIDEO_MODELS:
    motion_prompt = (
        "Keep the subject's face and full head visible throughout the entire clip. "
        "Maintain consistent character appearance and proportions. "
        + motion_prompt
    )
```

Also consider appending to the WAN negative prompt (`WAN_I2V_NEGATIVE_PROMPT` in `comfyui_client.py`):

```
"head cut off, face out of frame, subject partially visible, inconsistent face"
```

**Effort:** ~15 minutes. Low-risk, additive change.

### Action 3: Wire Character Reference Images into WAN 2.2 I2V Workflow

**Files:**
- `backend/vidpipe/services/comfyui_client.py` — `build_wan22_i2v_workflow()`
- `backend/vidpipe/services/comfyui_adapter.py` — `submit()`
- `docs/video_wan2_2_14B_i2v.json` — workflow template

This requires adding reference image nodes to the ComfyUI workflow. The FLF2V workflow already demonstrates how to do this (nodes 201, 202 for character refs). The same pattern can be adapted for I2V:

1. Upload character ref images to ComfyUI via the existing upload mechanism
2. Add `LoadImage` nodes for each ref in the workflow
3. Wire them as conditioning alongside the start keyframe

**Effort:** 2-4 hours. Requires ComfyUI workflow testing. Highest long-term impact but most complex.

### Action 4: Clean Up Dead Code Path

**Files:** `backend/vidpipe/pipeline/video_gen.py`, `backend/vidpipe/services/comfyui_adapter.py`

If Action 3 is deferred, remove or clearly mark the unused `char_ref_bytes` loading and passing to avoid confusion:

```python
# TODO: char_ref_bytes loaded but not yet wired into WAN 2.2 I2V workflow
# See docs/camera-pan.md Action 3 for the plan
```

**Effort:** 5 minutes.

## Priority Order

1. **Action 1** (storyboard prompt fix) — immediate, highest ROI
2. **Action 2** (WAN motion prompt enrichment) — quick, additive safety net
3. **Action 4** (dead code cleanup) — hygiene
4. **Action 3** (reference image wiring) — when ComfyUI workflow iteration is feasible
