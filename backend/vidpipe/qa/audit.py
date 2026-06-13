"""Adversarial production audit — orchestrator, criteria evaluation, report, CLI.

    python -m vidpipe.qa.audit <production_id> [--out DIR] [--api URL]
                               [--sample-every N] [--no-embeddings] [--no-vision]

Exit code is non-zero when the audit FAILS (any CRITICAL finding), so it gates
CI / the pytest wrapper. Designed to run where ``vidpipe`` imports and Vertex/
Ollama credentials are available (e.g. inside the backend container).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

from vidpipe.qa import audio as audio_mod
from vidpipe.qa import vision as vision_mod
from vidpipe.qa import criteria as C
from vidpipe.qa.criteria import Finding, Severity, Verdict

logger = logging.getLogger("vidpipe.qa")


# --------------------------------------------------------------------------
# Production graph
# --------------------------------------------------------------------------
def fetch_graph(api: str, pid: str) -> dict[str, Any]:
    with httpx.Client(base_url=api, timeout=60, follow_redirects=True) as cx:
        prod = cx.get(f"/api/productions/{pid}").raise_for_status().json()
        all_scenes = cx.get("/api/scenes").raise_for_status().json()
        items = all_scenes if isinstance(all_scenes, list) else all_scenes.get(
            "scenes", all_scenes.get("items", []))
        # scene_order lives on the list payload; the detail endpoint omits it, so
        # carry it through explicitly (otherwise scene labels/timeline scramble).
        order_by_id = {(s.get("scene_id") or s.get("id")): s.get("scene_order", 0)
                       for s in items if s.get("production_id") == pid}
        scenes = []
        for sid, order in order_by_id.items():
            d = cx.get(f"/api/scenes/{sid}").raise_for_status().json()
            d["scene_order"] = order
            scenes.append(d)
        scenes.sort(key=lambda d: d["scene_order"])

        # primary_ref_url lives on the actors LIST payload, not the detail endpoint.
        actors_list = cx.get("/api/asset-library/actors").json()
        actors_list = actors_list if isinstance(actors_list, list) else \
            actors_list.get("actors", actors_list.get("items", []))
        ref_by_actor = {a.get("id"): a.get("primary_ref_url") for a in actors_list}

        # Characters from the bible cast (on-screen only; with reference photos).
        characters: dict[str, dict] = {}
        bible_id = prod.get("production_bible_id")
        if bible_id:
            cast = cx.get(f"/api/production-bibles/{bible_id}/cast").json()
            cast = cast if isinstance(cast, list) else cast.get("cast", cast.get("items", []))
            for c in cast:
                if (c.get("role") or "").upper() == "NARRATOR":
                    continue
                actor_id = c.get("actor_id")
                ref_bytes = None
                ref_url = ref_by_actor.get(actor_id)
                if ref_url:
                    # Use the full-res image, not the thumbnail, for embeddings.
                    r = cx.get(ref_url.replace("?thumb=true", ""))
                    if r.status_code == 200:
                        ref_bytes = r.content
                characters[c.get("tag")] = {
                    "name": c.get("character_name"),
                    "appearance": c.get("character_description") or "",
                    "role": c.get("role"),
                    "ref_bytes": ref_bytes,
                }
    return {"production": prod, "scenes": scenes, "characters": characters}


def _download(api: str, url: str, dest: Path) -> Optional[Path]:
    if not url:
        return None
    with httpx.Client(base_url=api, timeout=120, follow_redirects=True) as cx:
        r = cx.get(url)
        if r.status_code != 200:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
    return dest


def _sample_frames(clip: Path, every_n: int, outdir: Path, limit: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    pat = str(outdir / "f_%03d.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(clip),
                    "-vf", f"select=not(mod(n\\,{every_n})),scale=512:-1",
                    "-vsync", "vfr", pat], capture_output=True, text=True)
    frames = sorted(outdir.glob("f_*.png"))
    if len(frames) > limit:  # even spread
        step = len(frames) / limit
        frames = [frames[int(i * step)] for i in range(limit)]
    return frames


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
async def _load_user_settings():
    try:
        from sqlalchemy import select
        from vidpipe.db.engine import async_session
        from vidpipe.db.models import UserSettings
        async with async_session() as s:
            return (await s.execute(select(UserSettings))).scalars().first()
    except Exception as e:  # noqa: BLE001
        logger.info("No UserSettings loaded (%s); Ollama cloud config may be absent.", e)
        return None


async def run_audit(api: str, pid: str, out: Path, *, sample_every: int,
                    do_embed: bool, do_vision: bool) -> Verdict:
    out.mkdir(parents=True, exist_ok=True)
    g = fetch_graph(api, pid)
    scenes, chars = g["scenes"], g["characters"]
    v = Verdict(production_id=pid, meta={
        "name": g["production"].get("name"),
        "n_scenes": len(scenes),
        "characters": {k: {"name": d["name"], "has_ref": d["ref_bytes"] is not None}
                       for k, d in chars.items()},
    })

    # Protagonist = first LEAD with a reference, else first character with a ref.
    lead = next((c for c in chars.values() if (c.get("role") or "").upper() == "LEAD"
                 and c["ref_bytes"]), None) or \
        next((c for c in chars.values() if c["ref_bytes"]), None)
    v.meta["protagonist"] = lead["name"] if lead else None
    if lead and lead["ref_bytes"]:
        (out / "reference.png").write_bytes(lead["ref_bytes"])

    timeline: list[dict] = []   # cumulative shot offsets in the concatenated master
    cursor = 0.0

    vision_model = scenes[0].get("vision_model") if scenes else None
    v.meta["vision_model"] = vision_model
    v.meta["embeddings_available"] = bool(do_embed and vision_mod.embeddings_available())
    adapter = None
    if do_vision and vision_model:
        # Reproduce the app's startup credential bootstrap so the selected vision
        # LLM (Vertex/Gemini or Ollama) authenticates exactly as the pipeline does.
        try:
            from vidpipe.services.vertex_client import load_vertex_credentials_from_db
            await load_vertex_credentials_from_db()
        except Exception as e:  # noqa: BLE001
            logger.warning("Vertex credential bootstrap failed: %s", e)
        from vidpipe.services.llm.registry import get_adapter
        adapter = get_adapter(vision_model, await _load_user_settings())

    for si, sc in enumerate(scenes, 1):
        sid = sc.get("scene_id") or sc.get("id")
        slug = sc.get("title") or f"scene {si}"
        sdir = out / f"S{si}"
        shots = sc.get("shots", [])
        audio_expected = bool(sc.get("audio_enabled"))

        for shot in shots:
            i = shot.get("shot_index")
            tag = f"S{si} shot {i}"
            # ---- establishing keyframe identity (scene's shot 0 start) ----
            if i == 0 and lead:
                kf = _download(api, shot.get("start_keyframe_url"), sdir / "establish.png")
                if kf:
                    # embedding
                    if v.meta["embeddings_available"]:
                        sim = vision_mod.embedding_similarity(lead["ref_bytes"], kf.read_bytes())
                        if sim is None:
                            v.add(Finding("identity.no_face", Severity.WARNING,
                                          f"{tag} establishing kf",
                                          "No face detected to embed (stylization or wrong subject)."))
                        elif sim < C.FACE_EMBED_MIN:
                            v.add(Finding("identity.embed_low", Severity.WARNING,
                                          f"{tag} establishing kf",
                                          f"Face-embedding similarity {sim:.2f} < {C.FACE_EMBED_MIN}.",
                                          {"similarity": sim}))
                    # vision LLM
                    if adapter:
                        verdict = await vision_mod.judge_identity(
                            adapter, kf.read_bytes(), character_name=lead["name"],
                            appearance=lead["appearance"],
                            scene_intent=f"{slug}: {sc.get('prompt','')[:300]}")
                        if verdict is None:
                            v.add(Finding("vision.error", Severity.WARNING, f"{tag} establishing kf",
                                          "Vision-LLM judge returned no verdict."))
                        else:
                            sev = (Severity.CRITICAL if verdict.identity_match < C.IDENTITY_VISION_MIN
                                   else Severity.WARNING if verdict.identity_match < C.IDENTITY_VISION_WARN
                                   else Severity.INFO)
                            if sev != Severity.INFO:
                                v.add(Finding(
                                    "identity.establishing", sev, f"{tag} establishing kf",
                                    f"{lead['name']} identity_match={verdict.identity_match:.2f} "
                                    f"({'; '.join(verdict.discrepancies) or 'see notes'}).",
                                    {"identity_match": verdict.identity_match,
                                     "observed": verdict.observed_subject,
                                     "discrepancies": verdict.discrepancies,
                                     "vision_model": vision_model}))
                            if verdict.scene_alignment < 0.5:
                                v.add(Finding("story.scene_alignment", Severity.WARNING,
                                              f"{tag} establishing kf",
                                              f"Weak scene alignment {verdict.scene_alignment:.2f}: "
                                              f"{verdict.scene_notes[:160]}",
                                              {"scene_alignment": verdict.scene_alignment}))

            # ---- per-shot clip audio + identity drift on sampled frames ----
            clip = _download(api, shot.get("clip_url"), sdir / f"shot{i}.mp4")
            if clip:
                m = audio_mod.analyze(str(clip), tag)
                if m.duration:
                    timeline.append({"start": cursor, "end": cursor + m.duration, "label": tag})
                    cursor += m.duration
                for f in audio_mod.findings(m, expect_audio=audio_expected):
                    v.add(f)
                if v.meta["embeddings_available"] and lead:
                    frames = _sample_frames(clip, sample_every, sdir / f"shot{i}_frames", limit=3)
                    sims = [vision_mod.embedding_similarity(lead["ref_bytes"], fp.read_bytes())
                            for fp in frames]
                    sims = [s for s in sims if s is not None]
                    if sims and min(sims) < C.FACE_EMBED_MIN:
                        v.add(Finding("identity.drift", Severity.WARNING, tag,
                                      f"Identity drift in clip: min frame similarity {min(sims):.2f}.",
                                      {"frame_similarities": [round(s, 3) for s in sims]}))

    # ---- master audio integrity + garble localization ----
    master = _download(api, f"/api/productions/{pid}/master-video", out / "master.mp4")
    if master:
        mm = audio_mod.analyze(str(master), "master")
        v.meta["master_audio"] = mm.as_dict()

        def _attribute(win: dict) -> str:
            mid = (win["start"] + win["end"]) / 2
            for seg in timeline:
                if seg["start"] <= mid < seg["end"]:
                    return seg["label"]
            return "master tail"

        for f in audio_mod.findings(mm, expect_audio=True):
            if f.code == "audio.garble" and f.evidence.get("window"):
                shot_lbl = _attribute(f.evidence["window"])
                f.where = f"{shot_lbl} (master {f.evidence['window']['start']:.0f}-{f.evidence['window']['end']:.0f}s)"
                f.message = f"{f.message} Attributed to {shot_lbl}."
            v.add(f)
        audio_mod.render_spectrogram(str(master), str(out / "master_spectrogram.png"))

    _write_reports(v, out)
    return v


def _write_reports(v: Verdict, out: Path) -> None:
    (out / "report.json").write_text(json.dumps(v.as_dict(), indent=2))
    lines = [f"# Video QA Audit — {v.meta.get('name')}",
             f"\n**Production:** `{v.production_id}`  ",
             f"**Result:** {'✅ PASS' if v.passed else '❌ FAIL'}  ",
             f"**Critical:** {len(v.critical)} · **Warnings:** {len(v.warnings)}  ",
             f"**Vision model:** {v.meta.get('vision_model')} · "
             f"**Face embeddings:** {'on' if v.meta.get('embeddings_available') else 'unavailable'}  ",
             f"**Protagonist:** {v.meta.get('protagonist')}\n"]
    for sev in (Severity.CRITICAL, Severity.WARNING, Severity.INFO):
        fs = [f for f in v.findings if f.severity == sev]
        if not fs:
            continue
        lines.append(f"\n## {sev.value.upper()} ({len(fs)})\n")
        for f in fs:
            lines.append(f"- **[{f.code}]** _{f.where}_ — {f.message}")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial video-QA audit of a production.")
    ap.add_argument("production_id")
    ap.add_argument("--api", default=os.getenv("VIDPIPE_QA_API_BASE", "http://localhost:8000"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(a.out or f"/tmp/qa/{a.production_id}")
    v = asyncio.run(run_audit(a.api, a.production_id, out,
                              sample_every=a.sample_every,
                              do_embed=not a.no_embeddings, do_vision=not a.no_vision))
    print((out / "report.md").read_text())
    print(f"\nReports + evidence in: {out}")
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
