"""ManifestingEngine orchestrator for complete CV/AI processing pipeline.

This module orchestrates the full manifesting pipeline: contact sheet assembly,
YOLO detection, face cross-matching, reverse-prompting, and tag assignment.
"""

import asyncio
import base64
import logging
import math
import shutil
import uuid
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Manifest
from vidpipe.services.cv_detection import CVDetectionService
from vidpipe.services.entity_extraction import _yolo_class_to_asset_type
from vidpipe.services.face_matching import FaceMatchingService
from vidpipe.services.reverse_prompt_service import ReversePromptService

logger = logging.getLogger(__name__)


class ManifestingEngine:
    """Orchestrator for the full manifesting pipeline."""

    def __init__(self, session: AsyncSession):
        """Initialize engine with database session.

        Args:
            session: Active async database session
        """
        self.session = session
        self.cv_detector = CVDetectionService()
        self.face_matcher = FaceMatchingService()
        self.reverse_prompter = ReversePromptService()

        # Progress tracking dict (shared reference with background task)
        self.progress = {
            "status": "initializing",
            "current_step": "initializing",
            "progress": {
                "uploads_total": 0,
                "uploads_processed": 0,
                "crops_total": 0,
                "crops_reverse_prompted": 0,
                "face_merges": 0,
            },
            "error": None,
        }

    def assemble_contact_sheet(
        self, manifest_id: uuid.UUID, assets: list[Asset]
    ) -> str:
        """Assemble contact sheet grid image (sync method for to_thread).

        Args:
            manifest_id: Manifest UUID
            assets: List of assets to include in contact sheet

        Returns:
            Path to saved contact sheet JPEG
        """
        # Grid layout: 4 columns, auto-calculated rows
        cols = 4
        rows = math.ceil(len(assets) / cols)
        thumb_size = 256
        label_height = 60
        cell_height = thumb_size + label_height
        title_height = 80

        # Canvas dimensions
        canvas_width = cols * thumb_size
        canvas_height = title_height + (rows * cell_height)

        # Create white canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        # Try to load DejaVu Sans font, fallback to default
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
            )
            label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except Exception:
            logger.warning("DejaVu font not found, using default font")
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Draw title
        draw.text(
            (20, 20),
            "PROJECT REFERENCE SHEET",
            fill="black",
            font=title_font,
        )

        # Place each asset thumbnail
        for idx, asset in enumerate(assets):
            row = idx // cols
            col = idx % cols

            x = col * thumb_size
            y = title_height + (row * cell_height)

            # Load and resize image
            try:
                # Resolve actual file path from reference_image_url
                if asset.reference_image_url and asset.reference_image_url.startswith("/api/assets/"):
                    # Pattern: /api/assets/{asset_id}/image
                    # Actual file: tmp/manifests/{manifest_id}/uploads/{asset_id}_*
                    pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
                    matches = glob(pattern)
                    if matches:
                        img_path = matches[0]
                    else:
                        logger.warning(f"Image not found for asset {asset.id}")
                        continue
                else:
                    # Direct path
                    img_path = asset.reference_image_url

                img = Image.open(img_path)
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.thumbnail((thumb_size, thumb_size))

                # Center thumbnail in cell
                thumb_x = x + (thumb_size - img.width) // 2
                thumb_y = y
                canvas.paste(img, (thumb_x, thumb_y))
            except Exception as e:
                logger.error(f"Failed to load image for asset {asset.id}: {e}")
                # Draw placeholder
                draw.rectangle(
                    [x, y, x + thumb_size, y + thumb_size],
                    outline="gray",
                    fill="lightgray",
                )

            # Draw label
            label_text = f"[{idx + 1}] {asset.name}\n{asset.asset_type}"
            draw.text(
                (x + 5, y + thumb_size + 5),
                label_text,
                fill="black",
                font=label_font,
            )

        # Save contact sheet
        output_dir = Path("tmp/manifests") / str(manifest_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "contact_sheet.jpg"
        canvas.save(output_path, format="JPEG", quality=90)

        logger.info(f"Contact sheet saved to {output_path}")
        return str(output_path)

    async def process_manifest(
        self, manifest_id: uuid.UUID, progress_callback=None
    ) -> dict:
        """Run the full manifesting pipeline.

        Args:
            manifest_id: Manifest UUID
            progress_callback: Optional callback for progress updates (unused, progress tracked via self.progress)

        Returns:
            Summary dict with processing statistics
        """
        logger.info(f"Starting manifesting pipeline for {manifest_id}")

        # Load manifest and assets
        manifest = await self.session.get(Manifest, manifest_id)
        if not manifest:
            raise ValueError(f"Manifest {manifest_id} not found")

        # Get all uploadable assets (uploaded images + video frames with reference_image_url)
        result = await self.session.execute(
            select(Asset).where(
                Asset.manifest_id == manifest_id,
                Asset.source.in_(["uploaded", "video_frame"]),
                Asset.reference_image_url.isnot(None),
            ).order_by(Asset.sort_order, Asset.created_at)
        )
        uploaded_assets = list(result.scalars().all())

        # Step 1: Assemble contact sheet
        self.progress["current_step"] = "contact_sheet"
        self.progress["status"] = "processing"
        logger.info("Step 1: Assembling contact sheet")

        contact_sheet_path = await asyncio.to_thread(
            self.assemble_contact_sheet, manifest_id, uploaded_assets
        )
        manifest.contact_sheet_url = f"/api/manifests/{manifest_id}/contact-sheet"
        await self.session.flush()

        # Step 2: YOLO detection and crop extraction
        self.progress["current_step"] = "yolo_detection"
        logger.info("Step 2: Running YOLO detection")

        # Find which uploads already have extracted children (incremental processing)
        existing_extracted = await self.session.execute(
            select(Asset.source_asset_id).where(
                Asset.manifest_id == manifest_id,
                Asset.source == "extracted",
                Asset.source_asset_id.isnot(None),
            )
        )
        already_processed_ids = {row[0] for row in existing_extracted.all()}

        # Filter to only unprocessed uploads
        new_uploads = [a for a in uploaded_assets if a.id not in already_processed_ids]
        if already_processed_ids:
            logger.info(
                f"Incremental processing: {len(new_uploads)} new uploads, "
                f"{len(already_processed_ids)} already processed"
            )

        self.progress["progress"]["uploads_total"] = len(new_uploads)

        # Load existing extracted assets (kept from previous runs)
        prev_extracted_result = await self.session.execute(
            select(Asset).where(
                Asset.manifest_id == manifest_id,
                Asset.source == "extracted",
            )
        )
        extracted_crops = list(prev_extracted_result.scalars().all())

        crops_dir = Path("tmp/manifests") / str(manifest_id) / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        for asset in new_uploads:
            # Resolve actual file path
            pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
            matches = glob(pattern)
            if not matches:
                logger.warning(f"Image not found for asset {asset.id}, skipping YOLO")
                continue

            img_path = matches[0]

            # Run YOLO detection in thread
            detections = await asyncio.to_thread(
                self.cv_detector.detect_objects_and_faces, img_path
            )

            # Create asset for each detection (objects + faces)
            all_detections = [
                {**d, "is_face": False} for d in detections["objects"]
            ] + [
                {**d, "is_face": True} for d in detections["faces"]
            ]

            for det in all_detections:
                crop_filename = f"{uuid.uuid4()}.jpg"
                crop_path = crops_dir / crop_filename

                # Save crop to disk
                await asyncio.to_thread(
                    self._save_crop, img_path, det["bbox"], str(crop_path)
                )

                # Create new Asset record
                new_asset = Asset(
                    manifest_id=manifest_id,
                    asset_type=_yolo_class_to_asset_type(det["class"]),
                    name=f"{asset.name} - {det['class']}",
                    manifest_tag="",  # Will be reassigned in finalization
                    reference_image_url=f"/api/assets/{{id}}/image",  # Placeholder, will be resolved
                    source="extracted",
                    source_asset_id=asset.id,
                    detection_class=det["class"],
                    detection_confidence=det["confidence"],
                    crop_bbox=det["bbox"],
                    is_face_crop=det.get("is_face", False),
                    sort_order=asset.sort_order,  # Inherit parent sort order
                )
                self.session.add(new_asset)
                await self.session.flush()

                # Update reference_image_url with actual asset ID
                new_asset.reference_image_url = f"/api/assets/{new_asset.id}/image"

                # Move crop to final location with asset ID prefix
                final_crop_path = crops_dir / f"{new_asset.id}_{crop_filename}"
                await asyncio.to_thread(
                    Path(crop_path).rename, final_crop_path
                )

                extracted_crops.append(new_asset)

            # Create scene/environment asset with detected objects masked out
            object_bboxes = [d["bbox"] for d in detections["objects"]]
            img_for_size = Image.open(img_path)
            w, h = img_for_size.size
            img_for_size.close()

            mask_coverage = self._bbox_mask_coverage(w, h, object_bboxes)
            if mask_coverage <= 0.40:
                scene_crop_filename = f"{uuid.uuid4()}.jpg"
                scene_crop_path = crops_dir / scene_crop_filename

                await asyncio.to_thread(
                    self._save_scene_masked, img_path, object_bboxes, str(scene_crop_path)
                )

                scene_asset = Asset(
                    manifest_id=manifest_id,
                    asset_type="ENVIRONMENT",
                    name=f"{asset.name} - scene",
                    manifest_tag="",
                    reference_image_url=f"/api/assets/{{id}}/image",
                    source="extracted",
                    source_asset_id=asset.id,
                    detection_class="scene",
                    detection_confidence=1.0,
                    is_face_crop=False,
                    sort_order=asset.sort_order,
                )
                self.session.add(scene_asset)
                await self.session.flush()

                scene_asset.reference_image_url = f"/api/assets/{scene_asset.id}/image"

                final_scene_path = crops_dir / f"{scene_asset.id}_{scene_crop_filename}"
                await asyncio.to_thread(
                    Path(scene_crop_path).rename, final_scene_path
                )

                extracted_crops.append(scene_asset)
            else:
                logger.info(
                    f"Skipping scene for {asset.name}: "
                    f"{mask_coverage:.0%} of image masked by detections"
                )

            self.progress["progress"]["uploads_processed"] += 1

        self.progress["progress"]["crops_total"] = len(extracted_crops)
        await self.session.flush()

        # Step 3: Face matching and cross-matching
        self.progress["current_step"] = "face_matching"
        logger.info("Step 3: Running face cross-matching")

        # Get all face crops — only generate embeddings for those without one
        face_crops = [a for a in extracted_crops if a.is_face_crop]

        if face_crops:
            # Generate embeddings (skip those already embedded)
            for face_asset in face_crops:
                if face_asset.face_embedding is not None:
                    continue

                pattern = f"tmp/manifests/{manifest_id}/crops/{face_asset.id}_*"
                matches = glob(pattern)
                if not matches:
                    continue

                crop_path = matches[0]
                try:
                    embedding = await asyncio.to_thread(
                        self.face_matcher.generate_embedding, crop_path
                    )
                    face_asset.face_embedding = embedding.tobytes()
                except ValueError:
                    logger.warning(f"No face detected in crop {face_asset.id}, skipping embedding")
                    face_asset.is_face_crop = False  # Reclassify — not actually a usable face crop

            await self.session.flush()

            # Cross-match faces
            face_crops_with_emb = [a for a in face_crops if a.face_embedding]

            if len(face_crops_with_emb) > 1:
                face_groups = await asyncio.to_thread(
                    self.face_matcher.cross_match_faces,
                    [{"embedding": bytes_to_np(a.face_embedding)} for a in face_crops_with_emb],
                )

                # Mark duplicates (keep highest confidence in each group)
                for group in face_groups:
                    if len(group) > 1:
                        # group contains integer indices into face_crops_with_emb
                        group_assets = [face_crops_with_emb[i] for i in group]
                        group_assets.sort(
                            key=lambda x: x.detection_confidence or 0.0, reverse=True
                        )

                        # Mark non-primary as duplicates
                        for dup_asset in group_assets[1:]:
                            dup_asset.description = (
                                f"(Duplicate of {group_assets[0].name}) "
                                + (dup_asset.description or "")
                            )

                        self.progress["progress"]["face_merges"] += len(group) - 1

        await self.session.flush()

        # Step 4: Reverse-prompting (only assets without existing reverse_prompt)
        self.progress["current_step"] = "reverse_prompting"
        logger.info("Step 4: Running reverse-prompting")

        all_assets = uploaded_assets + extracted_crops
        assets_needing_prompts = [a for a in all_assets if not a.reverse_prompt]
        logger.info(
            f"Reverse-prompting {len(assets_needing_prompts)} assets "
            f"({len(all_assets) - len(assets_needing_prompts)} already done)"
        )
        semaphore = asyncio.Semaphore(5)  # Rate limiting: 5 concurrent

        async def process_asset_reverse_prompt(asset: Asset):
            async with semaphore:
                # Resolve image path
                if asset.source in ("uploaded", "video_frame"):
                    pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
                else:
                    pattern = f"tmp/manifests/{manifest_id}/crops/{asset.id}_*"

                matches = glob(pattern)
                if not matches:
                    logger.warning(f"Image not found for asset {asset.id}")
                    return

                img_path = matches[0]

                # Run reverse-prompting
                result = await self.reverse_prompter.reverse_prompt_asset(
                    img_path, asset.asset_type, asset.name
                )

                # Update asset fields
                asset.reverse_prompt = result.get("reverse_prompt")
                asset.visual_description = result.get("visual_description")
                asset.quality_score = result.get("quality_score")

                # Update name if user didn't provide one and AI suggested one
                suggested_name = result.get("suggested_name")
                if suggested_name and (not asset.name or asset.name.startswith("Untitled")):
                    asset.name = suggested_name

                self.progress["progress"]["crops_reverse_prompted"] += 1

        # Process only assets that need reverse-prompting
        await asyncio.gather(*[process_asset_reverse_prompt(a) for a in assets_needing_prompts])
        await self.session.flush()

        # Step 5: Finalize - reassign manifest_tags
        self.progress["current_step"] = "finalizing"
        logger.info("Step 5: Finalizing tags and status")

        # Get all assets for this manifest, ordered by parent sort_order then confidence
        all_manifest_assets = await self.session.execute(
            select(Asset)
            .where(Asset.manifest_id == manifest_id)
            .order_by(Asset.sort_order, Asset.detection_confidence.desc())
        )
        all_manifest_assets = list(all_manifest_assets.scalars().all())

        # Reassign manifest_tags sequentially by type
        from vidpipe.services.manifest_service import TAG_PREFIX_MAP
        type_counters = {}

        for asset in all_manifest_assets:
            asset_type = asset.asset_type
            if asset_type not in type_counters:
                type_counters[asset_type] = 1
            else:
                type_counters[asset_type] += 1

            prefix = TAG_PREFIX_MAP.get(asset_type, "OTHER")
            asset.manifest_tag = f"{prefix}_{type_counters[asset_type]:02d}"

        # Update manifest final state
        manifest.asset_count = len(all_manifest_assets)
        manifest.status = "READY"
        await self.session.commit()

        self.progress["status"] = "complete"
        self.progress["current_step"] = "complete"

        logger.info(f"Manifesting complete: {len(uploaded_assets)} uploads, {len(extracted_crops)} crops")

        return {
            "uploads_processed": len(uploaded_assets),
            "crops_extracted": len(extracted_crops),
            "face_groups": self.progress["progress"]["face_merges"],
            "total_assets": len(all_manifest_assets),
        }

    async def process_new_uploads(
        self,
        manifest_id: uuid.UUID,
        new_uploads: list,  # list of NewUpload Pydantic models
        existing_face_embeddings: list,  # [(asset_id, embedding_bytes), ...]
    ) -> list:
        """Process new reference uploads added during a fork.

        Runs YOLO detection, face cross-matching (against inherited + new),
        and reverse-prompting. Does NOT update manifest status or contact sheet
        since the manifest is already READY from the parent.

        Args:
            manifest_id: Manifest UUID (shared with parent project)
            new_uploads: List of NewUpload Pydantic models (image_data base64, name, asset_type, ...)
            existing_face_embeddings: [(asset_id, embedding_bytes)] from inherited assets
                                      used for cross-matching against new face crops

        Returns:
            List of newly created Asset objects (both upload and extracted crops)
        """
        logger.info(
            f"process_new_uploads: {len(new_uploads)} uploads for manifest {manifest_id}"
        )

        from vidpipe.services.manifest_service import TAG_PREFIX_MAP

        # Prepare directories
        uploads_dir = Path("tmp/manifests") / str(manifest_id) / "uploads"
        crops_dir = Path("tmp/manifests") / str(manifest_id) / "crops"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Determine max tag numbers per type across ALL existing assets ---
        existing_result = await self.session.execute(
            select(Asset).where(Asset.manifest_id == manifest_id)
        )
        existing_assets = list(existing_result.scalars().all())

        # Build per-type counters from existing manifest_tags
        # e.g. CHAR_03 → CHARACTER counter = 3
        type_counters: dict[str, int] = {}
        for a in existing_assets:
            tag = a.manifest_tag or ""
            prefix = TAG_PREFIX_MAP.get(a.asset_type, "OTHER")
            if tag.startswith(prefix + "_"):
                try:
                    num = int(tag[len(prefix) + 1:])
                    type_counters[a.asset_type] = max(
                        type_counters.get(a.asset_type, 0), num
                    )
                except ValueError:
                    pass

        def _next_tag(asset_type: str) -> str:
            """Generate next manifest_tag for the given asset_type."""
            count = type_counters.get(asset_type, 0) + 1
            type_counters[asset_type] = count
            prefix = TAG_PREFIX_MAP.get(asset_type, "OTHER")
            return f"{prefix}_{count:02d}"

        # --- Step 2: Save uploaded images and create Asset rows ---
        new_upload_assets: list[Asset] = []

        for upload in new_uploads:
            # Decode base64 image data
            try:
                image_bytes = base64.b64decode(upload.image_data)
            except Exception as e:
                logger.warning(f"Failed to decode base64 for upload {upload.name}: {e}")
                continue

            # Determine file extension from image data header
            if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                ext = "png"
            elif image_bytes[:2] == b"\xff\xd8":
                ext = "jpg"
            elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
                ext = "webp"
            else:
                ext = "jpg"  # fallback

            # Generate tag for this asset type
            manifest_tag = _next_tag(upload.asset_type)

            # Create Asset row
            new_asset = Asset(
                manifest_id=manifest_id,
                asset_type=upload.asset_type,
                name=upload.name,
                manifest_tag=manifest_tag,
                user_tags=upload.tags,
                description=upload.description,
                source="uploaded",
                sort_order=len(existing_assets) + len(new_upload_assets),
                is_inherited=False,
            )
            self.session.add(new_asset)
            await self.session.flush()  # Get new_asset.id

            # Save image to disk with asset ID prefix
            img_filename = f"{new_asset.id}_{upload.name}.{ext}"
            img_path = uploads_dir / img_filename

            await asyncio.to_thread(img_path.write_bytes, image_bytes)

            # Set reference_image_url
            new_asset.reference_image_url = f"/api/assets/{new_asset.id}/image"

            new_upload_assets.append(new_asset)

        await self.session.flush()

        # --- Step 3: YOLO detection on new uploads ---
        new_extracted_crops: list[Asset] = []

        for asset in new_upload_assets:
            pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
            matches = glob(pattern)
            if not matches:
                logger.warning(f"Image not found for new upload asset {asset.id}, skipping YOLO")
                continue

            img_path_str = matches[0]

            try:
                detections = await asyncio.to_thread(
                    self.cv_detector.detect_objects_and_faces, img_path_str
                )
            except Exception as e:
                logger.warning(f"YOLO detection failed for asset {asset.id}: {e}")
                continue

            all_detections = [
                {**d, "is_face": False} for d in detections.get("objects", [])
            ] + [
                {**d, "is_face": True} for d in detections.get("faces", [])
            ]

            for det in all_detections:
                crop_filename = f"{uuid.uuid4()}.jpg"
                crop_path = crops_dir / crop_filename

                try:
                    await asyncio.to_thread(
                        self._save_crop, img_path_str, det["bbox"], str(crop_path)
                    )
                except Exception as e:
                    logger.warning(f"Failed to save crop for asset {asset.id}: {e}")
                    continue

                # Generate tag for extracted crop
                detected_type = _yolo_class_to_asset_type(det["class"])
                crop_tag = _next_tag(detected_type)

                crop_asset = Asset(
                    manifest_id=manifest_id,
                    asset_type=detected_type,
                    name=f"{asset.name} - {det['class']}",
                    manifest_tag=crop_tag,
                    reference_image_url=f"/api/assets/{{id}}/image",  # placeholder
                    source="extracted",
                    source_asset_id=asset.id,
                    detection_class=det["class"],
                    detection_confidence=det["confidence"],
                    crop_bbox=det["bbox"],
                    is_face_crop=det.get("is_face", False),
                    sort_order=asset.sort_order,
                    is_inherited=False,
                )
                self.session.add(crop_asset)
                await self.session.flush()  # Get crop_asset.id

                # Move crop to final location with asset ID prefix
                final_crop_path = crops_dir / f"{crop_asset.id}_{crop_filename}"
                await asyncio.to_thread(Path(str(crop_path)).rename, final_crop_path)

                # Update reference_image_url with actual asset ID
                crop_asset.reference_image_url = f"/api/assets/{crop_asset.id}/image"

                new_extracted_crops.append(crop_asset)

            # Create scene/environment asset with detected objects masked out
            object_bboxes = [d["bbox"] for d in detections.get("objects", [])]
            img_for_size = Image.open(img_path_str)
            w, h = img_for_size.size
            img_for_size.close()

            mask_coverage = self._bbox_mask_coverage(w, h, object_bboxes)
            if mask_coverage <= 0.40:
                scene_crop_filename = f"{uuid.uuid4()}.jpg"
                scene_crop_path = crops_dir / scene_crop_filename

                try:
                    await asyncio.to_thread(
                        self._save_scene_masked, img_path_str, object_bboxes, str(scene_crop_path)
                    )
                except Exception as e:
                    logger.warning(f"Failed to save scene crop for asset {asset.id}: {e}")
                else:
                    scene_tag = _next_tag("ENVIRONMENT")

                    scene_asset = Asset(
                        manifest_id=manifest_id,
                        asset_type="ENVIRONMENT",
                        name=f"{asset.name} - scene",
                        manifest_tag=scene_tag,
                        reference_image_url=f"/api/assets/{{id}}/image",
                        source="extracted",
                        source_asset_id=asset.id,
                        detection_class="scene",
                        detection_confidence=1.0,
                        is_face_crop=False,
                        sort_order=asset.sort_order,
                        is_inherited=False,
                    )
                    self.session.add(scene_asset)
                    await self.session.flush()

                    final_scene_path = crops_dir / f"{scene_asset.id}_{scene_crop_filename}"
                    await asyncio.to_thread(Path(str(scene_crop_path)).rename, final_scene_path)

                    scene_asset.reference_image_url = f"/api/assets/{scene_asset.id}/image"

                    new_extracted_crops.append(scene_asset)
            else:
                logger.info(
                    f"Skipping scene for {asset.name}: "
                    f"{mask_coverage:.0%} of image masked by detections"
                )

        await self.session.flush()

        # --- Step 4: Generate face embeddings for new face crops ---
        new_face_crops = [a for a in new_extracted_crops if a.is_face_crop]

        for face_asset in new_face_crops:
            pattern = f"tmp/manifests/{manifest_id}/crops/{face_asset.id}_*"
            matches = glob(pattern)
            if not matches:
                continue

            crop_path_str = matches[0]
            try:
                embedding = await asyncio.to_thread(
                    self.face_matcher.generate_embedding, crop_path_str
                )
                face_asset.face_embedding = embedding.tobytes()
            except ValueError:
                logger.warning(
                    f"No face detected in new crop {face_asset.id}, reclassifying"
                )
                face_asset.is_face_crop = False

        await self.session.flush()

        # --- Step 5: Cross-match faces (inherited + new) ---
        # Build combined face data for cross-matching using dict format
        # Inherited embeddings passed in as (asset_id, embedding_bytes) tuples
        combined_face_data: list[dict] = []

        # Add inherited face embeddings
        for _, emb_bytes in existing_face_embeddings:
            if emb_bytes:
                emb_array = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                combined_face_data.append({"embedding": emb_array, "source": "inherited"})

        # Add new face crop embeddings with back-references to assets
        new_face_data_refs: list[Asset] = []
        for face_asset in new_face_crops:
            if face_asset.face_embedding and face_asset.is_face_crop:
                emb_array = np.frombuffer(face_asset.face_embedding, dtype=np.float32).copy()
                combined_face_data.append({"embedding": emb_array, "source": "new"})
                new_face_data_refs.append(face_asset)

        if len(combined_face_data) > 1:
            try:
                face_groups = await asyncio.to_thread(
                    self.face_matcher.cross_match_faces,
                    combined_face_data,
                )

                # Count inherited embeddings offset
                inherited_offset = len(existing_face_embeddings)

                # Mark duplicate new face crops (duplicates of inherited OR other new)
                for group in face_groups:
                    if len(group) <= 1:
                        continue

                    # Check if any group member is in the "new" range
                    new_indices = [
                        idx for idx in group if idx >= inherited_offset
                    ]
                    inherited_indices = [
                        idx for idx in group if idx < inherited_offset
                    ]

                    if not new_indices:
                        continue

                    # If there's an inherited face in the group, all new ones are duplicates
                    if inherited_indices:
                        # Find primary: take the first inherited one
                        primary_label = "inherited asset"
                        for new_idx in new_indices:
                            asset_ref_idx = new_idx - inherited_offset
                            if 0 <= asset_ref_idx < len(new_face_data_refs):
                                dup_asset = new_face_data_refs[asset_ref_idx]
                                dup_asset.description = (
                                    f"(Duplicate of {primary_label}) "
                                    + (dup_asset.description or "")
                                )
                    else:
                        # All are new — keep highest confidence, mark others
                        new_assets_in_group = []
                        for new_idx in new_indices:
                            asset_ref_idx = new_idx - inherited_offset
                            if 0 <= asset_ref_idx < len(new_face_data_refs):
                                new_assets_in_group.append(new_face_data_refs[asset_ref_idx])

                        if len(new_assets_in_group) > 1:
                            new_assets_in_group.sort(
                                key=lambda x: x.detection_confidence or 0.0, reverse=True
                            )
                            primary = new_assets_in_group[0]
                            for dup_asset in new_assets_in_group[1:]:
                                dup_asset.description = (
                                    f"(Duplicate of {primary.name}) "
                                    + (dup_asset.description or "")
                                )
            except Exception as e:
                logger.warning(f"Face cross-matching failed for new uploads: {e}")

        await self.session.flush()

        # --- Step 6: Reverse-prompting on all new assets (uploads + crops) ---
        all_new_assets = new_upload_assets + new_extracted_crops
        semaphore = asyncio.Semaphore(5)

        async def _reverse_prompt_new(asset: Asset):
            async with semaphore:
                if asset.source == "uploaded":
                    pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
                else:
                    pattern = f"tmp/manifests/{manifest_id}/crops/{asset.id}_*"

                matches = glob(pattern)
                if not matches:
                    logger.warning(f"Image not found for new asset {asset.id}, skipping reverse-prompt")
                    return

                img_path_str = matches[0]

                try:
                    result = await self.reverse_prompter.reverse_prompt_asset(
                        img_path_str, asset.asset_type, asset.name
                    )
                    asset.reverse_prompt = result.get("reverse_prompt")
                    asset.visual_description = result.get("visual_description")
                    asset.quality_score = result.get("quality_score")

                    suggested_name = result.get("suggested_name")
                    if suggested_name and (not asset.name or asset.name.startswith("Untitled")):
                        asset.name = suggested_name
                except Exception as e:
                    logger.warning(f"Reverse-prompting failed for asset {asset.id}: {e}")

        await asyncio.gather(*[_reverse_prompt_new(a) for a in all_new_assets])
        await self.session.flush()

        logger.info(
            f"process_new_uploads complete: {len(new_upload_assets)} uploads, "
            f"{len(new_extracted_crops)} crops"
        )

        return all_new_assets

    async def reprocess_asset(self, asset_id: uuid.UUID) -> Asset:
        """Reprocess a single asset (re-run YOLO + reverse-prompting).

        Args:
            asset_id: Asset UUID

        Returns:
            Updated Asset instance
        """
        logger.info(f"Reprocessing asset {asset_id}")

        asset = await self.session.get(Asset, asset_id)
        if not asset:
            raise ValueError(f"Asset {asset_id} not found")

        # Resolve image path
        manifest_id = asset.manifest_id
        if asset.source == "uploaded":
            pattern = f"tmp/manifests/{manifest_id}/uploads/{asset.id}_*"
        else:
            pattern = f"tmp/manifests/{manifest_id}/crops/{asset.id}_*"

        matches = glob(pattern)
        if not matches:
            raise ValueError(f"Image file not found for asset {asset_id}")

        img_path = matches[0]

        # Re-run YOLO detection
        detections = await asyncio.to_thread(
            self.cv_detector.detect_objects_and_faces, img_path
        )

        all_dets = detections.get("objects", []) + detections.get("faces", [])
        if all_dets:
            # Take highest confidence detection
            det = max(all_dets, key=lambda d: d["confidence"])

            # Update detection fields
            asset.detection_class = det["class"]
            asset.detection_confidence = det["confidence"]
            asset.is_face_crop = det.get("is_face", False)
            asset.crop_bbox = det["bbox"]

        # Re-run reverse-prompting
        result = await self.reverse_prompter.reverse_prompt_asset(
            img_path, asset.asset_type, asset.name
        )

        # Update reverse-prompt fields
        asset.reverse_prompt = result.get("reverse_prompt")
        asset.visual_description = result.get("visual_description")
        asset.quality_score = result.get("quality_score")

        await self.session.commit()

        logger.info(f"Asset {asset_id} reprocessed successfully")
        return asset

    def _save_crop(self, source_path: str, bbox: list, output_path: str):
        """Save crop from image (sync for to_thread).

        Args:
            source_path: Path to source image
            bbox: Bounding box [x1, y1, x2, y2]
            output_path: Path to save crop
        """
        img = Image.open(source_path)
        crop = img.crop(bbox)
        if crop.mode == "RGBA":
            crop = crop.convert("RGB")
        crop.save(output_path, format="JPEG", quality=90)

    def _save_scene_masked(
        self, source_path: str, bboxes: list[list[float]], output_path: str
    ):
        """Save full image with detected object bboxes blacked out (sync for to_thread).

        Args:
            source_path: Path to source image
            bboxes: List of [x1, y1, x2, y2] bounding boxes to mask
            output_path: Path to save masked scene image
        """
        img = Image.open(source_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], fill="black")
        img.save(output_path, format="JPEG", quality=90)

    @staticmethod
    def _bbox_mask_coverage(
        width: int, height: int, bboxes: list[list[float]]
    ) -> float:
        """Compute fraction of image area covered by bounding boxes.

        Handles overlapping boxes correctly via a boolean pixel mask.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            bboxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            Coverage ratio in [0.0, 1.0]
        """
        if not bboxes or width <= 0 or height <= 0:
            return 0.0
        mask = np.zeros((height, width), dtype=bool)
        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(width, int(bbox[2]))
            y2 = min(height, int(bbox[3]))
            mask[y1:y2, x1:x2] = True
        return float(mask.sum()) / (width * height)


def bytes_to_np(embedding_bytes: bytes):
    """Convert embedding bytes back to numpy array."""
    import numpy as np
    return np.frombuffer(embedding_bytes, dtype=np.float32)
