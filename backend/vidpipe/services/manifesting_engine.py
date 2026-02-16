"""ManifestingEngine orchestrator for complete CV/AI processing pipeline.

This module orchestrates the full manifesting pipeline: contact sheet assembly,
YOLO detection, face cross-matching, reverse-prompting, and tag assignment.
"""

import asyncio
import logging
import math
import uuid
from glob import glob
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidpipe.db.models import Asset, Manifest
from vidpipe.services.cv_detection import CVDetectionService
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

        # Get all uploaded assets (source="uploaded" with reference_image_url)
        result = await self.session.execute(
            select(Asset).where(
                Asset.manifest_id == manifest_id,
                Asset.source == "uploaded",
                Asset.reference_image_url.isnot(None),
            ).order_by(Asset.sort_order, Asset.created_at)
        )
        uploaded_assets = list(result.scalars().all())

        self.progress["progress"]["uploads_total"] = len(uploaded_assets)

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

        extracted_crops = []
        crops_dir = Path("tmp/manifests") / str(manifest_id) / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        for asset in uploaded_assets:
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

            # Create asset for each detection
            for det in detections:
                crop_filename = f"{uuid.uuid4()}.jpg"
                crop_path = crops_dir / crop_filename

                # Save crop to disk
                await asyncio.to_thread(
                    self._save_crop, img_path, det["bbox"], str(crop_path)
                )

                # Create new Asset record
                new_asset = Asset(
                    manifest_id=manifest_id,
                    asset_type=asset.asset_type,  # Inherit from parent
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

            self.progress["progress"]["uploads_processed"] += 1

        self.progress["progress"]["crops_total"] = len(extracted_crops)
        await self.session.flush()

        # Step 3: Face matching and cross-matching
        self.progress["current_step"] = "face_matching"
        logger.info("Step 3: Running face cross-matching")

        # Get all face crops
        face_crops = [a for a in extracted_crops if a.is_face_crop]

        if face_crops:
            # Generate embeddings
            for face_asset in face_crops:
                pattern = f"tmp/manifests/{manifest_id}/crops/{face_asset.id}_*"
                matches = glob(pattern)
                if not matches:
                    continue

                crop_path = matches[0]
                embedding = await asyncio.to_thread(
                    self.face_matcher.generate_embedding, crop_path
                )
                face_asset.face_embedding = embedding.tobytes()

            await self.session.flush()

            # Cross-match faces
            embeddings = [
                (a.id, a.face_embedding) for a in face_crops if a.face_embedding
            ]

            if len(embeddings) > 1:
                face_groups = await asyncio.to_thread(
                    self.face_matcher.cross_match_faces,
                    [(aid, bytes_to_np(emb)) for aid, emb in embeddings],
                )

                # Mark duplicates (keep highest confidence in each group)
                for group in face_groups:
                    if len(group) > 1:
                        # Find highest confidence
                        group_assets = [
                            a for a in face_crops if a.id in [uuid.UUID(g) for g in group]
                        ]
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

        # Step 4: Reverse-prompting (ALL assets: uploaded + extracted)
        self.progress["current_step"] = "reverse_prompting"
        logger.info("Step 4: Running reverse-prompting")

        all_assets = uploaded_assets + extracted_crops
        semaphore = asyncio.Semaphore(5)  # Rate limiting: 5 concurrent

        async def process_asset_reverse_prompt(asset: Asset):
            async with semaphore:
                # Resolve image path
                if asset.source == "uploaded":
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

        # Process all assets concurrently with rate limiting
        await asyncio.gather(*[process_asset_reverse_prompt(a) for a in all_assets])
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

        if detections:
            # Take first detection (or highest confidence)
            det = max(detections, key=lambda d: d["confidence"])

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
        crop.save(output_path, format="JPEG", quality=90)


def bytes_to_np(embedding_bytes: bytes):
    """Convert embedding bytes back to numpy array."""
    import numpy as np
    return np.frombuffer(embedding_bytes, dtype=np.float32)
