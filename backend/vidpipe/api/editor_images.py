"""Minimal image upload/serve for the rich text editor."""

import re
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from vidpipe.api.asset_library import _save_upload, _serve_ref_image, _validate_image_upload

editor_images_router = APIRouter(prefix="/api/editor-images", tags=["editor-images"])

_FILENAME_RE = re.compile(r"^[a-f0-9-]+\.(png|jpe?g|webp)$")


@editor_images_router.post("", status_code=201)
async def upload_editor_image(file: UploadFile = File(...)):
    """Upload an image for use in the scene prompt editor."""
    content = await file.read()
    _validate_image_upload(file, content)

    ext = (file.filename or "image.png").rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg", "webp"):
        ext = "png"
    filename = f"{uuid.uuid4()}.{ext}"

    await _save_upload(content, file.content_type or "image/png", filename, "editor-images")
    return {"url": f"/api/editor-images/{filename}"}


@editor_images_router.get("/{filename}")
async def serve_editor_image(filename: str):
    """Serve a previously uploaded editor image."""
    if not _FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    from vidpipe.config import settings
    from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

    storage = get_storage_backend()
    if isinstance(storage, LocalStorageBackend):
        image_path = str(settings.storage.tmp_dir / "asset-library" / "editor-images" / filename)
    else:
        image_path = f"asset-library/editor-images/{filename}"

    return await _serve_ref_image(image_path)
