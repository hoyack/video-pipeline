"""Voice stem mixing utilities for generated voice scripts."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

from vidpipe.config import settings
from vidpipe.services.storage_backend import LocalStorageBackend, get_storage_backend

logger = logging.getLogger(__name__)


class VoiceMixerError(RuntimeError):
    """Raised when voice audio cannot be mixed."""


class VoiceMixer:
    """Build ffmpeg voice stems from rendered line audio."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.storage = get_storage_backend()
        self.base_dir = base_dir or settings.storage.tmp_dir

    async def build_stem(
        self,
        production_id: uuid.UUID,
        voice_script_id: uuid.UUID,
        lines: list[dict],
        *,
        scene_id: uuid.UUID | None = None,
        shot_id: uuid.UUID | None = None,
    ) -> tuple[str, float | None]:
        """Create a single MP3 voice stem from generated line audio.

        Lines are concatenated in script order. Timing offsets can be added
        later without changing the artifact contract.
        """
        ready_lines = [line for line in lines if line.get("audio_path")]
        if not ready_lines:
            raise VoiceMixerError("No generated voice line audio is available")

        output_key = self._output_key(production_id, voice_script_id, scene_id, shot_id)
        output_path = self.base_dir / output_key
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_paths = [await self._ensure_local(line["audio_path"]) for line in ready_lines]
        if len(input_paths) == 1:
            shutil.copyfile(input_paths[0], output_path)
        else:
            concat_file = output_path.with_suffix(".concat.txt")
            try:
                concat_file.write_text(
                    "".join(f"file '{path.resolve()}'\n" for path in input_paths),
                    encoding="utf-8",
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-c",
                        "copy",
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise VoiceMixerError(f"Voice mix failed: {exc}") from exc
            finally:
                if concat_file.exists():
                    concat_file.unlink()

        data = output_path.read_bytes()
        await self.storage.put(output_key, data, "audio/mpeg")
        return output_key, _probe_duration(output_path)

    async def _ensure_local(self, audio_path: str) -> Path:
        path = Path(audio_path)
        if path.is_absolute() and path.exists():
            return path

        if isinstance(self.storage, LocalStorageBackend):
            local_path = self.storage.resolve_local_path(audio_path)
            if local_path.exists():
                return local_path

        local_path = self.base_dir / audio_path
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(await self.storage.get(audio_path))
        return local_path

    @staticmethod
    def _output_key(
        production_id: uuid.UUID,
        voice_script_id: uuid.UUID,
        scene_id: uuid.UUID | None,
        shot_id: uuid.UUID | None,
    ) -> str:
        if shot_id is not None:
            scope = f"shot_{shot_id}"
        elif scene_id is not None:
            scope = f"scene_{scene_id}"
        else:
            scope = "script"
        return f"productions/{production_id}/voice/{voice_script_id}/mix/{scope}.mp3"


def _probe_duration(path: Path) -> float | None:
    """Return media duration using ffprobe when available."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        logger.debug("Could not probe duration for %s", path)
        return None
