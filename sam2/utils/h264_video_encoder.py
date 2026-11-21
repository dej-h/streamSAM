from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class H264VideoEncoder:
    """Encode BGR frames directly to a browser-playable H.264 MP4."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        width, height = frame_size
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
        ]
        # YUV 4:2:0 needs even dimensions. Keep all source pixels by padding.
        if width % 2 or height % 2:
            command.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
        command.extend(
            ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", str(output_path)]
        )

        # FFmpeg can write diagnostics without blocking the frame pipe.
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except BaseException as error:
            self._stderr.close()
            if isinstance(error, FileNotFoundError):
                raise RuntimeError(
                    "FFmpeg with libx264 is required to write H.264 video"
                ) from error
            raise

        assert self._process.stdin is not None
        self._stdin = self._process.stdin

    def write(self, frame: NDArray[np.uint8]) -> None:
        try:
            self._stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except BrokenPipeError as error:
            self._process.wait()
            raise self._encoding_error() from error

    def release(self) -> None:
        try:
            try:
                self._stdin.close()
            except BrokenPipeError:
                pass
            if self._process.wait() != 0:
                raise self._encoding_error()
        finally:
            self._stderr.close()

    def _encoding_error(self) -> RuntimeError:
        self._stderr.seek(0)
        details = self._stderr.read(4096).decode(errors="replace").strip()
        message = f"FFmpeg H.264 encoder exited with status {self._process.returncode}"
        return RuntimeError(f"{message}: {details}" if details else message)
