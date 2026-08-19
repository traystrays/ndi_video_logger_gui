"""Camera discovery, preview, and timestamped video recording."""

from __future__ import annotations

import csv
import math
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

FrameCallback = Callable[[int, np.ndarray], None]



def list_cameras(max_devices: int = 10) -> list[int]:
    """Return camera indices that OpenCV can open."""
    cameras: list[int] = []
    for index in range(max_devices):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                cameras.append(index)
        finally:
            capture.release()
    return cameras


def read_camera_frame(camera_index: int) -> np.ndarray:
    """
    Read video frames, used by a UI to preview a camera."""
    capture = cv2.VideoCapture(camera_index)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}.")
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from camera {camera_index}.")
        return frame
    finally:
        capture.release()


class VideoRecorder:
    """Record one camera and a timestamp for every written frame.

    Recording runs on a worker thread. ``frame_callback`` receives occasional
    frame copies and can be used by a GUI for a live preview.
    """

    def __init__(
        self,
        camera_index: int,
        video_path: str | Path,
        timestamp_path: str | Path,
        *,
        resolution: tuple[int, int] = (1280, 720),
        fps: float = 30.0,
        frame_callback: FrameCallback | None = None,
        preview_every: int = 2,
    ) -> None:
        self.camera_index = camera_index
        self.video_path = Path(video_path)
        self.timestamp_path = Path(timestamp_path)
        self.resolution = resolution
        self.requested_fps = fps
        self.frame_callback = frame_callback
        self.preview_every = max(1, preview_every)
        self.error: Exception | None = None
        self.frame_count = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_event = threading.Event()

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 5.0) -> None:
        """Start recording and wait until the camera/writer is ready."""
        if self.recording:
            raise RuntimeError(f"Camera {self.camera_index} is already recording.")
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.timestamp_path.parent.mkdir(parents=True, exist_ok=True)
        self.error = None
        self.frame_count = 0
        self._stop_event.clear()
        self._started_event.clear()
        self._thread = threading.Thread(
            target=self._record, name=f"camera-{self.camera_index}", daemon=True
        )
        self._thread.start()
        if not self._started_event.wait(timeout):
            self.stop()
            raise RuntimeError(f"Timed out opening camera {self.camera_index}.")
        if self.error is not None:
            error = self.error
            self.stop()
            raise RuntimeError(str(error)) from error

    def stop(self, timeout: float = 5.0) -> None:
        """Request a stop and wait for files and camera handles to close."""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError(f"Camera {self.camera_index} did not stop.")
        self._thread = None

    def _record(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        writer: cv2.VideoWriter | None = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 240:
                fps = self.requested_fps

            writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video file {self.video_path}.")

            self._started_event.set()
            with self.timestamp_path.open("w", newline="") as timestamp_file:
                timestamp_writer = csv.writer(timestamp_file)
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Camera {self.camera_index} stopped returning frames."
                        )
                    # Downstream matching localizes these local wall-clock values.
                    timestamp = datetime.now().isoformat(timespec="microseconds")
                    writer.write(frame)
                    timestamp_writer.writerow([timestamp])
                    if self.frame_callback and self.frame_count % self.preview_every == 0:
                        self.frame_callback(self.camera_index, frame.copy())
                    self.frame_count += 1
        except Exception as error:  # Exposed to the GUI through ``error``.
            self.error = error
            self._stop_event.set()
            self._started_event.set()
        finally:
            capture.release()
            if writer is not None:
                writer.release()


# Backwards-compatible name used by older scripts in this repository.
VideoCapture = VideoRecorder

class CameraDisplay:
    """Show live, labelled previews for several cameras in one window.

    OpenCV window and camera operations all live on the same worker thread.
    Closing the preview window (or pressing Escape/q while it has focus) releases
    every camera and ends that thread.
    """

    def __init__(
        self,
        cameras: list[int],
        *,
        window_name: str = "Detected cameras",
        tile_size: tuple[int, int] = (480, 270),
    ) -> None:
        self.cameras = list(cameras)
        self.window_name = window_name
        self.tile_size = tile_size
        self.error: Exception | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="camera-display", daemon=True
        )

    @property
    def showing(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        if not self.cameras:
            raise ValueError("No cameras were supplied for display.")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Close the display programmatically, such as during app shutdown."""
        self._stop_event.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        captures = {index: cv2.VideoCapture(index) for index in self.cameras}
        try:
            opened = {index: cap for index, cap in captures.items() if cap.isOpened()}
            if not opened:
                raise RuntimeError("None of the detected cameras could be reopened.")

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            while not self._stop_event.is_set():
                tiles = [self._read_tile(index, capture) for index, capture in opened.items()]
                cv2.imshow(self.window_name, self._make_grid(tiles))

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:  # The user closed the window between event checks.
                    break
        except Exception as error:
            self.error = error
        finally:
            for capture in captures.values():
                capture.release()
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

    def _read_tile(self, index: int, capture: cv2.VideoCapture) -> np.ndarray:
        width, height = self.tile_size
        ok, frame = capture.read()
        if ok:
            frame = cv2.resize(frame, (width, height))
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                frame, "No frame", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )
        cv2.rectangle(frame, (0, 0), (180, 42), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame, f"Camera {index}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _make_grid(tiles: list[np.ndarray]) -> np.ndarray:
        columns = math.ceil(math.sqrt(len(tiles)))
        rows = math.ceil(len(tiles) / columns)
        blank = np.zeros_like(tiles[0])
        padded = tiles + [blank] * (rows * columns - len(tiles))
        return cv2.vconcat([
            cv2.hconcat(padded[row * columns:(row + 1) * columns])
            for row in range(rows)
        ])


def display(cameras: list[int]) -> CameraDisplay:
    """Start a live multi-camera display and return its lifecycle handle."""
    camera_display = CameraDisplay(cameras)
    camera_display.start()
    return camera_display
