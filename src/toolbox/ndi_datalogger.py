"""GUI-independent NDI tracking data logger."""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from sksurgerynditracker.nditracker import NDITracker


class NDIDataLogger:
    """Own an ``NDITracker`` and write its frames to a CSV file.

    The class contains no Tkinter state. It can therefore be driven by a GUI,
    command-line program, or test. Reading runs on a worker thread because a
    tracker call may block while waiting for hardware.
    """

    def __init__(
        self,
        csv_path: str | Path,
        rom_path: str | Path,
        serial_port: str,
        *,
        use_quaternions: bool = True,
        tracker_type: str = "polaris",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rom_path = Path(rom_path)
        self.serial_port = serial_port
        self.use_quaternions = use_quaternions
        self.tracker_type = tracker_type
        self.tracker: NDITracker | None = None
        self.error: Exception | None = None
        self.rows_written = 0

        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def header(self) -> list[str]:
        if self.use_quaternions:
            pose = ["Tx", "Ty", "Tz", "Q0", "Qx", "Qy", "Qz"]
        else:
            pose = [
                "Tx", "Ty", "Tz", "R00", "R01", "R02", "R10", "R11",
                "R12", "R20", "R21", "R22",
            ]
        return ["Tool ID", "Timestamp", "Frame #", *pose, "Tracking Quality"]

    def start(self, timeout: float = 10.0) -> None:
        """Connect to the tracker and start writing frames."""
        if self.recording:
            raise RuntimeError("NDI logging is already running.")
        if not self.rom_path.is_file():
            raise FileNotFoundError(f"ROM file not found: {self.rom_path}")

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.error = None
        self.rows_written = 0
        self._stop_event.clear()
        self._started_event.clear()
        self._thread = threading.Thread(
            target=self._record, name="ndi-tracker", daemon=True
        )
        self._thread.start()
        if not self._started_event.wait(timeout):
            self.stop()
            raise RuntimeError("Timed out while connecting to the NDI tracker.")
        if self.error is not None:
            error = self.error
            self.stop()
            raise RuntimeError(str(error)) from error

    def stop(self, timeout: float = 10.0) -> None:
        """Stop logging and close the tracker connection."""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("NDI tracker did not stop.")
        self._thread = None

    def _record(self) -> None:
        try:
            settings = {
                "tracker type": self.tracker_type,
                "romfiles": [str(self.rom_path)],
                "serial port": self.serial_port,
            }
            self.tracker = NDITracker(settings)
            self.tracker.use_quaternions = self.use_quaternions
            self.tracker.start_tracking()

            with self.csv_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.header)
                self._started_event.set()
                while not self._stop_event.is_set():
                    frame = self.tracker.get_frame()
                    for row in self.rows_from_frame(frame):
                        writer.writerow(row)
                        self.rows_written += 1
        except Exception as error:  # Stored for the GUI to report on its thread.
            self.error = error
            self._stop_event.set()
            self._started_event.set()
        finally:
            if self.tracker is not None:
                try:
                    self.tracker.stop_tracking()
                except Exception:
                    pass
                try:
                    self.tracker.close()
                except Exception:
                    pass
                self.tracker = None

    def rows_from_frame(self, frame: tuple[Any, ...]) -> list[list[Any]]:
        """Convert the five lists returned by ``NDITracker.get_frame``."""
        if len(frame) < 5:
            raise ValueError("NDI frame must contain five values.")
        handles, timestamps, frame_numbers, tracking, quality = frame[:5]
        rows: list[list[Any]] = []
        for index, handle in enumerate(handles):
            pose = self._pose_values(tracking, index)
            rows.append([
                handle,
                self._timestamp(timestamps[index]),
                frame_numbers[index],
                *pose,
                quality[index],
            ])
        return rows

    def _pose_values(self, tracking: Any, index: int) -> list[float]:
        if self.use_quaternions:
            # NDITracker 0.2.x stores one (1, 7) array for each tool.
            values = tracking[index][0]
            return values.tolist() if hasattr(values, "tolist") else list(values)

        matrix = tracking[index]
        return [
            matrix[0][3], matrix[1][3], matrix[2][3],
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2],
        ]

    @staticmethod
    def _timestamp(value: Any) -> Any:
        """Keep tracker timestamps, using local time only when one is absent."""
        return value if value is not None else datetime.now().astimezone().isoformat(
            timespec="microseconds"
        )

    def __enter__(self) -> "NDIDataLogger":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


__all__ = ["NDIDataLogger"]
