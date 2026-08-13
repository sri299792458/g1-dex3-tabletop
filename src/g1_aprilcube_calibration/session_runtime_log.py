"""Compact append-only runtime evidence for long hardware sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from g1_aprilcube_calibration.models import utc_now_iso


class SessionRuntimeLog:
    """Write low-volume JSONL events without making logging a control dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_warning_emitted = False

    def append(self, event: str, **details: Any) -> None:
        if not event.strip():
            raise ValueError("runtime event must be non-empty")
        record = {"utc": utc_now_iso(), "event": event, **details}
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError as error:
            # Diagnostic logging must never interrupt the 250 Hz command owner.
            if not self._write_warning_emitted:
                print(f"warning: runtime log is unavailable: {error}")
                self._write_warning_emitted = True
