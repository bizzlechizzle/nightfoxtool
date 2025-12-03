"""
General utility functions for the Dad Cam Pipeline.
"""

import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List
import json


def run_command(
    cmd: List[str],
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """
    Run a shell command safely.

    Args:
        cmd: Command and arguments as list
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit
        timeout: Timeout in seconds

    Returns:
        CompletedProcess instance
    """
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        check=check,
        timeout=timeout
    )


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse various timestamp formats from EXIF/MOI metadata.

    Args:
        timestamp_str: Timestamp string in various formats

    Returns:
        datetime object or None if parsing fails
    """
    formats = [
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]

    # Remove timezone suffix if present (e.g., "-04:00")
    timestamp_str = re.sub(r'[+-]\d{2}:\d{2}$', '', timestamp_str.strip())

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    return None


def format_timecode(seconds: float, fps: float = 29.97, drop_frame: bool = True) -> str:
    """
    Format seconds as SMPTE timecode.

    Args:
        seconds: Time in seconds
        fps: Frame rate
        drop_frame: Use drop-frame notation (;) for 29.97fps

    Returns:
        Timecode string (HH:MM:SS:FF or HH:MM:SS;FF)
    """
    total_frames = int(seconds * fps)

    # Drop frame calculation for 29.97fps
    if drop_frame and abs(fps - 29.97) < 0.01:
        # Drop frame timecode skips frames 0 and 1 at the start of each minute
        # except every 10th minute
        d = total_frames // 17982  # 10-minute chunks
        m = total_frames % 17982
        total_frames += 18 * d + 2 * ((m - 2) // 1798)

    frames = total_frames % 30
    total_seconds = total_frames // 30
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60

    separator = ";" if drop_frame and abs(fps - 29.97) < 0.01 else ":"
    return f"{hours:02d}:{mins:02d}:{secs:02d}{separator}{frames:02d}"


def parse_timecode(timecode: str, fps: float = 29.97) -> float:
    """
    Parse SMPTE timecode to seconds.

    Args:
        timecode: Timecode string (HH:MM:SS:FF or HH:MM:SS;FF)
        fps: Frame rate

    Returns:
        Time in seconds
    """
    # Handle both : and ; separators
    parts = re.split(r'[:;]', timecode)
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: {timecode}")

    hours, mins, secs, frames = map(int, parts)
    total_seconds = hours * 3600 + mins * 60 + secs + frames / fps
    return total_seconds


def format_duration(seconds: float) -> str:
    """
    Format duration as human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0 or hours > 0:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def hex_to_int(hex_str: str) -> int:
    """
    Convert hexadecimal string to integer.

    Args:
        hex_str: Hex string (e.g., "00A", "01F")

    Returns:
        Integer value
    """
    return int(hex_str, 16)


def natural_sort_key(path: Path) -> Tuple[str, int, str]:
    """
    Generate sort key for natural sorting of file paths.
    Handles hexadecimal file naming (MOV008, MOV009, MOV00A, etc.)

    Args:
        path: File path

    Returns:
        Tuple for sorting (directory, hex_value, extension)
    """
    stem = path.stem
    # Extract numeric/hex portion from filename (e.g., "MOV00A" -> "00A")
    match = re.search(r'([0-9A-Fa-f]+)$', stem)
    if match:
        hex_part = match.group(1)
        try:
            return (str(path.parent), hex_to_int(hex_part), path.suffix)
        except ValueError:
            pass

    return (str(path.parent), 0, stem)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def ensure_list(value) -> list:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class ProgressBar:
    """Simple progress bar for CLI output."""

    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.width = width
        self.start_time = datetime.now()

    def update(self, increment: int = 1, suffix: str = "") -> None:
        """Update progress bar."""
        self.current += increment
        self._render(suffix)

    def _render(self, suffix: str = "") -> None:
        """Render progress bar to terminal."""
        if self.total == 0:
            pct = 100
        else:
            pct = int(100 * self.current / self.total)

        filled = int(self.width * self.current / self.total) if self.total > 0 else self.width
        bar = "█" * filled + "░" * (self.width - filled)

        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0 and self.current < self.total:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_duration(eta)
        else:
            eta_str = "--"

        line = f"\r{self.prefix} |{bar}| {pct:3d}% ({self.current}/{self.total}) ETA: {eta_str}"
        if suffix:
            line += f" - {suffix}"

        print(line, end="", flush=True)

    def finish(self, message: str = "Done") -> None:
        """Complete the progress bar."""
        self.current = self.total
        self._render()
        print(f" - {message}")
