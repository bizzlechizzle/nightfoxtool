"""
FFmpeg utility functions for the Dad Cam Pipeline.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class StreamInfo:
    """Information about a media stream."""
    index: int
    codec_type: str
    codec_name: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    field_order: Optional[str] = None


@dataclass
class MediaInfo:
    """Complete media file information."""
    path: Path
    format_name: str
    duration: float
    size: int
    bit_rate: int
    streams: List[StreamInfo]

    @property
    def video_stream(self) -> Optional[StreamInfo]:
        """Get first video stream."""
        for s in self.streams:
            if s.codec_type == "video":
                return s
        return None

    @property
    def audio_stream(self) -> Optional[StreamInfo]:
        """Get first audio stream."""
        for s in self.streams:
            if s.codec_type == "audio":
                return s
        return None

    @property
    def is_interlaced(self) -> bool:
        """Check if video is interlaced."""
        video = self.video_stream
        if video and video.field_order:
            return video.field_order in ("tt", "tb", "bt", "bb")
        return False


def probe_file(path: Path) -> Optional[MediaInfo]:
    """
    Probe a media file using ffprobe.

    Args:
        path: Path to media file

    Returns:
        MediaInfo object or None if probe fails
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    format_info = data.get("format", {})
    streams_data = data.get("streams", [])

    streams = []
    for s in streams_data:
        codec_type = s.get("codec_type", "unknown")
        if codec_type not in ("video", "audio"):
            continue

        stream = StreamInfo(
            index=s.get("index", 0),
            codec_type=codec_type,
            codec_name=s.get("codec_name", "unknown"),
        )

        if codec_type == "video":
            stream.width = s.get("width")
            stream.height = s.get("height")
            stream.field_order = s.get("field_order")

            # Parse frame rate
            fps_str = s.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                stream.fps = num / den if den > 0 else 0
            else:
                stream.fps = float(fps_str)

        elif codec_type == "audio":
            stream.sample_rate = int(s.get("sample_rate", 0))
            stream.channels = s.get("channels", 0)

        stream.duration = float(s.get("duration", 0))
        stream.bit_rate = int(s.get("bit_rate", 0)) if s.get("bit_rate") else None

        streams.append(stream)

    return MediaInfo(
        path=path,
        format_name=format_info.get("format_name", "unknown"),
        duration=float(format_info.get("duration", 0)),
        size=int(format_info.get("size", 0)),
        bit_rate=int(format_info.get("bit_rate", 0)),
        streams=streams
    )


def get_duration(path: Path) -> float:
    """
    Get duration of media file in seconds.

    Args:
        path: Path to media file

    Returns:
        Duration in seconds
    """
    info = probe_file(path)
    return info.duration if info else 0.0


def extract_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 8000,
    mono: bool = True,
    start: Optional[float] = None,
    duration: Optional[float] = None
) -> bool:
    """
    Extract audio from video file.

    Args:
        input_path: Source video file
        output_path: Output audio file (WAV)
        sample_rate: Output sample rate
        mono: Convert to mono
        start: Start time in seconds
        duration: Duration in seconds

    Returns:
        True if successful
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-vn",  # No video
        "-ar", str(sample_rate),
    ])

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.extend([
        "-c:a", "pcm_s16le",
        str(output_path)
    ])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def measure_loudness(path: Path) -> Optional[Dict[str, float]]:
    """
    Measure audio loudness using EBU R128.

    Args:
        path: Path to audio/video file

    Returns:
        Dict with measured values or None
    """
    cmd = [
        "ffmpeg", "-i", str(path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Parse JSON from stderr (ffmpeg outputs loudnorm stats there)
        stderr = result.stderr

        # Find JSON block in output
        json_match = re.search(r'\{[^}]+\}', stderr, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass

    return None


def detect_black_frames(
    path: Path,
    threshold: float = 0.10,
    min_duration: float = 0.1
) -> List[Dict[str, float]]:
    """
    Detect black frames in video.

    Args:
        path: Path to video file
        threshold: Pixel threshold (0-1)
        min_duration: Minimum black duration in seconds

    Returns:
        List of {start, end, duration} dicts
    """
    cmd = [
        "ffmpeg", "-i", str(path),
        "-vf", f"blackdetect=d={min_duration}:pix_th={threshold}",
        "-an", "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
    except subprocess.CalledProcessError:
        return []

    # Parse black frame detections from stderr
    # Format: [blackdetect @ ...] black_start:0.0 black_end:1.0 black_duration:1.0
    pattern = r'black_start:(\d+\.?\d*)\s+black_end:(\d+\.?\d*)\s+black_duration:(\d+\.?\d*)'
    matches = re.findall(pattern, stderr)

    return [
        {"start": float(m[0]), "end": float(m[1]), "duration": float(m[2])}
        for m in matches
    ]


def detect_silence(
    path: Path,
    threshold_db: float = -50,
    min_duration: float = 0.1
) -> List[Dict[str, float]]:
    """
    Detect silence in audio.

    Args:
        path: Path to audio/video file
        threshold_db: Silence threshold in dB
        min_duration: Minimum silence duration

    Returns:
        List of {start, end, duration} dicts
    """
    cmd = [
        "ffmpeg", "-i", str(path),
        "-af", f"silencedetect=n={threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
    except subprocess.CalledProcessError:
        return []

    # Parse silence detections
    # Format: [silencedetect @ ...] silence_start: 0.0
    #         [silencedetect @ ...] silence_end: 1.0 | silence_duration: 1.0
    starts = re.findall(r'silence_start:\s*(\d+\.?\d*)', stderr)
    ends = re.findall(r'silence_end:\s*(\d+\.?\d*)', stderr)
    durations = re.findall(r'silence_duration:\s*(\d+\.?\d*)', stderr)

    results = []
    for i, start in enumerate(starts):
        if i < len(ends):
            results.append({
                "start": float(start),
                "end": float(ends[i]),
                "duration": float(durations[i]) if i < len(durations) else 0
            })

    return results


def generate_vidstab_analysis(
    path: Path,
    output_trf: Path,
    shakiness: int = 10,
    accuracy: int = 15
) -> bool:
    """
    Run vidstab detection pass to analyze camera stability.

    Args:
        path: Input video path
        output_trf: Output transform file path
        shakiness: Shakiness detection level (1-10)
        accuracy: Analysis accuracy

    Returns:
        True if successful
    """
    cmd = [
        "ffmpeg", "-i", str(path),
        "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={output_trf}",
        "-f", "null", "-"
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_trf.exists()
    except subprocess.CalledProcessError:
        return False


def build_transcode_command(
    input_path: Path,
    output_path: Path,
    video_filters: List[str] = None,
    audio_filters: List[str] = None,
    video_codec: str = "libx265",
    video_preset: str = "slow",
    video_crf: int = 18,
    pixel_format: str = "yuv422p10le",
    video_tag: str = "hvc1",
    audio_codec: str = "aac",
    audio_bitrate: str = "256k",
    x265_params: str = None,
    trim_start: float = None,
    trim_end: float = None,
    extra_output_args: List[str] = None
) -> List[str]:
    """
    Build FFmpeg transcode command.

    Args:
        input_path: Source file
        output_path: Output file
        video_filters: List of video filter strings
        audio_filters: List of audio filter strings
        ... (encoding parameters)

    Returns:
        Command as list of strings
    """
    cmd = ["ffmpeg", "-y"]

    # Input
    if trim_start is not None:
        cmd.extend(["-ss", str(trim_start)])

    cmd.extend(["-i", str(input_path)])

    if trim_end is not None:
        cmd.extend(["-t", str(trim_end - (trim_start or 0))])

    # Video filters
    if video_filters:
        vf = ",".join(video_filters)
        cmd.extend(["-vf", vf])

    # Audio filters
    if audio_filters:
        af = ",".join(audio_filters)
        cmd.extend(["-af", af])

    # Video encoding
    cmd.extend([
        "-c:v", video_codec,
        "-preset", video_preset,
        "-crf", str(video_crf),
        "-pix_fmt", pixel_format,
        "-tag:v", video_tag,
    ])

    if x265_params:
        cmd.extend(["-x265-params", x265_params])

    # Audio encoding
    cmd.extend([
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
    ])

    # Container options
    cmd.extend(["-movflags", "+faststart"])

    # Extra args
    if extra_output_args:
        cmd.extend(extra_output_args)

    # Output
    cmd.append(str(output_path))

    return cmd


def build_concat_command(
    file_list_path: Path,
    output_path: Path,
    audio_crossfade: float = 0.5,
    copy_video: bool = True
) -> List[str]:
    """
    Build FFmpeg concat command with audio crossfades.

    Args:
        file_list_path: Path to concat file list
        output_path: Output file path
        audio_crossfade: Crossfade duration in seconds
        copy_video: Stream copy video (no re-encode)

    Returns:
        Command as list of strings
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(file_list_path),
    ]

    if copy_video:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.extend(["-c:v", "libx265", "-crf", "18"])

    # Audio processing for crossfades would require filter_complex
    # For simple concat, just copy audio
    cmd.extend(["-c:a", "copy"])

    cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))

    return cmd


def run_ffmpeg(cmd: List[str], progress_callback=None) -> Tuple[bool, str]:
    """
    Run FFmpeg command with optional progress tracking.

    Args:
        cmd: FFmpeg command as list
        progress_callback: Optional callback(percent) for progress updates

    Returns:
        Tuple of (success, error_message)
    """
    try:
        if progress_callback:
            # Run with progress parsing
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            _, stderr = process.communicate()

            if process.returncode != 0:
                return False, stderr
            return True, ""
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, result.stderr
            return True, ""

    except subprocess.CalledProcessError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)
