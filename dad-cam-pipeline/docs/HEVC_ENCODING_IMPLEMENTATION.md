# HEVC Encoding Implementation Guide

## Problem Statement

Video clips are not playing natively on Mac due to encoding configuration issues.

## Root Cause Analysis

### Issue 1: CRF Not Applied
**Location**: `scripts/transcode.py:380`
```python
if settings.video_crf:  # BUG: 0 evaluates to False
    cmd.extend(["-crf", str(settings.video_crf)])
```

When `video_crf = 0` (ProRes config), this condition is `False`, so CRF is never added.
Without CRF, x265 uses default quality settings resulting in ~300 Mbps output (unplayable).

**Fix**: Use explicit check `if settings.video_crf is not None and settings.video_crf > 0:`

### Issue 2: Incomplete File Validation
**Location**: `scripts/transcode.py:238-247`

The `_check_existing` method only checks if file exists and has duration > 0.
It doesn't validate:
- Correct codec profile
- Reasonable bitrate
- Complete moov atom

**Fix**: Add bitrate sanity check (< 50 Mbps for HEVC)

### Issue 3: Timeout Handling
**Location**: `scripts/transcode.py:413`

1-hour timeout is too short for long clips with medium preset.
Failed clips are left as incomplete files that pass existence checks.

**Fix**:
- Increase timeout to 2 hours
- Delete incomplete files on timeout
- Use "fast" preset for clips > 30 minutes

## Required Code Changes

### Change 1: Fix CRF condition (transcode.py:380)
```python
# BEFORE
if settings.video_crf:
    cmd.extend(["-crf", str(settings.video_crf)])

# AFTER
if settings.video_crf is not None and settings.video_crf > 0:
    cmd.extend(["-crf", str(settings.video_crf)])
```

### Change 2: Add output validation (transcode.py, new method)
```python
def validate_output(output_path: Path) -> Tuple[bool, str]:
    """Validate transcoded file is correct."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=bit_rate:stream=codec_name,profile",
             "-of", "json", str(output_path)],
            capture_output=True, text=True, timeout=30
        )
        import json
        data = json.loads(result.stdout)

        # Check codec
        streams = data.get("streams", [])
        if not streams or streams[0].get("codec_name") != "hevc":
            return False, "Wrong codec"

        # Check profile
        if streams[0].get("profile") != "Main 10":
            return False, f"Wrong profile: {streams[0].get('profile')}"

        # Check bitrate (should be < 50 Mbps for HEVC)
        bitrate = int(data.get("format", {}).get("bit_rate", 0))
        if bitrate > 50_000_000:
            return False, f"Bitrate too high: {bitrate/1_000_000:.1f} Mbps"

        return True, ""
    except Exception as e:
        return False, str(e)
```

### Change 3: Better timeout handling (transcode.py:408-430)
```python
try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200  # 2 hour timeout
    )

    if result.returncode != 0:
        # Clean up failed output
        if job.output_path.exists():
            job.output_path.unlink()
        error_lines = result.stderr.split('\n')
        error_msg = next(
            (line for line in reversed(error_lines) if line.strip()),
            "Unknown error"
        )
        return False, error_msg

    # Validate output
    valid, error = validate_output(job.output_path)
    if not valid:
        job.output_path.unlink()
        return False, f"Validation failed: {error}"

    return True, ""

except subprocess.TimeoutExpired:
    # Clean up incomplete file
    if job.output_path.exists():
        job.output_path.unlink()
    return False, "Transcode timeout (2 hours)"
```

### Change 4: Simplify settings (config/settings.py)
Remove ProRes branching - always use H.265 with fixed settings:
```python
@dataclass
class TranscodeSettings:
    """Video transcoding configuration."""
    video_codec: str = "libx265"
    video_preset: str = "medium"
    video_crf: int = 18
    pixel_format: str = "yuv420p10le"
    video_tag: str = "hvc1"
    video_profile: str = "main10"
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    colorspace: str = "bt709"
    deinterlace: bool = True
    deinterlace_filter: str = "yadif=0:-1:0"
    audio_codec: str = "aac"
    audio_bitrate: str = "256k"
    container: str = "mov"
    movflags: str = "+faststart"
```

## FFmpeg Command Template

The correct FFmpeg command for Mac-compatible HEVC:
```bash
ffmpeg -y -i INPUT \
  -vf "yadif=0:-1:0,setpts=PTS-STARTPTS" \
  -c:v libx265 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p10le \
  -profile:v main10 \
  -tag:v hvc1 \
  -color_primaries bt709 \
  -color_trc bt709 \
  -colorspace bt709 \
  -c:a aac \
  -b:a 256k \
  -movflags +faststart \
  OUTPUT.mov
```

## Validation Criteria

After encoding, each file must pass:
1. `codec_name` = "hevc"
2. `profile` = "Main 10"
3. `bit_rate` < 50,000,000 (50 Mbps)
4. Duration matches source (within 1 second)
5. File plays in QuickTime Player without stuttering

## Testing Procedure

1. Delete all clips in output directory
2. Run transcode on 3 test clips (short, medium, long)
3. Verify each with ffprobe
4. Test playback in QuickTime Player
5. If pass, run full batch
