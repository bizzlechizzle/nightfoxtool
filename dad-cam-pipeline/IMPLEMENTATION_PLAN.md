# Dad Cam Pipeline - Implementation Plan v2

## Overview

Transform legacy camcorder footage (TOD/MTS) into a polished, watchable wedding video with:
- Visually lossless H.265 encoding (CRF 18, 8-bit)
- Intelligent multicam switching (handheld preferred, tripod when shaky)
- Smart audio source selection (speeches from lapel, DJ for dances, camera as bed)
- Seamless J/L audio crossfades (0.25s)
- Shot boundary detection via TransNet V2

## Audio Sources

| File | Duration | Purpose | Priority |
|------|----------|---------|----------|
| `210101_003_Tr2.WAV` | 182 min | Speeches/ceremony lapel | 1 (highest) |
| `230825_0002.wav` | 73 min | DJ feed - first dances/reception | 2 |
| Handheld camera | varies | Ambient/backup | 3 |
| Tripod camera | varies | Ambient/backup | 4 |

## Changes Required

### 1. Fix Encoding Settings (`config/settings.py`)

```python
# BEFORE
pixel_format: str = "yuv420p10le"  # 10-bit

# AFTER
pixel_format: str = "yuv420p"  # 8-bit (matches source)
```

Keep CRF 18 for visually lossless.

### 2. Reduce Parallel Jobs

Default to 2 parallel jobs instead of CPU/2 (was 12).

### 3. Implement J/L Audio Crossfades (`scripts/assemble.py`)

Current: Simple concat with `-c:v copy`
New: FFmpeg `filter_complex` with `acrossfade`

```
Crossfade duration: 0.25 seconds (8 frames @ 29.97fps)
Video: Hard cut (no transition)
Audio: Linear crossfade overlap
```

### 4. Add TransNet V2 Shot Detection

New file: `scripts/detect_shots.py`

- Install TransNet V2 as dependency
- Run on all source clips
- Output: `analysis/shot_boundaries.json`

```json
{
  "clips": [
    {
      "path": "/path/to/clip.mov",
      "shots": [
        {"start": 0.0, "end": 12.5},
        {"start": 12.5, "end": 45.2},
        ...
      ]
    }
  ]
}
```

### 5. Add Stability Analysis (`scripts/analyze.py`)

Use FFmpeg `vidstabdetect` to score each clip's stability.

Output: `analysis/stability.json`

```json
{
  "clips": [
    {
      "path": "/path/to/clip.mov",
      "segments": [
        {"start": 0.0, "end": 30.0, "stability_score": 0.85},
        {"start": 30.0, "end": 45.0, "stability_score": 0.32},
        ...
      ],
      "average_stability": 0.65
    }
  ]
}
```

Stability score: 0.0 (extremely shaky) to 1.0 (tripod-stable)

### 6. Train Stability Threshold

Process:
1. Run stability analysis on all clips
2. Sample frames from high/low stability segments
3. Present to user: "Is this too shaky?"
4. Calibrate threshold based on feedback

Initial threshold: 0.4 (switch to tripod below this)

### 7. Intelligent Audio Source Selection

New file: `scripts/audio_mix.py`

Logic per time segment:
```
1. Identify which sources are active (not silence)
2. Check source type:
   - If lapel/speeches file is active AND has signal → use it (duck cameras to -18dB)
   - If DJ feed is active AND has music → use it (duck cameras to -24dB)
   - Otherwise → use best camera audio (highest SNR)
3. Crossfade 0.25s when switching sources
```

Detection methods:
- **Lapel active**: RMS in speech band (300Hz-3kHz) > threshold
- **DJ active**: RMS in music band (60Hz-8kHz) > threshold, bass presence
- **Silence**: Overall RMS < -50dB

Output: `analysis/audio_mix_decisions.json`

### 8. Multicam Switching Logic

New file: `scripts/multicam_edit.py`

Default: Handheld camera (main)

Switch to tripod when ALL true:
- Handheld stability < 0.4 (too shaky)
- At or near a shot boundary (within 0.5s)
- Tripod has footage at this timecode

Switch back to handheld when ALL true:
- Handheld stability > 0.5 (stable enough)
- At or near a shot boundary
- Minimum 5 seconds since last switch (hysteresis)

Output: `analysis/multicam_edit_decisions.json`

### 9. Update Discovery (`scripts/discover.py`)

Enhance audio file detection:
- Scan `Audio/` folder recursively
- Classify by:
  - Duration (long = ceremony, medium = reception)
  - Filename patterns (Tr1/Tr2 = lapel tracks)
  - Content analysis (speech vs music frequency profile)

### 10. Master Assembly with Mixed Audio

Update `scripts/assemble.py`:

1. Load multicam edit decisions
2. Load audio mix decisions
3. Build FFmpeg filter_complex:
   - Video: Cut between angles per multicam decisions
   - Audio: Mix sources per audio decisions with crossfades
4. Output single master file

## New Dependencies

Add to `requirements.txt`:
```
tensorflow>=2.6.0
transnetv2
Pillow>=8.0
```

## File Structure After Changes

```
scripts/
├── discover.py          # Updated: better audio classification
├── analyze.py           # Updated: add stability analysis
├── detect_shots.py      # NEW: TransNet V2 shot detection
├── transcode.py         # Updated: 8-bit encoding
├── audio_process.py     # Unchanged (per-clip normalization)
├── audio_mix.py         # NEW: intelligent source selection
├── sync_multicam.py     # Updated: sync all sources including audio files
├── multicam_edit.py     # NEW: switching decisions
└── assemble.py          # Updated: J/L crossfades + mixed audio + multicam
```

## Pipeline Phases (Updated)

```
Phase 1: Discovery
  - Scan video sources
  - Scan audio sources (NEW: classify lapel vs DJ)
  - Extract metadata, establish order

Phase 2: Analysis
  - Stuck pixel detection
  - Black frame detection
  - Vignette detection
  - Stability analysis (NEW)

Phase 3: Shot Detection (NEW)
  - Run TransNet V2 on all clips
  - Store shot boundaries

Phase 4: Transcoding
  - Deinterlace
  - Fix stuck pixels
  - Trim black frames
  - Encode H.265 CRF 18 8-bit
  - 2 parallel jobs

Phase 5: Audio Processing
  - Hum removal
  - 2-pass loudness normalization to -14 LUFS

Phase 6: Multicam Sync
  - Cross-correlate all sources
  - Align cameras + external audio

Phase 7: Audio Mix Decisions (NEW)
  - Analyze when each source is active
  - Determine source priority per segment
  - Output mix automation

Phase 8: Multicam Edit Decisions (NEW)
  - Combine stability + shot boundaries
  - Determine angle switches
  - Output edit decision list

Phase 9: Assembly
  - Build master with multicam cuts
  - Mix audio sources with crossfades
  - J/L crossfades at clip boundaries
  - Generate FCPXML for NLE
```

## Execution Order

1. Delete existing clips: `rm -rf Output/clips/*.mov`
2. Keep existing analysis (reuse stuck pixels, black frames, inventory)
3. Update code per this plan
4. Re-run pipeline with `--skip-discovery` flag (use existing inventory)

## Quality Targets

| Parameter | Value |
|-----------|-------|
| Video Codec | H.265/HEVC (libx265) |
| CRF | 18 (visually lossless) |
| Pixel Format | yuv420p (8-bit) |
| Audio | AAC 256kbps |
| Loudness | -14 LUFS (EBU R128) |
| Crossfade | 0.25s audio, hard video cut |
| Stability threshold | 0.4 (calibrate from footage) |
| Switch hysteresis | 5 seconds minimum |

## Risk Mitigation

1. **TransNet V2 GPU memory**: Run on CPU if GPU OOM, slower but works
2. **Audio sync drift**: Check sync at multiple points, warn if >2 frames drift
3. **Stability false positives**: Hysteresis prevents rapid switching
4. **Missing coverage**: If tripod doesn't cover a shaky segment, stay on handheld (something > nothing)

## Success Criteria

- [ ] All clips encoded to H.265 CRF 18 8-bit
- [ ] Master file plays smoothly with no jarring audio cuts
- [ ] Speeches clearly audible from lapel source
- [ ] DJ music clear during first dances
- [ ] Camera switches feel natural (at shot boundaries, not mid-action)
- [ ] Shaky handheld segments use tripod when available
- [ ] FCPXML imports correctly into DaVinci Resolve
