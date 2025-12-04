# Dad Cam Pipeline

Legacy camcorder footage processing pipeline for TOD/MTS files with multicam support.

## Quick Start

```bash
cd dad-cam-pipeline
source .venv/bin/activate
python dad_cam_pipeline.py --source "/path/to/footage" --output "/path/to/output"
```

## Pipeline Phases

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `discover.py` | Scan source folders, build inventory.json |
| 2 | `shot_detect.py` | Detect scene changes for smart cuts |
| 3 | `stability.py` | Analyze camera shake for multicam switching |
| 4 | `transcode.py` | Convert TOD/MTS to H.265 MOV |
| 5 | `audio_process.py` | Process external audio files |
| 6 | `sync_multicam.py` | Cross-correlate audio for sync offsets |
| 7 | `audio_mix.py` | Decide lapel vs camera audio per segment |
| 8 | `multicam_edit.py` | Generate camera switch decisions |
| 9 | `assemble.py` | Create master files and FCPXML |

## Output Structure

```
Output/
├── clips/           # Transcoded H.265 clips
│   ├── dad_cam_001.mov ... dad_cam_NNN.mov (main camera)
│   └── tripod_cam_001.mov ... (tripod camera)
├── master/
│   ├── dad_cam_complete.mov          # Combined master
│   ├── dad_cam_main_camera_docedit.mov    # Main camera only, J/L crossfades
│   └── dad_cam_tripod_camera_docedit.mov  # Tripod camera only, J/L crossfades
├── project/
│   ├── dad_cam_timeline.fcpxml       # FCP timeline
│   └── dad_cam_multicam.fcpxml       # FCP multicam
├── analysis/        # JSON decision files
│   ├── inventory.json
│   ├── audio_mix_decisions.json
│   ├── multicam_edit_decisions.json
│   └── sync_offsets.json
└── logs/            # Pipeline logs
```

## Key Features

### Doc Edits (Per-Camera VHS-Style Output)
- Creates separate master files for each camera source
- Uses 1-second S-curve (esin) audio crossfades at edit points
- Video: stream copy (no re-encode)
- Audio: AAC 256k with smooth J/L transitions

### Decision Files (Generated but NOT YET applied to master)
- `audio_mix_decisions.json` - When to use lapel vs camera audio
- `multicam_edit_decisions.json` - Camera switch points
- `sync_offsets.json` - Audio sync alignment

**KNOWN ISSUE:** The master file (`dad_cam_complete.mov`) does NOT apply these decisions yet. It's a simple concatenation. The doc edits provide working output while full decision application is pending.

## Skip Flags

```bash
--skip-analysis    # Skip phases 1-3
--skip-transcode   # Skip phase 4
--skip-audio       # Skip phase 5
```

## Dependencies

- Python 3.10+
- FFmpeg with libx265
- numpy, scipy (for audio analysis)

## Files Modified in This Session

| File | Changes |
|------|---------|
| `scripts/assemble.py` | Added doc edit feature with J/L crossfades |
| `scripts/transcode.py` | Added -shortest flag to prevent audio/video drift |
| `scripts/fix_audio_sync.py` | NEW: Fix existing clips with audio drift using atrim |
| `scripts/audio_process.py` | Fixed FCPXML encoding (text mode) |
| `dad_cam_pipeline.py` | Added --skip-audio flag |
| `AUDIT_REPORT.md` | Documents 10 critical issues with pipeline |
| `AUDIO_SYNC_FIX.md` | Implementation guide for audio sync fix |
| `FIX_PLAN.md` | Detailed fix plan for decision application |
| `IMPLEMENTATION_GUIDE.md` | Step-by-step guide for developers |

## Critical Issues (See AUDIT_REPORT.md)

1. Audio mix decisions generated but never applied
2. Multicam edit decisions generated but never applied
3. Sync offsets never applied during assembly
4. Master uses simple concat, not intelligent mixing

## Current Work In Progress

**Audio Sync Fix (WIP - NOT COMPLETE)**
- Added `-shortest` flag to transcode.py
- Created fix_audio_sync.py script using atrim filter
- Fixed 71/119 clips (4.7s drift eliminated)
- Doc edits regenerated but **still have issues**
- Need to debug remaining sync/crossfade problems
