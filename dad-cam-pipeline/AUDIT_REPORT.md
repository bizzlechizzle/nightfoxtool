# Dad Cam Pipeline - Critical Audit Report

**Date:** December 4, 2025
**Status:** CRITICAL ISSUES FOUND
**Result:** Master file has audio gaps and blank spots due to multiple pipeline failures

---

## Executive Summary

The current pipeline generates analysis data (audio mix decisions, multicam edit decisions, sync offsets) but **NEVER ACTUALLY USES ANY OF IT** during final assembly. The master file is a simple concatenation of camera clips with their original audio, ignoring all the intelligent processing that was supposed to happen.

---

## Critical Issues (Severity: BLOCKER)

### ISSUE 1: Audio Mix Decisions Are Generated But Never Applied

**Location:** `scripts/assemble.py` - `_create_master_concat()`

**Problem:**
- Phase 7 (`audio_mix.py`) generates `audio_mix_decisions.json` with 9 segments
- This file specifies when to use lapel mic (77.7%) vs camera audio (22.3%)
- **BUT `assemble.py` NEVER reads this file!**
- Master file uses only camera audio from each clip

**Evidence from code:**
```python
# assemble.py line 88 - just calls concat, no audio mix
master_path = self._create_master_concat(clips)
```

**Impact:** External lapel mic audio (higher quality) is completely ignored. User hears camera audio only.

---

### ISSUE 2: Multicam Edit Decisions Are Generated But Never Applied

**Location:** `scripts/assemble.py` - `_create_master_concat()`

**Problem:**
- Phase 8 (`multicam_edit.py`) generates `multicam_edit_decisions.json` with 26 segments, 25 switches
- Specifies when to switch between main camera (94.5%) and tripod camera (5.5%)
- **BUT `assemble.py` NEVER reads this file!**
- Master file uses clips in order from `clips/` directory

**Evidence from code:**
```python
# assemble.py line 110-145 - just gathers clips by filename order
def _gather_clips(self) -> List[ClipInfo]:
    clips = []
    for clip_path in sorted(clips_dir.glob("*.mov")):
        # ... just sorts by order number
```

**Impact:** All 26 multicam edit points are ignored. No camera switching happens.

---

### ISSUE 3: Sync Offsets Are Never Applied During Assembly

**Location:** `scripts/assemble.py`

**Problem:**
- Phase 6 (`sync_multicam.py`) generates `sync_offsets.json`:
  - Main camera: +22.8s offset
  - Tripod camera: -1.1s offset
- **BUT these offsets are never applied when assembling video!**

**Evidence from code:**
```python
# assemble.py - the offsets are passed to _generate_multicam_fcpxml
# but that function just creates a gap placeholder!
def _generate_multicam_fcpxml(self, clips, sync_data):
    # ... builds structure but line 631-635:
    ET.SubElement(spine, 'gap',  # <-- JUST A GAP!
        name="Multicam Edit Point",
        offset="0s",
        duration=f"{total_duration}s"
    )
```

**Impact:** Camera sources are not synchronized. Audio/video sync issues.

---

### ISSUE 4: Multicam FCPXML Is Just an Empty Gap

**Location:** `scripts/assemble.py` lines 631-635

**Problem:**
```python
# This is ALL that gets put in the multicam timeline:
ET.SubElement(spine, 'gap',
    name="Multicam Edit Point",
    offset="0s",
    duration=f"{total_duration}s"
)
```

**Impact:** The "multicam" FCPXML is useless - it's just an empty timeline with a gap.

---

### ISSUE 5: Lapel Mic Has Massive Silent Periods That Audio Mix Doesn't Handle

**Location:** `scripts/audio_mix.py`

**Problem from analysis data:**
```json
// From audio_mix_decisions.json - lapel mic 210101_003_Tr2.WAV:
{"start": 81.508896, "end": 3788.495396, "is_active": false}
// That's 61 MINUTES of SILENCE!
```

**BUT** the audio_mix logic selects `230825_0002.wav` (different lapel) during this time:
```json
{
  "start_time": 82.0,
  "end_time": 3789.0,
  "duration": 3707.0,
  "primary_source": "/Volumes/Jay/Dad Cam/Audio/Zoom F3/230825_0002.wav"
}
```

**Impact:** May result in wrong lapel file being selected if the selection logic has bugs.

---

### ISSUE 6: AAC Decoder Errors Causing Audio Gaps

**Location:** Assembly log output

**Problem:**
```
22:44:14 │ ERROR │ Simple concat failed: c:aac @ 0x121e09340]
Error submitting packet to decoder: Invalid data found when processing input
```

Batches 1 and 6 failed with AAC decoder errors. This means some clips in those batches may have corrupted or truncated audio.

**Root Cause:** Likely the audio processing phase (`audio_process.py`) corrupted some clips' AAC streams during remuxing.

---

### ISSUE 7: acrossfade Filter Chain Failures

**Location:** `scripts/assemble.py` lines 166-214

**Problem:** The acrossfade filter chain for audio crossfades is complex and prone to failure:
```python
for i in range(1, len(clips)):
    filter_parts.append(
        f"{current_audio}{next_audio}acrossfade=d={crossfade_duration}:c1=tri:c2=tri{out_label}"
    )
```

When clips have different audio parameters (sample rate, channels, duration), this chain fails.

**Impact:** Falls back to simple concat which has hard audio cuts and potential sync issues.

---

## Secondary Issues (Severity: MAJOR)

### ISSUE 8: No Video Trimming for Multicam

**Problem:** Even if multicam decisions were applied, there's no code to:
1. Extract portions of tripod clips based on source_start/source_end
2. Apply sync offsets to align clips
3. Create smooth video transitions at switch points

---

### ISSUE 9: External Audio Files Never Mixed Into Master

**Problem:** The pipeline discovers lapel mic WAV files but:
1. Never transcodes them
2. Never syncs them to the video timeline
3. Never mixes them into the master audio track

---

### ISSUE 10: Stability Analysis Uses Default Values

**Location:** `scripts/multicam_edit.py`

**Problem:**
```python
seg_stability = seg.get("stability_score", 0.7)  # Default 0.7
```

If stability analysis fails for any clip, it defaults to 0.7 which is above the switch threshold (0.5), so no switches happen for that clip.

---

## Root Cause Analysis

The pipeline was designed with a "generate decisions, then apply them" architecture, but:

1. **The "apply" step was never implemented** - `assemble.py` just concatenates clips
2. **Each phase saves JSON files but the next phase doesn't use them**
3. **The FCPXML generation is the only consumer of some data, but it just creates placeholders**

---

## Architecture Diagram (Current vs Expected)

### CURRENT (Broken):
```
Phase 7: audio_mix.py → audio_mix_decisions.json → [DEAD END - NEVER USED]
Phase 8: multicam_edit.py → multicam_edit_decisions.json → [DEAD END - NEVER USED]
Phase 9: assemble.py → [JUST CONCATENATES CLIPS IN ORDER]
```

### EXPECTED:
```
Phase 7: audio_mix.py → audio_mix_decisions.json ─┐
Phase 8: multicam_edit.py → multicam_edit_decisions.json ─┼→ Phase 9: assemble.py
Phase 6: sync_multicam.py → sync_offsets.json ────────────┘
                                                           ↓
                                               [Apply all decisions]
                                                           ↓
                                               [Master with mixed audio,
                                                camera switches, sync]
```

---

## Recommended Fix Priority

1. **P0 - CRITICAL:** Make assemble.py read and apply audio_mix_decisions.json
2. **P0 - CRITICAL:** Make assemble.py read and apply multicam_edit_decisions.json
3. **P0 - CRITICAL:** Make assemble.py apply sync_offsets.json
4. **P1 - HIGH:** Fix AAC decoder errors in audio processing
5. **P1 - HIGH:** Implement proper FCPXML multicam structure
6. **P2 - MEDIUM:** Improve audio fallback when lapel is silent
7. **P3 - LOW:** Add video transitions at camera switch points

---

## Files That Need Changes

| File | Changes Needed |
|------|----------------|
| `scripts/assemble.py` | Complete rewrite of assembly logic to use decision files |
| `scripts/audio_mix.py` | Better handling of silent periods, fallback to camera audio |
| `scripts/audio_process.py` | Fix AAC encoding that causes decoder errors |
| `dad_cam_pipeline.py` | Ensure decision files are passed between phases |

---

## Test Plan After Fixes

1. Verify audio_mix_decisions.json is read during assembly
2. Verify multicam_edit_decisions.json is read during assembly
3. Verify sync_offsets.json is applied
4. Verify no AAC decoder errors
5. Verify lapel audio is present in master when it should be
6. Verify camera switches happen at decision points
7. Verify audio crossfades are smooth
8. Verify FCPXML imports correctly into Final Cut Pro

---

## Conclusion

The pipeline's intelligent processing phases (audio mix, multicam edit, sync) all work and produce valid decision files. However, the final assembly phase completely ignores these decisions and just concatenates clips. This is why the master file has:

- **Audio gaps:** No external audio is mixed in
- **Blank spots:** Possible AAC corruption from audio processing
- **Wrong audio:** Camera audio instead of lapel
- **No camera switches:** Multicam decisions ignored

The fix requires a significant rewrite of `assemble.py` to actually implement the "apply decisions" logic that was planned but never coded.
