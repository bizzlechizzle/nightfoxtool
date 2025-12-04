#!/usr/bin/env python3
"""
drift_diagnosis.py - Diagnose audio/video drift in clips and timeline
======================================================================

Checks:
1. Individual clip A/V duration mismatches
2. Sample rate consistency
3. Frame rate consistency
4. Drift at multiple timeline points
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ClipDriftInfo:
    filename: str
    video_duration: float
    audio_duration: float
    drift: float
    sample_rate: int
    frame_rate: str
    has_drift: bool


def probe_clip(path: Path) -> Optional[ClipDriftInfo]:
    """Get detailed A/V info for a clip."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_streams',
        '-show_entries', 'stream=codec_type,duration,sample_rate,r_frame_rate',
        '-of', 'json', str(path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        v_dur = 0.0
        a_dur = 0.0
        sample_rate = 0
        frame_rate = "unknown"

        for stream in data.get('streams', []):
            codec_type = stream.get('codec_type')
            duration = float(stream.get('duration', 0) or 0)

            if codec_type == 'video':
                v_dur = duration
                frame_rate = stream.get('r_frame_rate', 'unknown')
            elif codec_type == 'audio':
                a_dur = duration
                sample_rate = int(stream.get('sample_rate', 0) or 0)

        drift = abs(v_dur - a_dur)

        return ClipDriftInfo(
            filename=path.name,
            video_duration=v_dur,
            audio_duration=a_dur,
            drift=drift,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            has_drift=drift > 0.04  # More than 1 frame at 24fps
        )
    except Exception as e:
        print(f"Error probing {path}: {e}")
        return None


def get_pts_at_time(file: Path, timestamp: str) -> Tuple[float, float, float]:
    """Get video and audio PTS at a specific timestamp."""
    # Get video frame PTS
    v_cmd = [
        'ffprobe', '-v', 'error',
        '-read_intervals', f'%{timestamp}%+#1',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pts_time',
        '-of', 'json', str(file)
    ]

    try:
        v_result = subprocess.run(v_cmd, capture_output=True, text=True, timeout=30)
        v_data = json.loads(v_result.stdout)
        v_pts = float(v_data['frames'][0]['pts_time']) if v_data.get('frames') else 0
    except:
        v_pts = 0

    # Get audio frame PTS
    a_cmd = [
        'ffprobe', '-v', 'error',
        '-read_intervals', f'%{timestamp}%+#1',
        '-select_streams', 'a:0',
        '-show_entries', 'frame=pts_time',
        '-of', 'json', str(file)
    ]

    try:
        a_result = subprocess.run(a_cmd, capture_output=True, text=True, timeout=30)
        a_data = json.loads(a_result.stdout)
        a_pts = float(a_data['frames'][0]['pts_time']) if a_data.get('frames') else 0
    except:
        a_pts = 0

    return v_pts, a_pts, abs(v_pts - a_pts)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose A/V drift")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--save", "-s", help="Save results to file")
    args = parser.parse_args()

    output_dir = Path(args.output)
    clips_dir = output_dir / "clips"
    timeline_dir = output_dir / "timeline"

    results = []

    print("=" * 70)
    print("DRIFT DIAGNOSIS REPORT")
    print("=" * 70)

    # ========================================
    # Phase 1.1: Individual Clip A/V Durations
    # ========================================
    print("\n" + "-" * 70)
    print("1.1 INDIVIDUAL CLIP A/V DURATIONS")
    print("-" * 70)

    clips_with_drift = []
    sample_rates = {}
    frame_rates = {}

    for clip_path in sorted(clips_dir.glob("dad_cam_*.mov")):
        info = probe_clip(clip_path)
        if info:
            results.append(info)

            # Track sample rates
            sample_rates[info.sample_rate] = sample_rates.get(info.sample_rate, 0) + 1

            # Track frame rates
            frame_rates[info.frame_rate] = frame_rates.get(info.frame_rate, 0) + 1

            if info.has_drift:
                clips_with_drift.append(info)
                print(f"DRIFT: {info.filename}")
                print(f"       video={info.video_duration:.3f}s audio={info.audio_duration:.3f}s")
                print(f"       diff={info.drift:.3f}s sample_rate={info.sample_rate}")

    # Also check tripod clips
    for clip_path in sorted(clips_dir.glob("tripod_cam_*.mov")):
        info = probe_clip(clip_path)
        if info:
            results.append(info)
            sample_rates[info.sample_rate] = sample_rates.get(info.sample_rate, 0) + 1
            frame_rates[info.frame_rate] = frame_rates.get(info.frame_rate, 0) + 1
            if info.has_drift:
                clips_with_drift.append(info)
                print(f"DRIFT: {info.filename}")
                print(f"       video={info.video_duration:.3f}s audio={info.audio_duration:.3f}s")
                print(f"       diff={info.drift:.3f}s sample_rate={info.sample_rate}")

    total_clips = len(results)
    drifted_clips = len(clips_with_drift)

    print(f"\nSummary: {drifted_clips}/{total_clips} clips have drift > 0.04s")

    if drifted_clips == 0:
        print("Individual clips appear OK - drift may be from assembly")

    # ========================================
    # Phase 1.2: Sample Rate Consistency
    # ========================================
    print("\n" + "-" * 70)
    print("1.2 SAMPLE RATE DISTRIBUTION")
    print("-" * 70)

    for rate, count in sorted(sample_rates.items()):
        print(f"  {rate} Hz: {count} clips")

    if len(sample_rates) > 1:
        print("\n  WARNING: Multiple sample rates detected!")
        print("  This can cause drift during assembly if not handled correctly.")
    else:
        print("\n  OK: All clips have same sample rate")

    # ========================================
    # Phase 1.3: Frame Rate Consistency
    # ========================================
    print("\n" + "-" * 70)
    print("1.3 FRAME RATE DISTRIBUTION")
    print("-" * 70)

    for rate, count in sorted(frame_rates.items()):
        print(f"  {rate}: {count} clips")

    if len(frame_rates) > 1:
        print("\n  WARNING: Multiple frame rates detected!")
    else:
        print("\n  OK: All clips have same frame rate")

    # ========================================
    # Phase 1.4: Timeline Drift at Multiple Points
    # ========================================
    print("\n" + "-" * 70)
    print("1.4 TIMELINE DRIFT AT MULTIPLE POINTS")
    print("-" * 70)

    main_timeline = timeline_dir / "dad_cam_main_timeline.mov"

    if main_timeline.exists():
        # Get total duration first
        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'csv=p=0', str(main_timeline)]
        dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
        total_dur = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 0

        # Test points (adjust based on actual duration)
        test_points = []
        if total_dur > 60:
            test_points.append("00:01:00")
        if total_dur > 900:
            test_points.append("00:15:00")
        if total_dur > 1800:
            test_points.append("00:30:00")
        if total_dur > 3600:
            test_points.append("01:00:00")
        if total_dur > 5400:
            test_points.append("01:30:00")
        if total_dur > 7200:
            test_points.append("02:00:00")

        print(f"\nTimeline duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
        print(f"\nTesting {len(test_points)} points:")
        print(f"{'Timestamp':<14} | {'Video PTS':>10} | {'Audio PTS':>10} | {'Drift':>8} | Status")
        print("-" * 60)

        timeline_drifts = []
        for ts in test_points:
            v_pts, a_pts, drift = get_pts_at_time(main_timeline, ts)
            status = "OK" if drift < 0.04 else "DRIFT!"
            timeline_drifts.append((ts, v_pts, a_pts, drift))
            print(f"{ts:<14} | {v_pts:>10.3f} | {a_pts:>10.3f} | {drift:>7.3f}s | {status}")

        # Check if drift is progressive (cumulative) or constant
        if len(timeline_drifts) >= 2:
            first_drift = timeline_drifts[0][3]
            last_drift = timeline_drifts[-1][3]
            drift_increase = last_drift - first_drift

            print(f"\nDrift at start: {first_drift:.3f}s")
            print(f"Drift at end:   {last_drift:.3f}s")
            print(f"Drift increase: {drift_increase:.3f}s")

            if drift_increase > 0.5:
                print("\nDIAGNOSIS: CUMULATIVE DRIFT - drift gets worse over time")
                print("          Likely cause: Sample rate mismatch or timestamp issues during concat")
            elif max(d[3] for d in timeline_drifts) > 0.04:
                print("\nDIAGNOSIS: CONSTANT DRIFT - drift is present but stable")
                print("          Likely cause: Initial offset or single problematic clip")
            else:
                print("\nDIAGNOSIS: NO SIGNIFICANT DRIFT DETECTED")
    else:
        print(f"  Timeline not found: {main_timeline}")

    # ========================================
    # Summary and Recommendations
    # ========================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if drifted_clips > 0:
        issues.append(f"- {drifted_clips} clips have individual A/V drift > 0.04s")

    if len(sample_rates) > 1:
        issues.append(f"- Mixed sample rates: {list(sample_rates.keys())}")

    if len(frame_rates) > 1:
        issues.append(f"- Mixed frame rates: {list(frame_rates.keys())}")

    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\nNo obvious issues found in clip analysis.")
        print("Drift may be perceptual or in a different timeline.")

    # Save results if requested
    if args.save:
        save_path = Path(args.save)
        with open(save_path, 'w') as f:
            f.write("DRIFT DIAGNOSIS RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("CLIPS WITH DRIFT (>0.04s):\n")
            for info in clips_with_drift:
                f.write(f"  {info.filename}: video={info.video_duration:.3f}s ")
                f.write(f"audio={info.audio_duration:.3f}s diff={info.drift:.3f}s\n")

            f.write(f"\nSAMPLE RATES:\n")
            for rate, count in sorted(sample_rates.items()):
                f.write(f"  {rate} Hz: {count} clips\n")

            f.write(f"\nFRAME RATES:\n")
            for rate, count in sorted(frame_rates.items()):
                f.write(f"  {rate}: {count} clips\n")

        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
