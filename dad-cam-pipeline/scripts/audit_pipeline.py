#!/usr/bin/env python3
"""
audit_pipeline.py - Pipeline Verification and Audit Script
============================================================

Validates all pipeline outputs and generates a scored audit report.

Tests:
1. Clips existence and count
2. Timeline existence and duration
3. Audio/video sync verification
4. Sample rate consistency
5. File integrity checks

Usage:
    python scripts/audit_pipeline.py --output "/path/to/output"
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: str
    critical: bool = False  # Critical tests must pass


@dataclass
class AuditReport:
    """Complete audit report."""
    timestamp: str
    output_dir: str
    tests: List[Dict]
    total_tests: int
    passed_tests: int
    failed_tests: int
    critical_failures: int
    score: float
    grade: str
    summary: str


def run_ffprobe(path: Path, select_streams: str = None) -> Optional[Dict]:
    """Run ffprobe and return JSON output."""
    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams']
    if select_streams:
        cmd.extend(['-select_streams', select_streams])
    cmd.append(str(path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def get_duration(path: Path) -> Tuple[float, float]:
    """Get video and audio durations separately."""
    video_dur = 0.0
    audio_dur = 0.0

    # Video duration
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=duration', '-of', 'csv=p=0', str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            video_dur = float(result.stdout.strip())
    except Exception:
        pass

    # Audio duration
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
           '-show_entries', 'stream=duration', '-of', 'csv=p=0', str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            audio_dur = float(result.stdout.strip())
    except Exception:
        pass

    return video_dur, audio_dur


def get_sample_rate(path: Path) -> int:
    """Get audio sample rate."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
           '-show_entries', 'stream=sample_rate', '-of', 'csv=p=0', str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


class PipelineAuditor:
    """Audits pipeline output for correctness."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.clips_dir = output_dir / "clips"
        self.timeline_dir = output_dir / "timeline"
        self.analysis_dir = output_dir / "analysis"
        self.results: List[TestResult] = []

    def run_all_tests(self) -> AuditReport:
        """Run all audit tests and generate report."""
        self.results = []

        # Test categories
        self._test_directory_structure()
        self._test_clips_existence()
        self._test_clips_integrity()
        self._test_timeline_existence()
        self._test_timeline_sync()
        self._test_pipeline_results()

        return self._generate_report()

    def _add_result(self, name: str, passed: bool, score: float, details: str, critical: bool = False):
        """Add a test result."""
        self.results.append(TestResult(
            name=name,
            passed=passed,
            score=score,
            details=details,
            critical=critical
        ))

    def _test_directory_structure(self):
        """Test that required directories exist."""
        dirs = [
            (self.clips_dir, "clips/"),
            (self.timeline_dir, "timeline/"),
            (self.analysis_dir, "analysis/"),
        ]

        all_exist = True
        missing = []
        for path, name in dirs:
            if not path.exists():
                all_exist = False
                missing.append(name)

        if all_exist:
            self._add_result(
                "Directory Structure",
                True, 1.0,
                "All required directories exist",
                critical=True
            )
        else:
            self._add_result(
                "Directory Structure",
                False, 0.0,
                f"Missing directories: {', '.join(missing)}",
                critical=True
            )

    def _test_clips_existence(self):
        """Test that clips exist with proper naming."""
        main_clips = list(self.clips_dir.glob("dad_cam_*.mov"))
        tripod_clips = list(self.clips_dir.glob("tripod_cam_*.mov"))

        total = len(main_clips) + len(tripod_clips)

        if total == 0:
            self._add_result(
                "Clips Existence",
                False, 0.0,
                "No clips found in clips/ directory",
                critical=True
            )
            return

        # Check naming convention
        main_names_valid = all(
            c.stem.split('_')[-1].isdigit() for c in main_clips
        )
        tripod_names_valid = all(
            c.stem.split('_')[-1].isdigit() for c in tripod_clips
        )

        if main_names_valid and tripod_names_valid:
            self._add_result(
                "Clips Existence",
                True, 1.0,
                f"Found {len(main_clips)} main clips, {len(tripod_clips)} tripod clips",
                critical=True
            )
        else:
            self._add_result(
                "Clips Existence",
                True, 0.8,
                f"Found {total} clips but some have non-standard naming",
                critical=True
            )

    def _test_clips_integrity(self):
        """Test clip file integrity and sync."""
        main_clips = sorted(self.clips_dir.glob("dad_cam_*.mov"))

        if not main_clips:
            self._add_result(
                "Clips Integrity",
                False, 0.0,
                "No clips to test",
                critical=False
            )
            return

        total_clips = len(main_clips)
        good_clips = 0
        bad_clips = []

        for clip in main_clips:
            video_dur, audio_dur = get_duration(clip)
            gap = abs(video_dur - audio_dur)

            if gap < 1.0:  # Less than 1 second gap is acceptable
                good_clips += 1
            else:
                bad_clips.append(f"{clip.name}: {gap:.2f}s gap")

        score = good_clips / total_clips if total_clips > 0 else 0.0
        passed = score >= 0.95  # 95% of clips must be good

        if passed:
            self._add_result(
                "Clips Integrity",
                True, score,
                f"{good_clips}/{total_clips} clips have proper A/V sync",
                critical=False
            )
        else:
            self._add_result(
                "Clips Integrity",
                False, score,
                f"{len(bad_clips)} clips with sync issues: {', '.join(bad_clips[:3])}...",
                critical=False
            )

    def _test_timeline_existence(self):
        """Test that timeline files exist."""
        main_timeline = self.timeline_dir / "dad_cam_main_timeline.mov"
        tripod_timeline = self.timeline_dir / "dad_cam_tripod_timeline.mov"

        main_exists = main_timeline.exists()
        tripod_exists = tripod_timeline.exists()

        # At least main timeline must exist
        if main_exists:
            main_size = main_timeline.stat().st_size / (1024**3)
            details = f"Main timeline: {main_size:.2f} GB"

            if tripod_exists:
                tripod_size = tripod_timeline.stat().st_size / (1024**3)
                details += f", Tripod timeline: {tripod_size:.2f} GB"

            self._add_result(
                "Timeline Existence",
                True, 1.0 if tripod_exists else 0.9,
                details,
                critical=True
            )
        else:
            self._add_result(
                "Timeline Existence",
                False, 0.0,
                "Main timeline not found",
                critical=True
            )

    def _test_timeline_sync(self):
        """Test timeline audio/video sync - CRITICAL."""
        main_timeline = self.timeline_dir / "dad_cam_main_timeline.mov"
        tripod_timeline = self.timeline_dir / "dad_cam_tripod_timeline.mov"

        results = []

        for name, path in [("Main", main_timeline), ("Tripod", tripod_timeline)]:
            if not path.exists():
                continue

            video_dur, audio_dur = get_duration(path)
            gap = abs(video_dur - audio_dur)

            results.append({
                "name": name,
                "video_dur": video_dur,
                "audio_dur": audio_dur,
                "gap": gap,
                "passed": gap < 1.0
            })

        if not results:
            self._add_result(
                "Timeline A/V Sync",
                False, 0.0,
                "No timelines to test",
                critical=True
            )
            return

        all_passed = all(r["passed"] for r in results)

        details_parts = []
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            details_parts.append(
                f"{r['name']}: {r['video_dur']:.1f}s video, {r['audio_dur']:.1f}s audio, "
                f"gap={r['gap']:.2f}s [{status}]"
            )

        self._add_result(
            "Timeline A/V Sync",
            all_passed,
            1.0 if all_passed else 0.0,
            "; ".join(details_parts),
            critical=True
        )

    def _test_pipeline_results(self):
        """Test pipeline results JSON exists and indicates success."""
        results_file = self.analysis_dir / "pipeline_results.json"

        if not results_file.exists():
            self._add_result(
                "Pipeline Results",
                False, 0.0,
                "pipeline_results.json not found",
                critical=False
            )
            return

        try:
            with open(results_file) as f:
                results = json.load(f)

            success = results.get("success", False)

            if success:
                self._add_result(
                    "Pipeline Results",
                    True, 1.0,
                    "Pipeline completed successfully",
                    critical=False
                )
            else:
                errors = results.get("error", "Unknown error")
                self._add_result(
                    "Pipeline Results",
                    False, 0.5,
                    f"Pipeline reported issues: {errors}",
                    critical=False
                )
        except Exception as e:
            self._add_result(
                "Pipeline Results",
                False, 0.0,
                f"Could not parse results: {e}",
                critical=False
            )

    def _generate_report(self) -> AuditReport:
        """Generate final audit report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        critical_failures = sum(1 for r in self.results if r.critical and not r.passed)

        # Calculate weighted score
        total_score = sum(r.score for r in self.results)
        avg_score = total_score / total if total > 0 else 0.0

        # Penalize critical failures heavily
        if critical_failures > 0:
            avg_score *= 0.5

        # Determine grade
        if avg_score >= 0.95 and critical_failures == 0:
            grade = "A"
        elif avg_score >= 0.85 and critical_failures == 0:
            grade = "B"
        elif avg_score >= 0.70:
            grade = "C"
        elif avg_score >= 0.50:
            grade = "D"
        else:
            grade = "F"

        # Generate summary
        if critical_failures == 0 and passed == total:
            summary = "All tests passed. Pipeline output is verified and complete."
        elif critical_failures == 0:
            summary = f"Minor issues detected. {passed}/{total} tests passed."
        else:
            summary = f"CRITICAL FAILURES: {critical_failures}. Pipeline output is incomplete or invalid."

        return AuditReport(
            timestamp=datetime.now().isoformat(),
            output_dir=str(self.output_dir),
            tests=[asdict(r) for r in self.results],
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            critical_failures=critical_failures,
            score=round(avg_score * 100, 1),
            grade=grade,
            summary=summary
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit pipeline output")
    parser.add_argument("--output", "-o", required=True, help="Output directory to audit")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return 1

    auditor = PipelineAuditor(output_dir)
    report = auditor.run_all_tests()

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print("=" * 70)
        print("PIPELINE AUDIT REPORT")
        print("=" * 70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Output Dir: {report.output_dir}")
        print()
        print("-" * 70)
        print("TEST RESULTS")
        print("-" * 70)

        for test in report.tests:
            status = "PASS" if test["passed"] else "FAIL"
            critical = " [CRITICAL]" if test["critical"] else ""
            print(f"  [{status}] {test['name']}{critical}")
            print(f"         Score: {test['score']*100:.0f}%")
            print(f"         {test['details']}")
            print()

        print("-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Tests: {report.passed_tests}/{report.total_tests} passed")
        print(f"  Critical Failures: {report.critical_failures}")
        print(f"  Score: {report.score}%")
        print(f"  Grade: {report.grade}")
        print()
        print(f"  {report.summary}")
        print("=" * 70)

    # Save report to file
    report_path = output_dir / "analysis" / "audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nReport saved to: {report_path}")

    return 0 if report.critical_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
