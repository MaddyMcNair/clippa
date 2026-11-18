"""
tests/runner.py - CLIPPA Test Suite Runner & Orchestrator

Usage:
    python tests/runner.py --tier 1,2,3
    python tests/runner.py --tier 4 --tier4-live
    python tests/runner.py --dry-run
    python tests/runner.py --all -v
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"

TIER_MAP = {
    "1": TESTS_DIR / "tier1_features",
    "2": TESTS_DIR / "tier2_boundaries",
    "3": TESTS_DIR / "tier3_combinations",
    "4": TESTS_DIR / "tier4_real_world",
}


class EnvironmentChecker:
    @staticmethod
    def check_all() -> dict:
        env_status = {
            "python": sys.version.split()[0],
            "ffmpeg": None,
            "ffprobe": None,
            "cuda_available": False,
            "gpu_name": None,
            "ollama_online": False,
            "errors": [],
            "warnings": [],
        }

        # Check Python
        if sys.version_info < (3, 10):
            env_status["errors"].append("Python 3.10+ required")

        # Check FFmpeg & ffprobe
        ffmpeg_bin = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
        ffprobe_bin = shutil.which("ffprobe") or os.environ.get("FFPROBE_PATH")

        # Check common Windows locations if not on PATH
        common_ffmpeg_dirs = [
            Path(r"C:\Program Files\FFmpeg\bin"),
            Path(r"C:\Program Files (x86)\FFmpeg\bin"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        ]
        for d in common_ffmpeg_dirs:
            if not ffmpeg_bin and (d / "ffmpeg.exe").exists():
                ffmpeg_bin = str(d / "ffmpeg.exe")
            if not ffprobe_bin and (d / "ffprobe.exe").exists():
                ffprobe_bin = str(d / "ffprobe.exe")

        if ffmpeg_bin:
            try:
                res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
                env_status["ffmpeg"] = res.stdout.splitlines()[0] if res.stdout else "Available"
            except Exception as e:
                env_status["ffmpeg"] = f"Error: {e}"
        else:
            env_status["warnings"].append("ffmpeg not found on PATH. Install via 'winget install Gyan.FFmpeg'")

        if ffprobe_bin:
            try:
                res = subprocess.run([ffprobe_bin, "-version"], capture_output=True, text=True)
                env_status["ffprobe"] = res.stdout.splitlines()[0] if res.stdout else "Available"
            except Exception as e:
                env_status["ffprobe"] = f"Error: {e}"
        else:
            env_status["warnings"].append("ffprobe not found on PATH. Install via 'winget install Gyan.FFmpeg'")

        # Check PyTorch / CUDA
        try:
            import torch
            env_status["cuda_available"] = torch.cuda.is_available()
            if env_status["cuda_available"]:
                env_status["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            env_status["cuda_available"] = False

        # Check Ollama
        try:
            import urllib.request
            req = urllib.request.urlopen("http://localhost:11434/api/version", timeout=1.0)
            if req.status == 200:
                env_status["ollama_online"] = True
        except Exception:
            env_status["ollama_online"] = False

        return env_status


def parse_args():
    parser = argparse.ArgumentParser(description="CLIPPA E2E Test Suite Runner")
    parser.add_argument("-t", "--tier", default="1,2,3", help="Tiers to execute: '1', '2', '3', '4', '1,2', or 'all' [default: 1,2,3]")
    parser.add_argument("--all", action="store_true", help="Execute all test tiers (equivalent to --tier all)")
    parser.add_argument("--tier4-live", action="store_true", help="Enable live YouTube network ingestion in Tier 4")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet pytest output")
    parser.add_argument("-k", "--keyword", help="Filter tests by keyword expression")
    parser.add_argument("-m", "--marker", help="Filter tests by marker expression")
    parser.add_argument("-x", "--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true", help="List discovered tests without executing")
    parser.add_argument("--report-json", default=str(ROOT_DIR / "test_report.json"), help="Output JSON report path")
    parser.add_argument("--report-md", default=str(ROOT_DIR / "test_report.md"), help="Output Markdown report path")
    parser.add_argument("--keep-artifacts", action="store_true", help="Retain generated media fixtures")
    return parser.parse_args()


def run_tests():
    args = parse_args()
    print("=" * 70)
    print("  CLIPPA End-to-End Test Suite Runner")
    print("=" * 70)

    # 1. Preflight environment check
    env = EnvironmentChecker.check_all()
    print(f"[*] Python: {env['python']}")
    print(f"[*] FFmpeg: {env['ffmpeg'] or 'NOT FOUND (Tests requiring FFmpeg will skip)'}")
    print(f"[*] FFprobe: {env['ffprobe'] or 'NOT FOUND (Tests requiring FFprobe will skip)'}")
    print(f"[*] CUDA: {'Available (' + (env.get('gpu_name') or '') + ')' if env['cuda_available'] else 'CPU Mode'}")
    print(f"[*] Ollama: {'Online' if env['ollama_online'] else 'Offline (heuristic virality fallback will test)'}")

    if env["errors"]:
        print("\n[!] Critical Environment Errors:")
        for err in env["errors"]:
            print(f"    - {err}")
        if not args.dry_run:
            sys.exit(3)

    if env["warnings"]:
        print("\n[i] Environment Notices:")
        for w in env["warnings"]:
            print(f"    - {w}")

    # 2. Resolve selected tiers
    selected_tiers = []
    if args.all or args.tier.lower() == "all":
        selected_tiers = ["1", "2", "3", "4"]
    else:
        for t in args.tier.split(","):
            t = t.strip()
            if t in TIER_MAP:
                selected_tiers.append(t)
            else:
                print(f"[ERROR] Invalid tier: '{t}'. Choose from 1, 2, 3, 4, or 'all'.")
                sys.exit(3)

    test_paths = [str(TIER_MAP[t]) for t in selected_tiers if TIER_MAP[t].exists()]
    if not test_paths:
        test_paths = [str(TIER_MAP[t]) for t in selected_tiers]

    print(f"[*] Selected Tiers: {', '.join(selected_tiers)}")
    print(f"[*] Target Directories:\n    " + "\n    ".join(test_paths))

    # 3. Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    pytest_cmd.extend(test_paths)

    if args.verbose:
        pytest_cmd.append("-v")
    elif args.quiet:
        pytest_cmd.append("-q")

    if args.fail_fast:
        pytest_cmd.append("-x")

    if args.keyword:
        pytest_cmd.extend(["-k", args.keyword])

    if args.marker:
        pytest_cmd.extend(["-m", args.marker])

    if args.dry_run:
        pytest_cmd.append("--collect-only")

    # Pass environment variables to pytest child process
    test_env = os.environ.copy()
    test_env["CLIPPA_TIER4_LIVE"] = "1" if args.tier4_live else "0"
    test_env["CLIPPA_KEEP_ARTIFACTS"] = "1" if args.keep_artifacts else "0"
    # Ensure current directory is on PYTHONPATH
    test_env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + test_env.get("PYTHONPATH", "")

    print("\n[*] Executing Pytest Harness...")
    sys.stdout.flush()
    sys.stderr.flush()
    start_time = time.time()
    try:
        proc = subprocess.run(pytest_cmd, env=test_env)
        exit_code = proc.returncode
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\n[FATAL] Failed to launch pytest: {e}")
        sys.exit(3)

    duration = time.time() - start_time
    print(f"\n[*] Execution finished in {duration:.2f}s with exit code {exit_code}")

    # 4. Generate structured report summary
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env,
        "selected_tiers": selected_tiers,
        "duration_seconds": round(duration, 2),
        "exit_code": exit_code,
        "status": "PASS" if exit_code == 0 else ("NO_TESTS" if exit_code == 5 else "FAIL"),
    }

    try:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[*] Saved JSON report to: {args.report_json}")
    except Exception as e:
        print(f"[!] Warning: Could not write JSON report: {e}")

    try:
        md_content = f"""# Test Execution Report
- **Timestamp**: {report['timestamp']}
- **Status**: {report['status']} (Exit Code: {exit_code})
- **Duration**: {report['duration_seconds']}s
- **Selected Tiers**: {', '.join(selected_tiers)}
- **Python**: {env['python']}
- **FFmpeg**: {env['ffmpeg'] or 'Not Found'}
- **CUDA**: {env['cuda_available']}
- **Ollama**: {'Online' if env['ollama_online'] else 'Offline'}
"""
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[*] Saved Markdown report to: {args.report_md}")
    except Exception as e:
        print(f"[!] Warning: Could not write Markdown report: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    run_tests()
