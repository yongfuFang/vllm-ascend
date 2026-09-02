#!/usr/bin/env python3
"""Run each unit-test file in a separate pytest process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TEST_FILES = [
    # fail, need deepseek-ai/DeepSeek-V2 config.json, wulan has not and sync failed
    "tests/ut/attention/a2/test_mla_precision.py",
    "tests/ut/attention/a2/test_sfa_v1_precision.py",  # need deepseek-ai/DeepSeek-V3.2-Exp config.json
    "tests/ut/patch/platform/test_patch_media_connector.py",  # need export VLLM_VERSION=0.26.0
    "tests/ut/kv_offload/a2/test_remote_decode_lifecycle.py",
]


def read_junit_counts(report_file: Path) -> dict[str, int]:
    root = ET.parse(report_file).getroot()
    suites = [root] if root.tag.rsplit("}", 1)[-1] == "testsuite" else list(root)

    counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        if suite.tag.rsplit("}", 1)[-1] != "testsuite":
            continue
        total = int(suite.attrib.get("tests", 0))
        failed = int(suite.attrib.get("failures", 0))
        errors = int(suite.attrib.get("errors", 0))
        skipped = int(suite.attrib.get("skipped", 0))
        counts["total"] += total
        counts["failed"] += failed
        counts["errors"] += errors
        counts["skipped"] += skipped
        counts["passed"] += total - failed - errors - skipped
    return counts


def normalize_junit_timestamp(timestamp: str) -> str:
    """Return a timestamp accepted by legacy JUnit report consumers."""
    iso_timestamp = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed_timestamp = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return timestamp
    return parsed_timestamp.replace(tzinfo=None).isoformat(timespec="microseconds")


def merge_junit_reports(report_files: list[Path], output_file: Path) -> None:
    merged_root = ET.Element("testsuites", name="pytest tests")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    total_time = 0.0

    for report_file in report_files:
        root = ET.parse(report_file).getroot()
        suites = [root] if root.tag.rsplit("}", 1)[-1] == "testsuite" else list(root)
        for suite in suites:
            if suite.tag.rsplit("}", 1)[-1] != "testsuite":
                continue
            timestamp = suite.attrib.get("timestamp")
            if timestamp:
                suite.set("timestamp", normalize_junit_timestamp(timestamp))
            merged_root.append(suite)
            for name in totals:
                totals[name] += int(suite.attrib.get(name, 0))
            total_time += float(suite.attrib.get("time", 0))

    merged_root.attrib.update({name: str(value) for name, value in totals.items()})
    merged_root.set("time", str(total_time))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(output_file, encoding="utf-8", xml_declaration=True)


def collect_test_files(paths: list[Path]) -> list[Path]:
    test_files: set[Path] = set()
    excluded_test_files = {(PROJECT_ROOT / path).resolve() for path in EXCLUDED_TEST_FILES}
    for path in paths:
        if path.is_file():
            if path.name.startswith("test_") and path.suffix == ".py":
                test_files.add(path)
            continue
        if path.is_dir():
            test_files.update(path.rglob("test_*.py"))
            continue
        raise FileNotFoundError(f"Test path does not exist: {path}")
    return sorted(test_file for test_file in test_files if test_file.resolve() not in excluded_test_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Collect and combine Python coverage for vllm_ascend.",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=PROJECT_ROOT / "tests/outputs/ut-coverage",
        help="Coverage output directory (default: tests/outputs/ut-coverage).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "tests/ut"],
        help="Test files or directories to scan (default: tests/ut).",
    )
    args = parser.parse_args()

    try:
        test_files = collect_test_files(args.paths)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if not test_files:
        print("No test_*.py files found.", file=sys.stderr)
        return 1

    coverage_env: dict[str, str] | None = None
    coverage_dir = args.coverage_dir.resolve()
    coverage_dir.mkdir(parents=True, exist_ok=True)
    results_file = coverage_dir / "results.xml"
    coverage_rc = PROJECT_ROOT / "tests/coveragerc"
    coverage_ok = True
    if args.coverage:
        version_result = subprocess.run(
            [sys.executable, "-m", "coverage", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if version_result.returncode != 0:
            print(
                "Coverage.py is not installed. Run: python -m pip install coverage",
                file=sys.stderr,
            )
            return 1

        coverage_env = os.environ.copy()
        coverage_env["COVERAGE_FILE"] = str(coverage_dir / ".coverage")
        erase_result = subprocess.run(
            [sys.executable, "-m", "coverage", "erase", f"--rcfile={coverage_rc}"],
            cwd=PROJECT_ROOT,
            env=coverage_env,
            check=False,
        )
        if erase_result.returncode != 0:
            print("Failed to clear previous coverage data.", file=sys.stderr)
            return 1

    failures: list[tuple[Path, int]] = []
    totals = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    missing_reports = 0
    report_files: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="vllm-ascend-ut-") as report_dir:
        for index, test_file in enumerate(test_files, start=1):
            print(f"\n=== [{index}/{len(test_files)}] {test_file} ===", flush=True)
            report_file = Path(report_dir) / f"pytest-{index}.xml"
            pytest_args = ["-sv", str(test_file), f"--junitxml={report_file}"]
            if args.coverage:
                command = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    f"--rcfile={coverage_rc}",
                    "--source=vllm_ascend",
                    "-m",
                    "pytest",
                    *pytest_args,
                ]
            else:
                command = [sys.executable, "-m", "pytest", *pytest_args]

            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=coverage_env,
                check=False,
            )

            if report_file.exists():
                report_files.append(report_file)
                counts = read_junit_counts(report_file)
                for name in totals:
                    totals[name] += counts[name]
            else:
                missing_reports += 1

            if result.returncode != 0:
                print(f"Failed: {test_file}", file=sys.stderr)
                failures.append((test_file, result.returncode))

        merge_junit_reports(report_files, results_file)

    print("\n=== Test case summary ===")
    print(f"Total:   {totals['total']}")
    print(f"Passed:  {totals['passed']}")
    print(f"Failed:  {totals['failed']}")
    print(f"Errors:  {totals['errors']}")
    print(f"Skipped: {totals['skipped']}")
    print(f"JUnit report: {results_file}")
    if missing_reports:
        print(f"Warning: {missing_reports} test file(s) produced no JUnit report.", file=sys.stderr)

    if args.coverage:
        print("\n=== Coverage summary ===", flush=True)
        combine_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--rcfile={coverage_rc}",
                str(coverage_dir),
            ],
            cwd=PROJECT_ROOT,
            env=coverage_env,
            check=False,
        )
        coverage_ok = combine_result.returncode == 0

        if coverage_ok:
            coverage_commands = [
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "report",
                    f"--rcfile={coverage_rc}",
                    "--show-missing",
                ],
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "html",
                    f"--rcfile={coverage_rc}",
                    "-d",
                    str(coverage_dir / "html"),
                ],
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "xml",
                    f"--rcfile={coverage_rc}",
                    "-o",
                    str(coverage_dir / "coverage.xml"),
                ],
            ]
            for command in coverage_commands:
                command_result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=coverage_env,
                    check=False,
                )
                if command_result.returncode != 0:
                    coverage_ok = False

        if coverage_ok:
            print(f"Coverage data: {coverage_dir / '.coverage'}")
            print(f"HTML report:  {coverage_dir / 'html/index.html'}")
            print(f"XML report:   {coverage_dir / 'coverage.xml'}")
        else:
            print("Failed to combine or generate coverage reports.", file=sys.stderr)

    if failures:
        print(f"\n=== Failed test files ({len(failures)}) ===", file=sys.stderr)
        for test_file, returncode in failures:
            print(f"{test_file} (exit code: {returncode})", file=sys.stderr)
        return 1

    if not coverage_ok:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
