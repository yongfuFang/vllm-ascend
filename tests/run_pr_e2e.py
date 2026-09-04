#!/usr/bin/env python3
"""Run only the pull-request tests explicitly listed below."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add one pytest file path or node ID per line. Comment out a line to skip it.
SELECTED_TESTS = [
    "tests/e2e/pull_request/one_card/test_attention_fa3.py",  # pass
    "tests/e2e/pull_request/one_card/test_batch_invariant.py",  # pass
    "tests/e2e/pull_request/one_card/test_qwen3_0_6b.py",  # pass
    "tests/e2e/pull_request/one_card/test_multi_instance.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_dflash.py",  # pass
    "tests/e2e/pull_request/one_card/test_batch_job_aware_scheduler_e2e.py",  # pass
    "tests/e2e/pull_request/one_card/test_camem.py",  # pass: unset -v PYTORCH_NPU_ALLOC_CONF
    "tests/e2e/pull_request/one_card/test_completion_with_prompt_embeds.py",  # pass
    "tests/e2e/pull_request/one_card/test_cpu_offloading.py",  # pass
    "tests/e2e/pull_request/one_card/test_cpu_weight_offload.py",  # pass
    "tests/e2e/pull_request/one_card/test_guided_decoding.py",  # pass
    "tests/e2e/pull_request/one_card/test_sampler.py",  # pass
    "tests/e2e/pull_request/one_card/test_simple_cpu_offload.py",  # pass
    "tests/e2e/pull_request/one_card/test_xlite.py",  # pass
    "tests/e2e/pull_request/one_card/model_runner_v2/test_uva.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py",  # pass
    "tests/e2e/pull_request/one_card/compile/test_graphex_norm_quant_fusion.py",  # pass
    "tests/e2e/pull_request/one_card/compile/test_graphex_qknorm_rope_fusion.py",  # pass
    "tests/e2e/pull_request/one_card/compile/test_norm_quant_fusion.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_ngram.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_ngram_npu.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_suffix.py",  # pass
    "tests/e2e/pull_request/one_card/test_qwen3_embedding_0_6b.py",  # pass
    "tests/e2e/pull_request/one_card/spec_decode/test_dspark.py",  # pass
    "tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py",  # pass
    "tests/e2e/pull_request/two_card/test_sequence_parallelism_moe.py",  # pass
    "tests/e2e/pull_request/two_card/test_offline_weight_load.py",  # pass
    "tests/e2e/pull_request/two_card/test_moe_routing_replay.py",  # pass
    "tests/e2e/pull_request/two_card/test_qwen3_moe_eplb.py",  # pass
    "tests/e2e/pull_request/two_card/test_disaggregated_encoder.py",  # pass
    "tests/e2e/pull_request/two_card/test_qwen3_performance.py",  # pass
    # "tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py",  # failed
    # "tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py",  # fail: need download picture
    # "tests/e2e/pull_request/one_card/spec_decode/test_eagle.py",  # fail: RedHatAI/Qwen3-8B-speculator.eagle3 not in model market
    # "tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py", # fail
    # "tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py", # fail

    # two card
    # "tests/e2e/pull_request/two_card/test_xlite.py",  # fail
    # "tests/e2e/pull_request/two_card/test_qwen3_30b_a3b.py",  # fail
    # "tests/e2e/pull_request/two_card/test_external_launcher.py",  # fail
    # "tests/e2e/pull_request/two_card/test_data_parallel.py", # fail
]


def normalize_junit_timestamp(timestamp: str) -> str:
    """Return a timestamp accepted by legacy JUnit report consumers."""
    iso_timestamp = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed_timestamp = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return timestamp
    return parsed_timestamp.replace(tzinfo=None).isoformat(timespec="microseconds")


def merge_junit_reports(report_files: list[Path], output_file: Path) -> None:
    merged_root = ET.Element("testsuites", name="pull-request tests")
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
    ET.ElementTree(merged_root).write(output_file, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Collect coverage and generate coverage.xml.",
    )
    args = parser.parse_args()

    if not SELECTED_TESTS:
        print("No pull-request tests are selected.", file=sys.stderr)
        return 1

    passed_tests: list[str] = []
    failed_tests: list[str] = []
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = PROJECT_ROOT / "tests" / "outputs" / "pull_request"
    log_dir = output_dir / run_timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    junit_dir = log_dir / "junit"
    junit_dir.mkdir(exist_ok=True)
    results_file = output_dir / "results.xml"
    report_files: list[Path] = []
    coverage_file = log_dir / ".coverage"
    coverage_xml = output_dir / "coverage.xml"
    coverage_rc = PROJECT_ROOT / "tests" / "coveragerc"
    coverage_env: dict[str, str] | None = None
    coverage_ok = True

    if args.coverage:
        coverage_env = os.environ.copy()
        coverage_env["COVERAGE_FILE"] = str(coverage_file)
        coverage_version = subprocess.run(
            [sys.executable, "-m", "coverage", "--version"],
            cwd=PROJECT_ROOT,
            env=coverage_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if coverage_version.returncode != 0:
            print(
                "coverage.py is not installed. Run: python -m pip install coverage",
                file=sys.stderr,
            )
            if coverage_version.stderr:
                print(coverage_version.stderr.strip())
            return 1

        erase_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "erase",
                f"--rcfile={coverage_rc}",
            ],
            cwd=PROJECT_ROOT,
            env=coverage_env,
            check=False,
        )
        if erase_result.returncode != 0:
            print("Failed to initialize coverage data.")
            return 1

    print(f"Logs: {log_dir}", flush=True)

    for index, test in enumerate(SELECTED_TESTS, start=1):
        log_name = test.removeprefix("tests/").removesuffix(".py")
        log_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in log_name)
        log_file = log_dir / f"{index:03d}-{log_name}.log"
        report_file = junit_dir / f"{index:03d}-{log_name}.xml"
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
                "-sv",
                test,
                f"--junitxml={report_file}",
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-sv",
                test,
                f"--junitxml={report_file}",
            ]
        print(
            f"\n=== [{index}/{len(SELECTED_TESTS)}] Running: {test} ===",
            flush=True,
        )
        print(f"Log: {log_file}", flush=True)

        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=coverage_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            returncode = process.wait()

        if report_file.exists():
            report_files.append(report_file)
        else:
            print(f"Warning: no JUnit report produced for {test}", file=sys.stderr)

        if returncode == 0:
            passed_tests.append(test)
            print(f"=== PASSED: {test} ===", flush=True)
        else:
            failed_tests.append(test)
            print(
                f"=== FAILED ({returncode}): {test} ===",
                flush=True,
            )

    merge_junit_reports(report_files, results_file)

    if args.coverage:
        print("\nCombining coverage data...", flush=True)
        combine_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--rcfile={coverage_rc}",
                str(log_dir),
            ],
            cwd=PROJECT_ROOT,
            env=coverage_env,
            check=False,
        )
        coverage_ok = combine_result.returncode == 0
        if coverage_ok:
            xml_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "xml",
                    f"--rcfile={coverage_rc}",
                    "-o",
                    str(coverage_xml),
                ],
                cwd=PROJECT_ROOT,
                env=coverage_env,
                check=False,
            )
            coverage_ok = xml_result.returncode == 0
        if not coverage_ok:
            print("Failed to generate coverage.xml.", file=sys.stderr)

    print("\n=== Pull-request test summary ===")
    print(f"Total:  {len(SELECTED_TESTS)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Logs:   {log_dir}")
    print(f"JUnit:  {results_file}")
    if args.coverage and coverage_ok:
        print(f"Coverage: {coverage_xml}")

    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")

    return 1 if failed_tests or not coverage_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
