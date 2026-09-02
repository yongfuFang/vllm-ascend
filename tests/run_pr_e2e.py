#!/usr/bin/env python3
"""Run only the pull-request tests explicitly listed below."""

from __future__ import annotations

import subprocess
import sys
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
    # "tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py",  # failed
    # "tests/e2e/pull_request/one_card/test_qwen3_5_0_8b.py",  # fail: need picture
    # "tests/e2e/pull_request/one_card/spec_decode/test_eagle.py",  # fail: RedHatAI/Qwen3-8B-speculator.eagle3 not in model market
    # "tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py", # fail
    # "tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py", # fail
]


def main() -> int:
    if not SELECTED_TESTS:
        print("No pull-request tests are selected.", file=sys.stderr)
        return 1

    passed_tests: list[str] = []
    failed_tests: list[str] = []
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "tests" / "outputs" / "pull_request" / run_timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logs: {log_dir}", flush=True)

    for index, test in enumerate(SELECTED_TESTS, start=1):
        command = [sys.executable, "-m", "pytest", "-sv", test]
        log_name = test.removeprefix("tests/").removesuffix(".py")
        log_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in log_name
        )
        log_file = log_dir / f"{index:03d}-{log_name}.log"
        print(
            f"\n=== [{index}/{len(SELECTED_TESTS)}] Running: {test} ===",
            flush=True,
        )
        print(f"Log: {log_file}", flush=True)

        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
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

        if returncode == 0:
            passed_tests.append(test)
            print(f"=== PASSED: {test} ===", flush=True)
        else:
            failed_tests.append(test)
            print(
                f"=== FAILED ({returncode}): {test} ===",
                flush=True,
            )

    print("\n=== Pull-request test summary ===")
    print(f"Total:  {len(SELECTED_TESTS)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Logs:   {log_dir}")

    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")

    return 1 if failed_tests else 0


if __name__ == "__main__":
    raise SystemExit(main())
