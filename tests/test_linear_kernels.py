import json
import subprocess
import sys

from scudo import linear_kernels
from scudo.notebook_cells import SETUP


def test_kernels_mirror_the_transformers_qwen3_5_imports():
    modules = {(module, function) for _, module, function in linear_kernels.KERNELS}
    assert modules == {
        ("causal_conv1d", "causal_conv1d_fn"),
        ("causal_conv1d", "causal_conv1d_update"),
        ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
        ("fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    }


def test_probe_reports_kernel_or_fallback_without_raising():
    status = linear_kernels.probe((
        ("stdlib", "json", "dumps"),
        ("stdlib", "json", "missing_function"),
        ("absent-dist", "scudo_module_that_does_not_exist", "fn"),
    ))
    assert status["dumps"] == "kernel stdlib"
    assert status["missing_function"].startswith("fallback PyTorch:")
    assert status["fn"].startswith("fallback PyTorch: ModuleNotFoundError")


def test_module_prints_one_json_status_per_kernel():
    result = subprocess.run([sys.executable, "-m", "scudo.linear_kernels"],
                            check=True, capture_output=True, text=True)
    status = json.loads(result.stdout)
    assert set(status) == {function for _, _, function in linear_kernels.KERNELS}


def test_setup_logs_kernels_and_keeps_the_source_build_opt_in():
    assert "-m\", \"scudo.linear_kernels\"" in SETUP
    assert 'RUN_LOG["linear_attention_kernels"]' in SETUP
    assert "BUILD_CAUSAL_CONV1D = False" in SETUP
    assert "causal-conv1d==1.7.0" in SETUP
