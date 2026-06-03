"""Import-time ONNX Runtime patch for Xense SDK 2.0."""

from pathlib import Path

import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
PATCHED_DIFF_MODEL = SCRIPT_DIR / "xense2_diff_minmax.onnx"
if not PATCHED_DIFF_MODEL.exists():
    raise FileNotFoundError(f"patched Xense 2.0 diff model not found: {PATCHED_DIFF_MODEL}")

_orig_InferenceSession = ort.InferenceSession


def _is_diff_session(sess) -> bool:
    inputs = [(i.name, tuple(i.shape), i.type) for i in sess.get_inputs()]
    outputs = [(o.name, tuple(o.shape), o.type) for o in sess.get_outputs()]

    return (
        inputs == [("image", (1, 3, 240, 144), "tensor(float16)")]
        and outputs == [("image_plain", (1, 3, 240, 144), "tensor(float16)")]
    )


def _patched_InferenceSession(*args, **kwargs):
    original_sess = _orig_InferenceSession(*args, **kwargs)

    if not _is_diff_session(original_sess):
        return original_sess

    print("[xense2_patch] intercept diff model")

    providers = original_sess.get_providers()
    provider_options = original_sess.get_provider_options()
    cuda_opts = provider_options.get("CUDAExecutionProvider")

    if "CUDAExecutionProvider" in providers and cuda_opts is not None:
        patched_providers = [
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]
    else:
        patched_providers = providers

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3

    patched_sess = _orig_InferenceSession(
        str(PATCHED_DIFF_MODEL),
        sess_options=so,
        providers=patched_providers,
    )

    print("[xense2_patch] patched diff providers:", patched_sess.get_providers())
    return patched_sess


ort.InferenceSession = _patched_InferenceSession
