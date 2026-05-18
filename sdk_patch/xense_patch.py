from pathlib import Path
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
PATCHED_DIFF_MODEL = Path(SCRIPT_DIR / "diff_model_minmax.onnx")


def make_strict_cuda_session_from_file(model_path: Path, old_sess: ort.InferenceSession) -> ort.InferenceSession:
    cuda_opts = dict(old_sess.get_provider_options().get("CUDAExecutionProvider", {}))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3

    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    return ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=[("CUDAExecutionProvider", cuda_opts)],
    )


def patch_xense_diff_model(sensor, model_path: Path = PATCHED_DIFF_MODEL, strict: bool = True) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"patched diff model not found: {model_path}")

    old = sensor._infer_engine._diff_model

    if strict:
        patched = make_strict_cuda_session_from_file(model_path, old)
    else:
        cuda_opts = dict(old.get_provider_options().get("CUDAExecutionProvider", {}))
        patched = ort.InferenceSession(
            str(model_path),
            providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
        )

    sensor._infer_engine._diff_model = patched

    providers = patched.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Unexpected providers after patch: {providers}")

    print("[xense_patch] patched _diff_model with strict CUDA session")