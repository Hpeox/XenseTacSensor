from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def import_sensor_client():
    return importlib.import_module("XenseTacSensor.io.sensor_client")


class FakeSensorInstance:
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.released = False

    def exportRuntimeConfig(self, _path) -> None:
        pass

    def selectSensorInfo(self, *_outputs):
        return "rec", "force", "force_norm", "force_resultant"

    def release(self) -> None:
        self.released = True


def make_xensesdk_module(create_calls: list[tuple[str, dict]], create_observer=None):
    class FakeOutputType:
        Rectify = object()
        Force = object()
        ForceNorm = object()
        ForceResultant = object()

    class FakeSensor:
        OutputType = FakeOutputType

        @staticmethod
        def create(sensor_id: str, **kwargs):
            if create_observer is not None:
                create_observer()
            create_calls.append((sensor_id, kwargs))
            return FakeSensorInstance(sensor_id)

    module = types.ModuleType("xensesdk")
    module.Sensor = FakeSensor
    return module


def test_xense_sdk_from_executable_parses_supported_envs():
    sensor_client = import_sensor_client()

    assert (
        sensor_client._xense_sdk_from_executable(
            "/home/robot/miniconda3/envs/xense2/bin/python"
        )
        == ("xense2", "2.0")
    )
    assert (
        sensor_client._xense_sdk_from_executable(
            "/home/robot/miniconda3/envs/Xense310/bin/python"
        )
        == ("Xense310", "1.x")
    )


def test_xense_sdk_from_executable_rejects_unknown_env():
    sensor_client = import_sensor_client()

    with pytest.raises(RuntimeError, match="unsupported Xense conda env"):
        sensor_client._xense_sdk_from_executable(
            "/home/robot/miniconda3/envs/unknown/bin/python"
        )


def test_initialize_uses_sdk_1x_create_signature_and_patch(monkeypatch, tmp_path):
    sensor_client = import_sensor_client()
    create_calls: list[tuple[str, dict]] = []
    patch_calls: list[FakeSensorInstance] = []

    patch_module = types.ModuleType("XenseTacSensor.sdk_patch.xense_patch")
    patch_module.patch_xense_diff_model = lambda sensor: patch_calls.append(sensor)

    monkeypatch.setattr(sys, "executable", "/home/robot/miniconda3/envs/Xense310/bin/python")
    monkeypatch.setattr(sensor_client.Settings, "save_dir", tmp_path)
    monkeypatch.setitem(sys.modules, "xensesdk", make_xensesdk_module(create_calls))
    monkeypatch.setitem(sys.modules, "XenseTacSensor.sdk_patch.xense_patch", patch_module)

    client = sensor_client.SensorClient("sensor-0", "sensor-1", use_gpu=False)
    client.initialize()

    assert create_calls == [
        ("sensor-0", {"use_gpu": False}),
        ("sensor-1", {"use_gpu": False}),
    ]
    assert [sensor.sensor_id for sensor in patch_calls] == ["sensor-0", "sensor-1"]


def test_initialize_uses_sdk_20_import_patch_before_create(monkeypatch, tmp_path):
    sensor_client = import_sensor_client()
    create_calls: list[tuple[str, dict]] = []
    patch_module_name = "XenseTacSensor.sdk_patch.xense2_ort_patch"
    sys.modules.pop(patch_module_name, None)

    def observe_create() -> None:
        assert patch_module_name in sys.modules

    onnxruntime = types.ModuleType("onnxruntime")
    onnxruntime.InferenceSession = object()
    onnxruntime.SessionOptions = lambda: types.SimpleNamespace(
        graph_optimization_level=None,
        log_severity_level=None,
    )
    onnxruntime.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=1)

    monkeypatch.setattr(sys, "executable", "/home/robot/miniconda3/envs/xense2/bin/python")
    monkeypatch.setattr(sensor_client.Settings, "save_dir", tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    monkeypatch.setitem(
        sys.modules,
        "xensesdk",
        make_xensesdk_module(create_calls, create_observer=observe_create),
    )

    client = sensor_client.SensorClient("sensor-0", "sensor-1", use_gpu=False)
    client.initialize()

    assert create_calls == [
        ("sensor-0", {}),
        ("sensor-1", {}),
    ]


def test_acquisition_initialize_sends_error_on_sensor_init_failure(monkeypatch):
    from XenseTacSensor.config.settings import Settings
    from XenseTacSensor.core.service import AcquisitionService
    from XenseTacSensor.core.state import ServiceState
    from XenseTacSensor.protocol.messages import ErrorCode, MsgType

    class FakeSensor:
        def initialize(self):
            raise RuntimeError("bad sdk env")

        def release(self) -> None:
            pass

    class FakeUds:
        def __init__(self):
            self.sent = []

        def start_server(self) -> None:
            pass

        def wait_client(self) -> None:
            pass

        def send_message(self, msg_type, frame_id=-1, payload=None) -> None:
            self.sent.append((msg_type, frame_id, payload))

    service = AcquisitionService(Settings())
    service.sensor = FakeSensor()
    service.uds = FakeUds()

    service.initialize()

    assert service.state == ServiceState.STOPPED
    assert service.uds.sent == [
        (
            MsgType.ERROR,
            0,
            {"code": int(ErrorCode.SENSOR_INIT_FAIL), "reason": "sensor init failed: bad sdk env"},
        )
    ]
