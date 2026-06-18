from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from queue import Empty

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def import_sensor_client():
    return importlib.import_module("XenseTacSensor.io.sensor_client")


class FakeSensorInstance:
    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self.released = False
        self.exported_paths: list[Path] = []

    def exportRuntimeConfig(self, path) -> None:
        self.exported_paths.append(Path(path))

    def selectSensorInfo(self, *_outputs):
        return "rec", "force", "force_norm", "force_resultant"

    def release(self) -> None:
        self.released = True


class FakeQueue:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.puts = []
        self.closed = False
        self.joined = False

    def put(self, item):
        self.puts.append(item)

    def get(self, timeout=None):
        if not self.items:
            raise Empty
        return self.items.pop(0)

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True


class FakeProcess:
    def __init__(self, alive=True):
        self._alive = alive
        self.exitcode = None
        self.join_calls = []
        self.terminated = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.exitcode = -15


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


def test_worker_sensor_uses_sdk_1x_create_signature_and_patch(monkeypatch):
    sensor_client = import_sensor_client()
    create_calls: list[tuple[str, dict]] = []
    patch_calls: list[FakeSensorInstance] = []

    patch_module = types.ModuleType("XenseTacSensor.sdk_patch.xense_patch")
    patch_module.patch_xense_diff_model = lambda sensor: patch_calls.append(sensor)

    monkeypatch.setitem(sys.modules, "XenseTacSensor.sdk_patch.xense_patch", patch_module)

    sensor_client._create_worker_sensor(
        "sensor-0",
        use_gpu=False,
        sdk_version="1.x",
        Sensor=make_xensesdk_module(create_calls).Sensor,
    )

    assert create_calls == [("sensor-0", {"use_gpu": False})]
    assert [sensor.sensor_id for sensor in patch_calls] == ["sensor-0"]


def test_worker_sensor_uses_sdk_20_import_patch_before_create(monkeypatch):
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
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    monkeypatch.setitem(
        sys.modules,
        "xensesdk",
        make_xensesdk_module(create_calls, create_observer=observe_create),
    )

    sdk_version, Sensor = sensor_client.load_sensor_api()
    sensor_client._create_worker_sensor(
        "sensor-0",
        use_gpu=False,
        sdk_version=sdk_version,
        Sensor=Sensor,
    )

    assert create_calls == [("sensor-0", {})]


def test_runtime_config_dir_uses_instance_save_dir(monkeypatch, tmp_path):
    sensor_client = import_sensor_client()
    monkeypatch.setattr(sensor_client.time, "strftime", lambda _format: "20260618_120000")

    runtime_config_dir = sensor_client._runtime_config_dir(tmp_path)

    assert runtime_config_dir == tmp_path / "20260618_120000"
    assert runtime_config_dir.is_dir()


def test_sensor_worker_exports_runtime_config_to_supplied_dir(monkeypatch, tmp_path):
    sensor_client = import_sensor_client()
    sensor_instances: list[FakeSensorInstance] = []

    class FakeOutputType:
        Rectify = object()
        Force = object()
        ForceNorm = object()
        ForceResultant = object()

    class FakeSensor:
        OutputType = FakeOutputType

        @staticmethod
        def create(sensor_id: str, **_kwargs):
            sensor = FakeSensorInstance(sensor_id)
            sensor_instances.append(sensor)
            return sensor

    monkeypatch.setattr(sensor_client, "load_sensor_api", lambda: ("2.0", FakeSensor))
    runtime_config_dir = tmp_path / "runtime_config"
    runtime_config_dir.mkdir()
    command_queue = FakeQueue([("stop", None)])
    result_queue = FakeQueue()

    sensor_client._sensor_worker_main(
        sensor_index=0,
        sensor_id="sensor-0",
        use_gpu=False,
        runtime_config_dir=runtime_config_dir,
        command_queue=command_queue,
        result_queue=result_queue,
    )

    assert sensor_instances[0].exported_paths == [runtime_config_dir]
    assert sensor_instances[0].released is True
    assert result_queue.puts[0]["type"] == "ready"


def test_read_frame_combines_two_worker_results():
    sensor_client = import_sensor_client()
    client = sensor_client.SensorClient("sensor-0", "sensor-1", use_gpu=False)
    client._initialized = True
    client._command_queues = [FakeQueue(), FakeQueue()]
    client._worker_processes = [FakeProcess(), FakeProcess()]
    client._result_queue = FakeQueue(
        [
            {
                "type": "frame",
                "sensor_index": 1,
                "frame_id": 7,
                "timestamp_ns": 222,
                "payload": ("rec1", "force1", "norm1", "result1"),
            },
            {
                "type": "frame",
                "sensor_index": 0,
                "frame_id": 7,
                "timestamp_ns": 111,
                "payload": ("rec0", "force0", "norm0", "result0"),
            },
        ]
    )

    frame = client.read_frame(7)

    assert [queue.puts for queue in client._command_queues] == [[("read", 7)], [("read", 7)]]
    assert frame.frame_id == 7
    assert frame.timestamp_ns_0 == 111
    assert frame.timestamp_ns_1 == 222
    assert frame.rec_0 == "rec0"
    assert frame.force_resultant_1 == "result1"


def test_wait_for_results_rejects_frame_id_mismatch():
    sensor_client = import_sensor_client()
    client = sensor_client.SensorClient("sensor-0", "sensor-1", use_gpu=False)
    client._result_queue = FakeQueue(
        [
            {
                "type": "frame",
                "sensor_index": 0,
                "frame_id": 6,
                "timestamp_ns": 111,
                "payload": ("rec0", "force0", "norm0", "result0"),
            }
        ]
    )
    client._worker_processes = [FakeProcess(), FakeProcess()]

    with pytest.raises(RuntimeError, match="unexpected worker frame_id"):
        client._wait_for_results("frame", expected_frame_id=7, timeout_s=1.0)


def test_release_stops_workers_and_terminates_lingering_process():
    sensor_client = import_sensor_client()
    client = sensor_client.SensorClient(
        "sensor-0",
        "sensor-1",
        worker_stop_timeout_s=0.25,
    )
    client._initialized = True
    client._sensor_api = "worker"
    command_queues = [FakeQueue(), FakeQueue()]
    result_queue = FakeQueue()
    client._command_queues = command_queues
    client._result_queue = result_queue
    stopped_process = FakeProcess(alive=False)
    lingering_process = FakeProcess(alive=True)
    client._worker_processes = [stopped_process, lingering_process]

    client.release()

    assert [queue.puts for queue in command_queues] == [[("stop", None)], [("stop", None)]]
    assert [queue.closed for queue in command_queues] == [True, True]
    assert result_queue.closed is True
    assert stopped_process.join_calls == [0.25]
    assert lingering_process.join_calls == [0.25, 1.0]
    assert lingering_process.terminated is True
    assert client._command_queues == []
    assert client._result_queue is None
    assert client._initialized is False


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


def test_collect_once_sends_error_and_pauses_on_sensor_read_failure(tmp_path):
    from XenseTacSensor.config.settings import Settings
    from XenseTacSensor.core.service import AcquisitionService
    from XenseTacSensor.core.state import ServiceState
    from XenseTacSensor.protocol.messages import ErrorCode, MsgType

    class FakeSensor:
        def read_frame(self, frame_id):
            raise RuntimeError("read failed")

        def release(self) -> None:
            pass

    class FakeUds:
        def __init__(self):
            self.sent = []

        def send_message(self, msg_type, frame_id=-1, payload=None) -> None:
            self.sent.append((msg_type, frame_id, payload))

        def close(self) -> None:
            pass

    settings = Settings(save_dir=tmp_path)
    service = AcquisitionService(settings)
    service.sensor = FakeSensor()
    service.uds = FakeUds()
    service.state = ServiceState.COLLECTING
    service._next_frame_deadline = 0.0

    service._collect_once()

    assert service.state == ServiceState.PAUSED
    assert service.uds.sent == [
        (
            MsgType.ERROR,
            0,
            {"code": int(ErrorCode.SENSOR_READ_FAIL), "reason": "read failed"},
        )
    ]
