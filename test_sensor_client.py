from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from queue import Empty

import numpy as np
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


def _tactile_data_dict(sensor0: np.ndarray, sensor1: np.ndarray) -> dict:
    frames = {}
    for index, (force0, force1) in enumerate(zip(sensor0, sensor1)):
        frames[f"{index:05d}"] = {
            "OG000544_force_resultant": np.asarray(force0, dtype=np.float64),
            "OG001009_force_resultant": np.asarray(force1, dtype=np.float64),
        }
    return {"events": {}, "frames_data": frames}


def test_tactile_qc_allows_both_zero_force():
    from XenseTacSensor.core.tactile_qc import compute_tactile_qc

    zeros = np.zeros((4, 6), dtype=np.float64)

    result = compute_tactile_qc(
        _tactile_data_dict(zeros, zeros),
        sensor_ids=("OG000544", "OG001009"),
        zero_force_mean_tolerance=0.1,
        edge_warning_threshold=0.5,
        edge_window_samples=15,
    )

    assert result.manifest["ok"] is True
    assert [sensor["zero_force"] for sensor in result.manifest["sensors"]] == [True, True]
    assert result.preview is not None
    assert result.preview.force_resultant.shape == (2, 4, 6)


def test_tactile_qc_fails_exactly_one_zero_force():
    from XenseTacSensor.core.tactile_qc import compute_tactile_qc

    zeros = np.zeros((4, 6), dtype=np.float64)
    nonzero = np.ones((4, 6), dtype=np.float64)

    result = compute_tactile_qc(
        _tactile_data_dict(zeros, nonzero),
        sensor_ids=("OG000544", "OG001009"),
        zero_force_mean_tolerance=0.1,
        edge_warning_threshold=0.5,
        edge_window_samples=15,
    )

    assert result.manifest["ok"] is False
    assert [sensor["zero_force"] for sensor in result.manifest["sensors"]] == [True, False]


def test_tactile_qc_edge_warning_and_preview_write(tmp_path):
    from XenseTacSensor.core.tactile_qc import compute_tactile_qc, write_tactile_preview_npz

    normal = np.full((20, 6), 0.05, dtype=np.float64)
    edge = np.full((20, 6), 0.05, dtype=np.float64)
    edge[:15, 2] = 0.6

    result = compute_tactile_qc(
        _tactile_data_dict(normal, edge),
        sensor_ids=("OG000544", "OG001009"),
        zero_force_mean_tolerance=0.1,
        edge_warning_threshold=0.5,
        edge_window_samples=15,
    )

    assert result.manifest["ok"] is True
    assert result.manifest["has_warning"] is True
    assert [sensor["edge_warning"] for sensor in result.manifest["sensors"]] == [False, True]
    assert result.preview is not None
    output_path = write_tactile_preview_npz(result.preview, tmp_path / "preview.npz")
    with np.load(output_path, allow_pickle=False) as data:
        assert set(data.files) == {
            "sensor_ids",
            "frame_index",
            "force_resultant",
            "edge_warning",
            "edge_max",
        }
        np.testing.assert_array_equal(data["edge_warning"], [False, True])


def test_mock_sensor_client_uses_production_tensor_schema_without_sdk():
    from XenseTacSensor.io.sensor_client import MockSensorClient

    client = MockSensorClient()
    warmup = client.initialize()
    frame = client.read_frame(7)

    assert warmup.frame_id == -1
    assert frame.frame_id == 7
    assert frame.timestamp_ns_0 > 0
    assert frame.timestamp_ns_1 > 0
    for value in (frame.rec_0, frame.rec_1):
        assert value.shape == (700, 400, 3)
        assert value.dtype == np.dtype(np.uint8)
        assert not value.any()
    for value in (
        frame.force_0,
        frame.force_norm_0,
        frame.force_1,
        frame.force_norm_1,
    ):
        assert value.shape == (35, 20, 3)
        assert value.dtype == np.dtype(np.float64)
        assert not value.any()
    for value in (frame.force_resultant_0, frame.force_resultant_1):
        assert value.shape == (6,)
        assert value.dtype == np.dtype(np.float64)
        assert not value.any()

    client.release()
    with pytest.raises(RuntimeError, match="released"):
        client.read_frame(8)


def test_xense_app_injects_mock_sensor_client(monkeypatch):
    from XenseTacSensor import app
    from XenseTacSensor.io.sensor_client import MockSensorClient

    captured = {}

    class FakeService:
        def __init__(self, settings, sensor_client=None):
            captured["settings"] = settings
            captured["sensor_client"] = sensor_client

        def run_forever(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(app, "AcquisitionService", FakeService)
    monkeypatch.setattr(sys, "argv", ["xense", "--mock"])

    app.main()

    assert isinstance(captured["sensor_client"], MockSensorClient)
    assert captured["ran"] is True


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
        == ("xense2", "2.0.1")
    )
    assert (
        sensor_client._xense_sdk_from_executable(
            "/home/robot/miniconda3/envs/xense2_bak/bin/python"
        )
        == ("xense2_bak", "2.0")
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


def test_worker_sensor_uses_sdk_201_import_patch_before_create(monkeypatch):
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
    assert sdk_version == "2.0.1"
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


def test_mock_client_runs_existing_acquisition_lifecycle_and_persists_tactile_data(tmp_path):
    from XenseTacSensor.config.settings import Settings
    from XenseTacSensor.core.service import AcquisitionService
    from XenseTacSensor.core.state import ServiceState
    from XenseTacSensor.io.sensor_client import MockSensorClient
    from XenseTacSensor.protocol.messages import MsgType

    class FakeUds:
        def __init__(self):
            self.inbound = []
            self.sent = []
            self.started = False
            self.waited = False
            self.closed = False

        def start_server(self) -> None:
            self.started = True

        def wait_client(self) -> None:
            self.waited = True

        def try_recv_message(self, max_wait_s=0.0):
            del max_wait_s
            return self.inbound.pop(0) if self.inbound else None

        def send_message(self, msg_type, frame_id=-1, payload=None) -> None:
            self.sent.append((msg_type, frame_id, payload))

        def close(self) -> None:
            self.closed = True

    settings = Settings(
        uds_path=str(tmp_path / "xense.sock"),
        shm_name=f"xense_mock_{tmp_path.name}",
        save_dir=tmp_path / "runtime_frames",
        tactile_preview_dir=tmp_path / "preview",
        target_fps=1000.0,
    )
    uds = FakeUds()
    service = AcquisitionService(settings, sensor_client=MockSensorClient())
    service.uds = uds
    try:
        service.initialize()
        assert uds.started is True
        assert uds.waited is True
        assert service.shm_writer is not None
        schema = service.shm_writer.schema()
        assert [(tensor["shape"], tensor["dtype"]) for tensor in schema["tensors"]] == [
            ((700, 400, 3), "uint8"),
            ((35, 20, 3), "float64"),
            ((35, 20, 3), "float64"),
            ((6,), "float64"),
            ((700, 400, 3), "uint8"),
            ((35, 20, 3), "float64"),
            ((35, 20, 3), "float64"),
            ((6,), "float64"),
        ]

        service._set_state(ServiceState.WAIT_START)
        uds.inbound.append((MsgType.START_REQ, -1, {}))
        service._process_control_messages()
        assert service.state == ServiceState.COLLECTING
        uds.inbound.append((MsgType.PAUSE_REQ, -1, {}))
        service._process_control_messages()
        assert service.state == ServiceState.PAUSED
        uds.inbound.append((MsgType.START_REQ, -1, {}))
        service._process_control_messages()
        service._collect_once()
        frame_payload = next(
            payload
            for msg_type, _frame_id, payload in uds.sent
            if msg_type == MsgType.FRAME_READY
        )
        assert frame_payload["timestamp_ns_0"] > 0
        assert frame_payload["timestamp_ns_1"] > 0
        uds.inbound.append((MsgType.DEMO_DONE_REQ, -1, {}))
        service._process_control_messages()

        done_payload = next(
            payload
            for msg_type, _frame_id, payload in reversed(uds.sent)
            if msg_type == MsgType.ACK and payload and payload.get("cmd") == "DEMO_DONE_REQ"
        )
        assert service.state == ServiceState.WAIT_START
        assert done_payload["saved_file"].startswith("data_TAC_")
        assert done_payload["xense_tactile_postcheck"]["ok"] is True
        assert done_payload["xense_tactile_postcheck"]["has_warning"] is False
        assert done_payload["xense_tactile_preview"]["ok"] is True

        saved_path = settings.save_dir / done_payload["saved_file"]
        saved = np.load(saved_path, allow_pickle=True).item()
        frame = saved["frames_data"]["00000"]
        assert frame[f"{settings.sensor_id_0}_rec"].shape == (700, 400, 3)
        assert frame[f"{settings.sensor_id_1}_force"].shape == (35, 20, 3)
        assert frame[f"{settings.sensor_id_0}_force_resultant"].shape == (6,)
        assert not frame[f"{settings.sensor_id_0}_force_resultant"].any()
        assert not frame[f"{settings.sensor_id_1}_force_resultant"].any()
        assert Path(done_payload["xense_tactile_preview"]["path"]).exists()

        uds.inbound.extend(
            [
                (MsgType.START_REQ, -1, {}),
                (MsgType.DEMO_DISCARD_REQ, -1, {}),
                (MsgType.STOP_REQ, -1, {}),
            ]
        )
        service._process_control_messages()
        service._collect_once()
        service._process_control_messages()
        service._process_control_messages()
        assert service.state == ServiceState.STOPPED
        assert any(
            msg_type == MsgType.ACK and payload and payload.get("cmd") == "DEMO_DISCARD_REQ"
            for msg_type, _frame_id, payload in uds.sent
        )
        assert any(
            msg_type == MsgType.ACK and payload and payload.get("cmd") == "STOP_REQ"
            for msg_type, _frame_id, payload in uds.sent
        )
    finally:
        service.shutdown()
    assert uds.closed is True
