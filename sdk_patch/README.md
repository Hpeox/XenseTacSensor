# Xense SDK `_diff_model` CUDA Patch

## 背景

在 Xense SDK 的默认推理路径中，`_infer_engine._diff_model` 使用 ONNX Runtime 执行。profiling 发现原始 `_diff_model` 中最后的：`Clip(x, 0, 1)`会被 ONNX Runtime 分配到 `CPUExecutionProvider`，并插入若干 `Cast` 节点，导致 SDK 在 30 Hz 运行时产生较高 CPU 占用。该 patch 将 `_diff_model` 中的最后一层 `Clip(x, 0, 1)` 改写为数学等价的：`Min(Max(x, 0), 1)`从而避免该节点 fallback 到 CPU。

当前仓库同时支持 Xense SDK `1.x` 和 `2.0`。两版 patch 的触发时机不同：

- SDK `1.x`：创建 sensor 后替换 SDK 内部 `_infer_engine._diff_model` session。
- SDK `2.0`：必须在导入 `xensesdk.Sensor` 前导入 ORT patch，让 `InferenceSession`
  创建时自动拦截 diff model。

## 文件

```text
sdk_patch/
├── xense_patch.py
├── diff_model_minmax.onnx
├── xense2_ort_patch.py
└── xense2_diff_minmax.onnx
```

其中：

- `diff_model_minmax.onnx`：SDK `1.x` 使用的 patched `_diff_model`
- `xense_patch.py`：SDK `1.x` 使用，用于在 `Sensor.create()` 后替换 SDK 内部 `_diff_model`
- `xense2_diff_minmax.onnx`：SDK `2.0` 使用的 patched `_diff_model`，文件名刻意和
  `1.x` 模型不同，避免混淆
- `xense2_ort_patch.py`：SDK `2.0` 使用，导入即 patch ONNX Runtime session 创建逻辑

## SDK 1.x 使用方法

在创建传感器后、第一次调用 `selectSensorInfo()` 前执行 patch：

```python
from xensesdk import Sensor
from sdk_patch.xense_patch import patch_xense_diff_model

sensor_0 = Sensor.create(sensor_id_0, use_gpu=True)
sensor_1 = Sensor.create(sensor_id_1, use_gpu=True)

patch_xense_diff_model(sensor_0)
patch_xense_diff_model(sensor_1)

# warm-up / normal inference
rec_0, force_0, force_norm_0, force_resultant_0 = sensor_0.selectSensorInfo(
    Sensor.OutputType.Rectify,
    Sensor.OutputType.Force,
    Sensor.OutputType.ForceNorm,
    Sensor.OutputType.ForceResultant,
)
```

推荐顺序：

```text
Sensor.create(...)
patch_xense_diff_model(...)
第一次 warm-up selectSensorInfo(...)
正式循环
```

不建议在正式循环运行过程中动态替换模型。

## SDK 2.0 使用方法

SDK `2.0` 必须在 `from xensesdk import Sensor` 之前导入 patch：

```python
from XenseTacSensor.sdk_patch import xense2_ort_patch  # import-time patch
from xensesdk import Sensor

sensor_0 = Sensor.create(sensor_id_0)
sensor_1 = Sensor.create(sensor_id_1)
```

`xense2_ort_patch.py` 使用自身文件位置解析 `xense2_diff_minmax.onnx`，不依赖当前工作目录。

## 实现方式

SDK `1.x` 的 `patch_xense_diff_model()` 会：

1. 读取 SDK 原始 `_diff_model` 的 CUDA provider options；
2. 使用 `diff_model_minmax.onnx` 创建新的 ONNX Runtime session；
3. 设置：

```python
session.disable_cpu_ep_fallback = 1
```

1. 将新的 session 写回：

```python
sensor._infer_engine._diff_model
```

这样如果后续模型中仍有节点需要 CPU fallback，程序会直接报错，而不是静默退回 CPU。

## 验证结果

已完成以下验证：

### 1. Provider 验证

patched 后在 strict 模式下，三个模型均只观察到 `CUDAExecutionProvider`：

```text
_diff_model   CUDAExecutionProvider only
_depth_model  CUDAExecutionProvider only
_flow_model   CUDAExecutionProvider only
```

### 2. 数值一致性验证

在空载和模拟受力工况下，对比：

```text
original vs patched
original vs original
```

结果显示 patched 引入的误差与原始模型重复运行误差同量级，未观察到额外数值风险。

### 3. 性能验证

30 Hz 单传感器测试中：

```text
baseline:
  select avg ≈ 5.674 ms
  steady CPU ≈ 640%

patched_strict:
  select avg ≈ 4.959 ms
  steady CPU ≈ 40%–45%
```

主要收益是显著降低 CPU 占用，并消除 `_diff_model` 的 CPU fallback；平均单帧耗时只小幅下降。

## 注意事项

- 该 patch 依赖 SDK 内部私有成员：

```python
sensor._infer_engine._diff_model
```

因此 SDK 更新后需要重新验证。

- `get_providers()` 可能仍显示：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

这不一定表示发生了 CPU fallback。实际判断应以 strict fallback 禁用测试或 ORT profiling 为准。

- 当前 patch 不启用 CUDA Graph。此前测试表明，直接在 SDK 的 `session.run()` 路径中启用 CUDA Graph 会导致运行时异常。

## 建议

在实际任务中默认启用该 patch。若后续升级 SDK、ONNX Runtime、CUDA 或模型文件，需要重新运行：

1. strict CUDA fallback 测试；
2. replay 数值一致性测试；
3. 30 Hz CPU 占用测试。
