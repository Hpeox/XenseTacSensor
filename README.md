# XenseTacSensor

## 项目概述
XenseTacSensor 是一个双传感器触觉采集服务，提供以下能力：
- 双传感器串行采集（sensor_0 + sensor_1）
- UDS 控制指令驱动采集状态机
- 共享内存输出当前帧（用于外部进程低延迟读取）
- 本地字典缓存，并按 demo 粒度落盘

当前实现重点是采集服务本体，不包含 ROS2 节点实现。

## 依赖
- Python 3.10+
- numpy
- xensesdk
- Linux 环境（依赖 Unix Domain Socket 与共享内存）

## 当前目录
- `app.py`: 服务入口
- `config/settings.py`: 配置项
- `core/service.py`: 采集主流程与状态机动作
- `core/state.py`: 状态定义与状态迁移约束
- `io/sensor_client.py`: 传感器封装
- `io/shm_writer.py`: 共享内存 schema 与写入
- `io/uds_channel.py`: UDS 收发
- `io/local_store.py`: 内存字典缓存与落盘
- `protocol/messages.py`: 协议消息定义

## 状态机模型
状态：
- `BOOT`
- `INIT`
- `WAIT_START`
- `COLLECTING`
- `PAUSED`
- `STOPPED`

关键语义：
- `WAIT_START`: demo 间等待态，不读取传感器
- `COLLECTING`: 正在采集
- `PAUSED`: 暂停采集（不自动落盘）

## UDS 指令语义
消息类型在 `protocol/messages.py` 中定义。

控制面：
- `INIT_REQ`: 请求初始化状态
- `INIT_READY`: 服务端已初始化完成
- `START_REQ`: 开始采集（`WAIT_START`/`PAUSED` -> `COLLECTING`）
- `PAUSE_REQ`: 暂停采集（`COLLECTING` -> `PAUSED`）
- `DEMO_DONE_REQ`: 当前 demo 结束并落盘（`COLLECTING` -> `WAIT_START`）
- `DEMO_DISCARD_REQ`: 当前 demo 放弃，不落盘（`COLLECTING` -> `WAIT_START`）
- `STOP_REQ`: 停止服务并退出

通用返回：
- `ACK`: 成功应答
- `ERROR`: 失败应答，payload 中包含 code 与 reason

数据通知：
- `FRAME_READY`: 新帧可读，携带 `frame_id` 与双时间戳 payload

## Warmup 与帧编号
- sensor 初始化阶段会读取一帧 warmup 数据（`frame_id = -1`），仅用于 shm schema 探测。
- 业务采集帧从 `frame_id = 0` 开始递增。

## 共享内存格式（v2 双缓冲）
本项目当前仅定义并支持 v2 双缓冲协议（无旧协议兼容要求）。

### 冲突核对与修正
- 旧文档中的单缓冲 HEADER_FMT=<QqqBB6x 已废弃。
- 旧文档中的 writing/valid 位切换流程已废弃，替换为 `latest_index` 双槽位发布。
- 浮点张量不会在 shm writer 中强制转 float32；dtype 以 seed frame 推导并保存在 schema 中。

### 内存布局
共享内存按如下顺序布局：
1) Global Header
2) Slot 0: Slot Header + Slot Payload
3) Slot 1: Slot Header + Slot Payload

相关常量：
```
SHM_LAYOUT_VERSION = 2
SLOT_COUNT = 2
GLOBAL_HEADER_FMT = <I4x
SLOT_HEADER_FMT = <QQqq
```

Global Header 字段：
- `latest_index`: uint32，表示当前最新完整帧所在槽位（0 或 1）

Slot Header 字段：
- `seq`: uint64，奇数表示写入中，偶数表示稳定快照
- `frame_id`: uint64
- `timestamp_ns_0`: int64
- `timestamp_ns_1`: int64

默认写入张量键：
- `rec_0`, `force_0`, `force_norm_0`, `force_resultant_0`
- `rec_1`, `force_1`, `force_norm_1`, `force_resultant_1`

偏移公式：
```
slot_stride = slot_header_size + payload_size
slot_base(i) = global_header_size + i * slot_stride
slot_payload_base(i) = slot_base(i) + slot_header_size
tensor_abs_offset(i, tensor) = slot_payload_base(i) + tensor.offset
```

### 写入发布流程
每次写入一帧：
1) 读取 `latest_index`，选择非最新槽位 `write_slot`
2) 将 `write_slot` 的 `seq` 写为奇数（写入开始）
3) 写入该槽位全部 payload
4) 将 `write_slot` 的 `seq` 写为偶数并更新 `frame_id`/`timestamp`（写入完成）
5) 更新 `latest_index = write_slot` 完成发布

### 读 SHM 协议（建议实现）
读取最新帧时建议执行以下流程：
1) 读取 `latest_a`
2) 读取 `latest_a` 槽位 header 得到 `seq_a`/`frame_id`/`timestamp`
3) 若 `seq_a` 为奇数，重试
4) 按偏移公式复制该槽位 payload 到本地
5) 再读取同槽位 header 得到 `seq_b`
6) 再读取 `latest_b`
7) 仅当 `latest_a == latest_b` 且 `seq_a == seq_b` 且 `seq_b` 为偶数时，判定本次读取成功；否则重试

失败重试语义：
- 在竞争窗口内重试是预期行为。
- 建议限制单次读取最大重试次数（例如 100 次），超过后上报读取失败并进入下一轮读取。

初始帧语义：
- 服务启动后在首帧发布前，latest 槽位可能仍是初始化内容，读端应允许 `frame_id=0` 且 `timestamp=0` 的初始化状态。

## Demo 落盘策略
缓存容器为 `local_store.data_dict`。

行为：
- `DEMO_DONE_REQ`: flush 到 `data_demo_{demo_tag}.npy`，然后 clear 内存
- `DEMO_DISCARD_REQ`: 仅 clear 内存，不写盘
- `STOP_REQ`: 若内存仍有数据，执行一次兜底 flush

## SDK Patch
项目包含一个针对 Xense SDK `_diff_model` 的小补丁，用于避免最后的 `Clip` 节点回退到 CPU，从而降低 CPU 占用并提升稳定性。
详细说明与使用方法见：[sdk_patch/README.md](sdk_patch/README.md)。

## 启动方式
在项目根目录执行：

```
python -m XenseTacSensor.app --uds-path /tmp/xense_sensor.sock --shm-name xense_sensor_frame --fps 30
```

可选参数：
- `--uds-path`: UDS socket 路径
- `--shm-name`: 共享内存名称
- `--fps`: 目标采样频率，必须大于 0

## 最小 UDS 测试客户端
新增了一个最小联调客户端：
- `XenseTacSensor/uds_test_client.py`

用途：
- 键盘控制状态切换命令发送
- 异步接收并显示 UDS 消息（`ACK`/`ERROR`/`FRAME_READY` 等）
- 自动重连、消息日志落盘、脚本化回归序列
- 可选启用 SHM 联调 reader，并由 UDS `ACK` 事件同步控制 reader 启停

控制消息约定：
- 客户端发送控制类 UDS 消息时默认使用 `frame_id=-1`。
- 脚本模式和交互模式在发送 `s/p/d/x/q` 前会先等待 `INIT_READY`，避免服务端未就绪导致状态错误。
- `q`（`STOP_REQ`）会等待服务端 `ACK` 后再退出客户端。

常用联调参数：
- `--with-shm-reader`: 启用 SHM 联调 reader，同一进程内联动 UDS 与共享内存读取。
- `--init-timeout`: 等待 `INIT_READY` 的超时时间。
- `--ack-timeout`: 等待 `START_REQ` / `DEMO_DONE_REQ` / `STOP_REQ` `ACK` 的超时时间。
- `--done-stop-delay-ms`: 发送 `DEMO_DONE_REQ` 后，reader 延迟停止的时间。
- `--reader-capture-timeout-ms`: 首帧捕获窗口时长，默认按 `frame_id=0` 进行严格检查。
- `--reader-capture-poll-ms`: 首帧捕获阶段的轮询间隔。
- `--reader-target-hz`: reader 稳态读取频率。
- `--reader-dephase-every-n`: 稳态读取中每 N 帧加入一次去相位偏置。
- `--reader-dephase-ms`: 去相位偏置的毫秒数。

启动示例：

```
python -m XenseTacSensor.uds_test_client --uds-path /tmp/xense_sensor.sock --with-shm-reader
```

键盘命令（输入后回车）：
- `h`: 显示帮助
- `i`: `INIT_REQ`
- `s`: `START_REQ`
- `p`: `PAUSE_REQ`
- `d`: `DEMO_DONE_REQ`
- `x`: `DEMO_DISCARD_REQ`
- `q`: 发送 `STOP_REQ`，等待 `ACK` 后退出客户端
- `e`: 仅退出客户端（不发送 `STOP_REQ`）

脚本回归示例：

```
python -m XenseTacSensor.uds_test_client \
	--with-shm-reader \
	--uds-path /tmp/xense_sensor.sock \
	--script "s,wait:2,p,wait:1,s,d,q"
```

说明：
- `wait:N` 表示等待 N 秒。
- 可通过 `--script-file` 指定脚本文件（每行一个 token，支持 # 注释）。
- 默认日志路径为 `./runtime/uds_test_client.log.jsonl`，可通过 `--log-file` 覆盖。
- 客户端会在断连后按 `--retry`（默认 1 秒）自动重连。
- `q` token 仅在收到 `STOP_REQ` `ACK` 后才结束脚本流程；若 `ACK` 超时会报错并保持客户端运行。
- 如果启用了 `--with-shm-reader`，则脚本中的 `s/d` 会同步控制 reader 启停；`d` 后 reader 不等待 `ACK`，而是在 `--done-stop-delay-ms` 后停止。

## 最小 SHM 读端联合测试
新增最小 SHM 读端联调脚本：
- `XenseTacSensor/shm_read_test_client.py`

用途：
- 连接已有共享内存并按 v2 协议读取最新槽位
- 执行 `latest`+`seq` 双重校验并统计重试次数
- 验证 `frame_id` 单调、槽位切换和基本读写一致性

联调行为：
- strict 首帧检测基准为 frame_id=0。
- 在 --with-shm-reader 模式下，发送 DEMO_DONE_REQ 后读端不会等待 ACK，而是按 --done-stop-delay-ms（默认 100ms）延迟后停止 reader，避免落盘阶段阻塞过久。

启动示例：

```
python -m XenseTacSensor.uds_test_client \
	--with-shm-reader \
	--uds-path /tmp/xense_sensor.sock \
	--script "s,wait:2,x,s,wait:1,d,wait:1,s,wait:2,q"
```

SHM 读端独立示例：

```
python -m XenseTacSensor.shm_read_test_client --shm-name xense_sensor_frame --duration 5
```

## 已知限制
- 当前 UDS 连接模型为单连接。
- `wait_client` 为阻塞 accept，需确保控制端按预期连接。
- `demo_tag` 当前按秒生成，若同秒触发多个 demo，文件名可能冲突。

## TODO
- 将 `core/service.py` 中的 `service.collectonce` 改成并行实现以降低时序误差

## 后续建议
- 增加 UDS 断连重连逻辑
- 增加分段落盘策略（长 demo 防止内存过高）
- 完善联调脚本与时延统计工具
