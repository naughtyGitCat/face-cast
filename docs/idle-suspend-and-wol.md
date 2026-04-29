# Idle Suspend + Wake-on-LAN

GPU 服务端长期空转浪费功耗。face-cast 让 `.11` (GPU server) 在 N 分钟无请求后自动进
hibernate (S4) 或 sleep (S3)，`.13` (UI/客户端) 在打请求前先探活 → 不通就发 WoL 唤
醒 → 等就绪再继续。

## 架构

```
.13 触发 /api/scan
   ↓ spawn `face-cast extract <path>`
.13 子进程 _wake_server() → ensure_alive(server_url)
.13 GET /health (2s 超时)
   ↓ 不通?
.13 同时:
     (a) 直接 UDP magic packet → 10.100.100.255:9
     (b) ssh root@10.100.100.2 "etherwake -i br-lan -b <mac>"   ← 主路径
.11 NIC 收到 → BIOS WoL 触发 → 系统唤醒 (S4 ~30-60s / S3 ~5s)
.11 face-cast 进程随系统恢复 (NSSM 不会重启它)
.13 每 3s GET /health, 每 15s 再发 packet, 直到 wake_timeout_s 或活
.13 走原 extract 流程: model_info → 抽帧 → POST /detect ...
.11 每个请求 bump _last_request_at
   ↓ extract 结束, 无新请求
.11 idle watcher 检测 N 秒无请求 → SetSuspendState(hibernate=1, force=1)
.11 进 S4, 等下一次 WoL
```

> ⚠ **直接 UDP 广播在多网卡 Windows 上不可靠**: 实测 .13 装了 Hyper-V 后, OS 路由
> 表写着广播走物理 NIC, 但实际 packet 被虚拟交换机 NDIS 层吞掉, .11 收不到. 同
> 样的代码从 Mac 或 OpenWrt 路由器 → .11 唤醒成功. **生产部署必须配 SSH gateway
> 通道兜底** (见配置).

## 配置

### `.11` 服务端

NSSM 服务环境变量：

| 变量 | 值 | 说明 |
|---|---|---|
| `FACE_IDLE_SUSPEND_SECS` | `600` | 多久无请求自动 suspend；`0`/未设 = 关闭 |
| `FACE_IDLE_SUSPEND_MODE` | `hibernate` 或 `sleep` | S4 (零功耗) / S3 (待机) |

S4 (hibernate) 必须先在系统层启用：

```powershell
powercfg /hibernate on
powercfg /a   # 确认 "休眠" 出现在 "可用睡眠状态" 列表
```

NIC WoL 必须开（驱动层）。检查：

```powershell
Get-NetAdapter | Where-Object Status -eq Up | ForEach-Object {
  $pwr = Get-NetAdapterPowerManagement -Name $_.Name -ErrorAction SilentlyContinue
  '{0,-30} mac={1} wol_magic={2}' -f $_.Name, $_.MacAddress, $pwr.WakeOnMagicPacket
}
```

`wol_magic=Enabled` 才行。BIOS 里也得开（一般默认开）。

NSSM 改 env：

```powershell
$nssm = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\nssm.exe"
& $nssm set face-cast AppEnvironmentExtra `
    "FACE_MODEL_ROOT=E:\face-cast\models" `
    "FACE_IDLE_SUSPEND_SECS=600" `
    "FACE_IDLE_SUSPEND_MODE=hibernate"
Restart-Service face-cast
```

> ⚠ NSSM 的 `AppEnvironmentExtra` 是**全量覆盖**，必须把所有 env var 一次写完。
> 漏了 `FACE_MODEL_ROOT` 会导致服务找不到 InsightFace 模型。

### `.13` 客户端

`E:\face-cast\config.toml` `[server]` 段：

```toml
[server]
url = "http://10.100.100.11:9000"

# 直发 WoL (这条单独不可靠, 见架构图说明)
mac = "58:11:22:BE:6F:4F"
broadcast = "10.100.100.255"
wake_timeout_s = 90

# SSH gateway 兜底 (生产必配)
wol_gateway = "root@10.100.100.2"
wol_gateway_cmd = "etherwake -i br-lan -b {mac}"
wol_gateway_ssh_key = "E:/face-cast/keys/wol-key"
```

#### 生成 SSH key (一次性配置)

让 LocalSystem 服务能 SSH 到路由器代发 etherwake：

```powershell
# 1. 在 .13 上以 the2n 身份生成 key
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_facewake" -N ""

# 2. 把公钥贴到路由器 (本例 OpenWrt, 走 dropbear)
ssh root@10.100.100.2 "cat >> /etc/dropbear/authorized_keys" \
    < "$env:USERPROFILE\.ssh\id_ed25519_facewake.pub"

# 3. 复制到 SYSTEM 可读位置, 并锁紧 ACL (OpenSSH 强制要求私钥只能由
#    运行用户访问, 否则报 'Bad permissions')
New-Item -ItemType Directory -Force -Path E:\face-cast\keys | Out-Null
Copy-Item "$env:USERPROFILE\.ssh\id_ed25519_facewake" E:\face-cast\keys\wol-key
icacls E:\face-cast\keys\wol-key /inheritance:r
icacls E:\face-cast\keys\wol-key /remove:g 'the2n'
icacls E:\face-cast\keys\wol-key /grant:r 'SYSTEM:F'

# 4. 测试 (从 SYSTEM 上下文跑等价测试)
ssh -i E:\face-cast\keys\wol-key root@10.100.100.2 "etherwake -i br-lan -b 58:11:22:BE:6F:4F"
```

> ⚠ ACL 必须严格. `icacls /grant:r 'the2n:R'` 会让 ssh-client 拒绝, 报 *"Bad
> permissions. Try removing permissions for user: ..."*. 只能保留 SYSTEM, 移除
> 所有其他用户.

## 测试结果

测试日 2026-04-29 14:50-12:30。machine state: `.13` 装了 Hyper-V (Default
Switch + 物理 NIC), `.11` BIOS+NIC WoL 启用, S3 启用, S4 (hibernate) 启动测试
前用 `powercfg /hibernate on` 启用.

### Step 1: S3 sleep 唤醒 ✓

| 时间 | 动作 | 结果 |
|---|---|---|
| 11:33 | NSSM env: `SECS=120 MODE=sleep`; `Restart-Service face-cast` | OK |
| 11:36 | 等 180s; `curl http://10.100.100.11:9000/health` | timeout (4s) → .11 sleeping ✓ |
| 11:36 | `.13` 触发 `/api/scan F:\China\王东瑶` | job `0072e32f` 起跑 |
| 11:36 | 看 job log: `[wakeup] 不通, 发 WoL → mac=... bcast=...` | 发包成功 |
| 11:36-37 | 6 次 retry 每 15s, .11 仍不响应 | **直发 WoL 失败** |
| 11:37 | 90s 超时, extract 报 `ConnectTimeout` to `/model/info` | rc=1 |

诊断步骤 (11:50-12:18):
- 从 Mac (10.100.100.x) 直发 WoL → .11 醒 ✓
- 从 OpenWrt 路由器 (.2) `etherwake -i br-lan` → .11 醒 ✓
- 从 .13 (the2n 交互式) 直发 WoL → .11 不醒 ✗
- 从 .13 (LocalSystem 服务上下文) 直发 WoL → .11 不醒 ✗

结论: **.13 的 Hyper-V Extensible Switch 在 NDIS 层拦截了出站 broadcast**, 即
便 OS 路由表说该走物理 NIC. `Find-NetRoute -RemoteIPAddress 10.100.100.255` 返
回 `InterfaceAlias=以太网`, 但实际 packet 不出该接口.

修法: 加 SSH gateway 路径 (`wol_gateway = root@10.100.100.2`). `.13` 跑
`ssh root@.2 etherwake -i br-lan -b <mac>`, 让路由器代发. 路由器是双网口设备,
不带虚拟交换机, broadcast 可靠.

### Step 1b: S3 sleep + SSH gateway 唤醒 ✓ (生产可用)

| 时间 | 动作 | 结果 |
|---|---|---|
| 12:24 | 等 200s 让 .11 自然进 S3 | timeout (4s) ✓ |
| 12:28 | `/api/scan` job `433de1e4` | 起跑 |
| 12:28 | log: `[wakeup] WoL via 3 ifaces` + `SSH gateway: rc=255 Bad permissions...` | key ACL 太宽 |
| 12:29 | 修 ACL: 仅 SYSTEM:F | OK |
| 12:29 | log: `SSH gateway: ran 'etherwake ...'` | 成功 |
| 12:29 | log: `[wakeup] online after 86s` | .11 唤醒 ✓ (前 5 次 retry 浪费在 ACL 上) |
| 12:30 | extract 处理 8 个视频, rc=0, 1 张新脸 | 端到端跑通 ✓ |

实测下次 (ACL 已对) 唤醒约 15-30s 内首发即中.

### Step 2: S4 hibernate 唤醒 — **未现场验证**

切换 S4 配置时遇到 SSH 问题 (`Mac → .11` 的 ssh key 不知何故失效, `Permission
denied (publickey,password,keyboard-interactive)`), 没法在本测试 session 里更
新 .11 的 NSSM env. 已知:

- `powercfg /hibernate on` 已在 .11 上跑 (步骤 0)
- `powercfg /a` 显示 "休眠 (S4)" 在可用列表
- 客户端代码 (ensure_alive + SSH gateway) 跟 S3 完全相同, 跟 sleep state 无关
- `etherwake` 包从 router 出, 从 BIOS WoL 角度看 S3/S4 行为一样 (都是 NIC 收
  magic packet 触发 ACPI wake event), 唤醒时间略长

**用户后续验证**: SSH 修好后, 在 .11 上跑那段 NSSM `set ... SECS=600 MODE=hibernate`
配置, 重启服务, 等 12 分钟 (10 min idle + 2 min 内 poll), `Test-NetConnection
-Port 22` 应该 fail (S4 比 S3 还断更多), 然后从 .13 触发 /api/scan 看是否能起
来. 关键判据: extract job log 出现 `[wakeup] online after Ns` (N 通常 30-60).

## 排错

### `.11` 没自动 suspend
- 确认服务 log 有 `[idle-suspend] watcher 启动: 阈值 ...` 行
- `FACE_IDLE_SUSPEND_SECS` 必须 > 0；`0` 或空就是关闭
- 服务必须 LocalSystem 跑（默认）；普通用户帐号没 SeShutdownPrivilege

### WoL 不唤醒
- BIOS / NIC 驱动 WoL 没开（最常见）
- 用错 broadcast：`255.255.255.255` 路由器不转，要子网广播 (`10.100.100.255`)
- MAC 写错：必须是物理网卡的 MAC，不是 vEthernet/Hyper-V/ZeroTier 那些虚拟
- Windows 的 "Allow this device to wake the computer" 在 NIC 属性里也要开
- Windows 10/11 默认 "Fast Startup" (混合关机) 关机后不响应 WoL；hibernate 不受影响
- **客户端在多网卡 Windows 上**: 直发 WoL 可能被 Hyper-V 虚拟交换机 NDIS 层拦截.
  job log 看到 `[wakeup] WoL packet sent via N interface(s)` 但 .11 不醒, 八成是这个.
  必须配 `wol_gateway` 走外部代发

### SSH gateway 报 `Bad permissions`
OpenSSH client 强制私钥文件 ACL 严格. 给 `the2n:R` 也算"泄露", 必须只留
`SYSTEM:F`:

```powershell
icacls E:\face-cast\keys\wol-key /inheritance:r
icacls E:\face-cast\keys\wol-key /remove:g 'the2n'
icacls E:\face-cast\keys\wol-key /grant:r 'SYSTEM:F'
```

### CUDA 唤醒后挂
- 症状：`/health` 返回 `ok=true` 但 `/detect` 报错 / 慢
- 原因：onnxruntime-gpu 的 CUDA context 在 hibernate 后没正确恢复
- 临时方案：NSSM 配合 PowerShell 写 wake event handler 让服务自动 restart
- 检查：`/health` 的 `providers` 字段还是不是 `CUDAExecutionProvider`

### 扫描中途 .11 被 suspend
- `FACE_IDLE_SUSPEND_SECS` 设太小（< 一个视频处理时间）。默认 600 (10 min)
  远大于单视频耗时 (3-5s)，正常 batch 不会触发
- 单次请求间隔 > timeout 才会出问题。如果有这种用法，加大 timeout

## 实现细节

### `.11` 端代码 (`src/face_cast/server/main.py`)

- `before_request` hook bump `_last_request_at`
- daemon thread 每 `min(60, max(15, timeout/4))` 秒查一次 idle
- `ctypes.windll.powrprof.SetSuspendState(hibernate, force, disable_wake_event)`
- `SetSuspendState` 阻塞到唤醒，返回后 reset 计时器避免立刻再睡

### `.13` 端代码 (`src/face_cast/client/wakeup.py`)

- `send_wol`: 多网卡时枚举本机所有 IPv4, 各 bind 一次发包 (兜底 Hyper-V 路由问题)
- `send_wol_via_ssh`: 跑 `ssh -i <key> <gateway> "<cmd template>"`, 默认
  `etherwake -i br-lan -b {mac}`
- `is_alive`: `urllib.request.urlopen(url + "/health", timeout=2)` 看 HTTP 状态码
- `ensure_alive`: 探活 → 同时打两条 WoL 通路 → 轮询 + 重发 → 超时返回
- 在 `face-cast extract` / `face-cast detect` 入口前置一次

### config.toml `[server]` 完整字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `url` | `""` | server 健康检查 + extract 调用地址 |
| `mac` | `""` | 物理 NIC MAC, 直发 WoL 用 |
| `broadcast` | `255.255.255.255` | UDP 广播目标地址, 推荐子网定向 |
| `wake_timeout_s` | `90` | WoL 后最多等多久 server 起来 |
| `wol_gateway` | `""` | SSH 代发用的远端, e.g. `root@10.100.100.2` |
| `wol_gateway_cmd` | `etherwake -i br-lan -b {mac}` | 要在 gateway 跑的命令模板, `{mac}` 会被替换 |
| `wol_gateway_ssh_key` | `""` | SSH 私钥绝对路径 (LocalSystem 可读) |

## 历史决策

- **为啥 hibernate 而非 sleep**：S3 风扇还转、内存供电不省功；S4 完全断电.
  家用唤醒慢一点 (30-60s) 可接受
- **为啥不用 `shutdown.exe /h`**：fork 子进程开销 + 不能切 sleep；`SetSuspendState`
  原生 API 直接调
- **为啥不在每个 request 后立刻 suspend**：连续多个请求是常态（extract batch），
  每次都 sleep-wake 反而费电也卡顿. idle timer 是常规做法
- **为啥要 SSH gateway 兜底**：实测 Hyper-V Extensible Switch 在 NDIS 层吃掉了
  从 .13 出去的 broadcast packet, 这是 Windows 主机网络的已知坑. 把 WoL 的责
  任推给一个没装 Hyper-V 的中继节点 (路由器) 是最便宜的解
- **为啥 client 端不重试 per-request**：现状是 batch 之初探活一次就够, 扫描期间
  持续刷新 server idle. 如果未来出现"长间隔单次请求"场景, 可在 `FaceClient`
  加 retry 装饰器
