"""Wake-on-LAN + 健康探活 — 用于配合 face-cast server 端的 idle suspend.

流程:
  1. 任何要打 server 的 CLI 入口 (extract / detect) 启动时先调 ``ensure_alive()``
  2. ``ensure_alive`` 先 GET /health (短超时); 通了就直接走
  3. 不通就发 WoL magic packet 到广播地址, 然后每 3s 探活一次, 等到
     ``wake_timeout_s`` (默认 90s) 或活了为止
  4. 活了再返回, 调用方继续走原流程
  5. 扫描期间持续打请求, server idle timer 不会触发
"""

from __future__ import annotations

import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _local_ipv4s() -> list[str]:
    """枚举本机所有非环回非 link-local IPv4. Windows / Linux 都能用.

    动机: 服务作为 LocalSystem 跑时, 默认路由可能挑到错的接口 (e.g.
    Hyper-V vEthernet 而不是物理网卡), 广播包永远到不了目标 LAN.
    把 socket 显式 bind 到每张接口各发一次, 哪条对的总有一发命中.
    """
    out: set[str] = set()
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        out.update(ips)
    except OSError:
        pass
    return sorted(
        ip for ip in out
        if not ip.startswith(("127.", "169.254."))
    )


def send_wol(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> None:
    """发 magic packet 唤醒指定 MAC 的机器.

    magic packet = 6 字节 0xFF + 16 次重复 MAC = 102 字节, 走 UDP 广播.

    ``broadcast``: 子网定向广播 (e.g. ``10.100.100.255``) 比 limited
    broadcast (``255.255.255.255``) 更可靠 — 路由器一般不转发后者.

    实现细节: 多网卡机器 (Hyper-V / WSL / VPN / Docker) 上, OS 默认路由可能
    把广播包推到错的接口. 我们枚举所有本机 IPv4, 显式 bind 到每张接口各发
    一次, 哪条对的总有一发到. 加上 ``''`` (让 OS 自己挑) 作兜底.
    """
    mac_clean = mac.replace(":", "").replace("-", "").strip()
    if len(mac_clean) != 12:
        raise ValueError(f"bad MAC: {mac!r}")
    try:
        mac_bytes = bytes.fromhex(mac_clean)
    except ValueError as e:
        raise ValueError(f"bad MAC hex: {mac!r}") from e

    packet = b"\xff" * 6 + mac_bytes * 16

    # 所有本机 IP + 一个空字符串 (让 OS 默认挑)
    bind_ips: list[str] = [*_local_ipv4s(), ""]
    sent_from: list[str] = []
    for src_ip in bind_ips:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            if src_ip:
                try:
                    s.bind((src_ip, 0))
                except OSError:
                    continue  # 这张接口没法 bind, 跳过
            # 同时打两个常见 WoL 端口, 部分设备只听其中一个
            for p in (port, 7):
                try:
                    s.sendto(packet, (broadcast, p))
                except OSError:
                    pass
            sent_from.append(src_ip or "<default>")
        except OSError:
            pass
        finally:
            s.close()
    if sent_from:
        print(f"[wakeup] WoL packet sent via {len(sent_from)} interface(s): "
              f"{', '.join(sent_from)}", file=sys.stderr, flush=True)


def send_wol_via_ssh(
    mac: str,
    *,
    gateway: str,
    cmd_template: str = "etherwake -i br-lan -b {mac}",
    ssh_key: str | None = None,
    timeout: float = 8.0,
) -> bool:
    """通过 SSH 到 gateway (e.g. ``root@10.100.100.2``) 跑 ``etherwake``.

    动机: 某些 Windows 多网卡场景 (Hyper-V 虚拟交换机存在时), 主机发的 UDP
    broadcast 不会真正到达物理 LAN 上的睡眠机器, 导致 :func:`send_wol`
    形同虚设. SSH 到一台同 LAN 上能可靠发广播的设备 (路由器最佳) 让它代发.

    需要 gateway 上有 ``etherwake``, 且 ssh key 已配好免密登录.

    返回 True iff 进程退出码为 0. 出错只 print 不抛.
    """
    cmd = cmd_template.format(mac=mac)
    args = ["ssh", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=5"]
    if ssh_key:
        args += ["-i", ssh_key]
    args += [gateway, cmd]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, text=True)
        if r.returncode == 0:
            print(f"[wakeup] SSH gateway {gateway}: ran {cmd!r}",
                  file=sys.stderr, flush=True)
            return True
        print(f"[wakeup] SSH gateway {gateway} rc={r.returncode}: "
              f"stderr={r.stderr.strip()[:200]!r}",
              file=sys.stderr, flush=True)
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"[wakeup] SSH gateway {gateway} failed: {e}",
              file=sys.stderr, flush=True)
        return False


def is_alive(server_url: str, timeout: float = 2.0) -> bool:
    """GET ``<server_url>/health``, 超时/连接失败/非 200 都算 down."""
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def _emit(mac: str, broadcast: str,
          gateway: str | None, gateway_cmd: str, gateway_ssh_key: str | None) -> None:
    """同时尝试两条通路: 直接 UDP 广播 + SSH gateway. 哪条通哪条赢."""
    send_wol(mac, broadcast)
    if gateway:
        send_wol_via_ssh(
            mac, gateway=gateway,
            cmd_template=gateway_cmd,
            ssh_key=gateway_ssh_key,
        )


def ensure_alive(
    server_url: str,
    *,
    mac: str | None = None,
    broadcast: str = "255.255.255.255",
    wake_timeout_s: int = 90,
    poll_interval_s: float = 3.0,
    gateway: str | None = None,
    gateway_cmd: str = "etherwake -i br-lan -b {mac}",
    gateway_ssh_key: str | None = None,
) -> bool:
    """确保 ``server_url`` 活着, 不活就 WoL 唤醒并等. 成功返回 True.

    优先级:
      - 配了 mac → 发直接 UDP 广播
      - 配了 gateway → 同时 SSH 过去跑 etherwake (绕开本机网卡 Hyper-V 拦截)
      - 都没配 → 只做一次探活, 不通直接返回 False
    """
    if is_alive(server_url):
        return True

    if not mac and not gateway:
        return False

    print(
        f"[wakeup] {server_url} 不通, 发 WoL → mac={mac} bcast={broadcast}"
        + (f" gateway={gateway}" if gateway else ""),
        file=sys.stderr, flush=True,
    )
    _emit(mac or "", broadcast, gateway, gateway_cmd, gateway_ssh_key)

    deadline = time.time() + wake_timeout_s
    last_resend = 0.0
    while time.time() < deadline:
        time.sleep(poll_interval_s)
        if is_alive(server_url):
            elapsed = wake_timeout_s - int(deadline - time.time())
            print(f"[wakeup] online after {elapsed}s", file=sys.stderr, flush=True)
            return True
        # 每 15s 再打一次 (有些设备需要多发)
        now = time.time()
        if now - last_resend > 15:
            print(f"[wakeup] still waiting... ({int(deadline - now)}s left)",
                  file=sys.stderr, flush=True)
            _emit(mac or "", broadcast, gateway, gateway_cmd, gateway_ssh_key)
            last_resend = now

    print(f"[wakeup] TIMEOUT after {wake_timeout_s}s, server still down",
          file=sys.stderr, flush=True)
    return False
