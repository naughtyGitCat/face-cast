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

import socket
import sys
import time
import urllib.error
import urllib.request


def send_wol(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> None:
    """发 magic packet 唤醒指定 MAC 的机器.

    magic packet = 6 字节 0xFF + 16 次重复 MAC = 102 字节, 走 UDP 广播.

    ``broadcast``: 子网定向广播 (e.g. ``10.100.100.255``) 比 limited
    broadcast (``255.255.255.255``) 更可靠 — 路由器一般不转发后者.
    """
    mac_clean = mac.replace(":", "").replace("-", "").strip()
    if len(mac_clean) != 12:
        raise ValueError(f"bad MAC: {mac!r}")
    try:
        mac_bytes = bytes.fromhex(mac_clean)
    except ValueError as e:
        raise ValueError(f"bad MAC hex: {mac!r}") from e

    packet = b"\xff" * 6 + mac_bytes * 16
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # 同时打两个常见 WoL 端口, 部分设备只听其中一个
        for p in (port, 7):
            try:
                s.sendto(packet, (broadcast, p))
            except OSError:
                pass
    finally:
        s.close()


def is_alive(server_url: str, timeout: float = 2.0) -> bool:
    """GET ``<server_url>/health``, 超时/连接失败/非 200 都算 down."""
    url = server_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def ensure_alive(
    server_url: str,
    *,
    mac: str | None = None,
    broadcast: str = "255.255.255.255",
    wake_timeout_s: int = 90,
    poll_interval_s: float = 3.0,
) -> bool:
    """确保 ``server_url`` 活着, 不活就 WoL 唤醒并等. 成功返回 True.

    没配 mac → 不发 WoL, 仅做一次探活.
    """
    if is_alive(server_url):
        return True

    if not mac:
        return False

    print(
        f"[wakeup] {server_url} 不通, 发 WoL → mac={mac} bcast={broadcast}",
        file=sys.stderr, flush=True,
    )
    send_wol(mac, broadcast)

    deadline = time.time() + wake_timeout_s
    last_log = 0.0
    while time.time() < deadline:
        time.sleep(poll_interval_s)
        if is_alive(server_url):
            elapsed = wake_timeout_s - int(deadline - time.time())
            print(f"[wakeup] online after {elapsed}s", file=sys.stderr, flush=True)
            return True
        # 每 15s 再发一次 magic packet (有些设备需要多发)
        now = time.time()
        if now - last_log > 15:
            print(f"[wakeup] still waiting... ({int(deadline - now)}s left)",
                  file=sys.stderr, flush=True)
            send_wol(mac, broadcast)
            last_log = now

    print(f"[wakeup] TIMEOUT after {wake_timeout_s}s, server still down",
          file=sys.stderr, flush=True)
    return False
