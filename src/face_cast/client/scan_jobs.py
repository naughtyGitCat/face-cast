"""扫描/聚类后台任务运行器.

设计:
  - subprocess.Popen 起独立子进程, 不阻塞 web 进程
  - stdout+stderr 重定向到 ``<jobs_dir>/<job_id>.log``
  - 元数据 (cmd/状态/时间/退出码) 落 ``<jobs_dir>/<job_id>.json``
  - JobManager 内存里维护 ``Popen`` 句柄; UI 重启句柄丢失, 但靠 pid + log
    文件状态可以推出来 (psutil 不依赖, 用 ``os.kill(pid, 0)`` 探活)
  - cluster job 同时只允许一个 (HDBSCAN 全局算)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock


def _is_pid_alive(pid: int) -> bool:
    """跨平台探活."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # Windows: subprocess kill query via WMIC 慢, 简化用 OpenProcess
            import ctypes  # noqa: PLC0415
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == STILL_ACTIVE
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


@dataclass
class JobInfo:
    id: str
    kind: str                # "detect" | "cluster"
    target: str              # detect 是 path; cluster 是 ""
    cmd: list[str]           # for display
    status: str              # "running" | "done" | "failed" | "stale"
    started_at: float
    ended_at: float | None = None
    returncode: int | None = None
    pid: int | None = None
    log_path: str = ""

    @property
    def elapsed(self) -> float:
        end = self.ended_at if self.ended_at else time.time()
        return round(end - self.started_at, 1)


class JobManager:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._meta: dict[str, JobInfo] = {}
        self._lock = Lock()
        self._load_existing()

    # ─── 持久化 ───────────────────────────────────────────────────────

    def _meta_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _log_path_for(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    def _save(self, job: JobInfo) -> None:
        self._meta_path(job.id).write_text(
            json.dumps(asdict(job), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_existing(self) -> None:
        for p in sorted(self.jobs_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                info = JobInfo(**d)
                # 跨重启对账: running 状态的 job 看 pid 是否还活
                if info.status == "running":
                    if info.pid and _is_pid_alive(info.pid):
                        # 进程还活着, 但我们丢了 Popen, 状态保留 running
                        # 之后 reap() 探活拿不到 returncode, 就让它 stale
                        pass
                    else:
                        info.status = "stale"
                        info.ended_at = info.ended_at or time.time()
                        self._save(info)
                self._meta[info.id] = info
            except Exception:  # noqa: BLE001
                continue

    # ─── 启动 ─────────────────────────────────────────────────────────

    def _spawn(
        self,
        *,
        kind: str,
        target: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
    ) -> JobInfo:
        job_id = uuid.uuid4().hex[:8]
        log = self._log_path_for(job_id)
        log_fp = log.open("wb", buffering=0)

        full_env = os.environ.copy()
        full_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            full_env.update(env)

        # Windows 下加 CREATE_NEW_PROCESS_GROUP, 服务进程被 NSSM stop 时
        # 不会顺带把子进程杀掉
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=full_env,
            creationflags=creationflags,
        )

        info = JobInfo(
            id=job_id,
            kind=kind,
            target=target,
            cmd=cmd,
            status="running",
            started_at=time.time(),
            pid=proc.pid,
            log_path=str(log),
        )
        with self._lock:
            self._procs[job_id] = proc
            self._meta[job_id] = info
            self._save(info)
        return info

    def start_extract(
        self,
        *,
        face_cast_bin: Path,
        path: str,
        db_path: Path,
        server_url: str,
        frames: int = 15,
    ) -> JobInfo:
        """跑 ``face-cast extract <path>`` (per-folder 抽帧 + face embedding).

        注意 CLI 命令叫 ``extract``, 不是 ``detect``. CLI 里 ``detect`` 实际是
        HDBSCAN 聚类那一步 (历史命名遗留, 别迷惑).
        """
        cmd = [
            str(face_cast_bin), "extract", path,
            "--db", str(db_path),
            "--server", server_url,
            "--frames", str(frames),
        ]
        return self._spawn(kind="extract", target=path, cmd=cmd)

    def start_cluster(
        self,
        *,
        face_cast_bin: Path,
        db_path: Path,
        server_url: str,
    ) -> JobInfo | None:
        """跑 ``face-cast detect`` (全局 HDBSCAN 聚类). 同时只允许一个."""
        with self._lock:
            for j in self._meta.values():
                if j.kind == "cluster" and j.status == "running":
                    return None  # 已有, 拒绝
        cmd = [
            str(face_cast_bin), "detect",
            "--db", str(db_path),
            "--server", server_url,
        ]
        return self._spawn(kind="cluster", target="", cmd=cmd)

    # ─── 收尸 ─────────────────────────────────────────────────────────

    def reap(self) -> None:
        """对所有内存里的 Popen 探活, 落地 ended_at/returncode."""
        with self._lock:
            done = []
            for jid, proc in self._procs.items():
                rc = proc.poll()
                if rc is not None:
                    info = self._meta.get(jid)
                    if info is None:
                        continue
                    info.status = "done" if rc == 0 else "failed"
                    info.returncode = rc
                    info.ended_at = time.time()
                    self._save(info)
                    done.append(jid)
            for jid in done:
                self._procs.pop(jid, None)

    # ─── 查询 ─────────────────────────────────────────────────────────

    def list_jobs(self, *, limit: int = 50) -> list[JobInfo]:
        self.reap()
        with self._lock:
            jobs = sorted(self._meta.values(),
                          key=lambda j: j.started_at, reverse=True)
        return jobs[:limit]

    def get(self, job_id: str) -> JobInfo | None:
        self.reap()
        return self._meta.get(job_id)

    def has_running(self) -> bool:
        self.reap()
        with self._lock:
            return any(j.status == "running" for j in self._meta.values())

    @staticmethod
    def _normalize_log_text(text: str) -> str:
        """规范化子进程日志: 把 \\r 提升成换行 (rich 进度条在非 TTY 下会用 \\r
        覆盖同一行) + 去 ANSI 控制序列 + 折叠连续重复行 (避免百行同样的进度
        条占屏).
        """
        import re  # noqa: PLC0415
        # 常见 ANSI: ESC [ ... letter 和 ESC [ ? ... letter
        text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        out: list[str] = []
        prev: str | None = None
        for line in text.split("\n"):
            if line != prev:
                out.append(line)
            prev = line
        return "\n".join(out)

    def tail_log(self, job_id: str, *, lines: int = 60) -> str:
        log = self._log_path_for(job_id)
        if not log.exists():
            return ""
        try:
            text = log.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return ""
        all_lines = self._normalize_log_text(text).split("\n")
        return "\n".join(all_lines[-lines:])

    def read_log_full(self, job_id: str) -> str:
        log = self._log_path_for(job_id)
        if not log.exists():
            return ""
        try:
            return self._normalize_log_text(
                log.read_bytes().decode("utf-8", errors="replace")
            )
        except OSError:
            return ""
