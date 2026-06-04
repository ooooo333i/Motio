"""단일 GPU 직렬화 락 + 공통 async subprocess 실행기.

검출/추론은 모두 이 락 아래에서 한 번에 하나만 실행된다(단일 GPU 안정).
서브프로세스 stdout/stderr 를 로그파일로 흘리며 전체를 문자열로 반환.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

# 프로세스 전역 GPU 직렬화 락 (검출·추론 공유)
GPU_LOCK = asyncio.Lock()


async def run_logged(
    cmd: list[str],
    log_path: str | Path | None,
    cwd: str | Path,
    env: dict | None = None,
) -> tuple[int, str]:
    """cmd 실행. (returncode, 전체 출력) 반환. log_path 지정 시 append 로 기록."""
    env2 = dict(os.environ)
    env2.setdefault("PYOPENGL_PLATFORM", "")  # HaMeR egl 우회
    if env:
        env2.update(env)

    proc = await asyncio.create_subprocess_exec(
        *[str(c) for c in cmd],
        cwd=str(cwd),
        env=env2,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    chunks: list[bytes] = []
    log_file = open(log_path, "ab") if log_path else None
    try:
        assert proc.stdout is not None
        async for line in proc.stdout:
            # 반환 문자열(chunks)은 원본 유지 — manifest 의 "[OPT_MANIFEST]" 라인
            # 머리부 파싱이 깨지지 않도록 타임스탬프는 로그파일에만 붙인다(병목 분석용).
            chunks.append(line)
            if log_file:
                ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
                log_file.write(f"[{ts}] ".encode() + line)
                log_file.flush()
        await proc.wait()
    finally:
        if log_file:
            log_file.close()

    return proc.returncode or 0, b"".join(chunks).decode("utf-8", "replace")
