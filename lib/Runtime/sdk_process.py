"""Bound SDK jobs and surface simulator fatal diagnostics without a hanging receive."""

import os
import json
from pathlib import Path
import signal
import subprocess
import time


def run_sdk(command, directory, env, log, timeout, *, progress_check=None):
    if timeout <= 0:
        raise ValueError("SDK timeout must be positive")
    started = time.monotonic()

    def stage(state, detail=None):
        (Path(directory) / "execution-stage.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "elapsed_seconds": time.monotonic() - started,
                    "detail": detail,
                }
            )
            + "\n"
        )

    stage("sdk_running")
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if progress_check is not None:
                progress_check()
            sim = Path(directory) / "sim.log"
            if sim.exists():
                with sim.open("rb") as stream:
                    stream.seek(max(0, sim.stat().st_size - 131072))
                    tail = stream.read().decode(errors="replace")
                fatal = next(
                    (
                        line
                        for line in tail.splitlines()
                        if "FATAL:" in line or "Stopping due to fatal error" in line
                    ),
                    None,
                )
                if fatal:
                    raise RuntimeError("SDK simulator fatal; inspect sim.log: " + fatal)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"SDK job exceeded {timeout}s; preserved {directory}"
                )
            time.sleep(0.2)
        if progress_check is not None:
            progress_check()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        stage("sdk_finished")
    except BaseException as error:
        stage("sdk_failed", str(error))
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
