#!/usr/bin/env python3
from __future__ import annotations

import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO, cast

payload = Path('/content/f5tts_fullgraph_speedup_bench.py')
if not payload.exists():
    raise SystemExit(f'MISSING_PAYLOAD={payload}')

cmd = [
    sys.executable,
    str(payload),
    '--work-dir',
    '/content/f5tts_no_autotune_l4_v1',
    '--samples',
    '1000',
    '--fetch-rows',
    '5000',
    '--epochs',
    '1',
    '--candidates',
    'compiled_branchless_dynamic_no_autotune_pw',
]
print('RUNNER_CMD=' + ' '.join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
selector = selectors.DefaultSelector()
selector.register(proc.stdout, selectors.EVENT_READ)
started = time.monotonic()
# Single-candidate run. If it cannot finish in ~45 min, it is not a wall-time candidate.
timeout_s = 2700
while True:
    events = selector.select(timeout=30)
    if events:
        for key, _ in events:
            stream = cast(TextIO, key.fileobj)
            line = stream.readline()
            if line:
                print(line, end='', flush=True)
    else:
        elapsed = time.monotonic() - started
        print(f'RUNNER_HEARTBEAT elapsed_s={elapsed:.1f} returncode={proc.poll()}', flush=True)
        if elapsed > timeout_s and proc.poll() is None:
            print(f'RUNNER_TIMEOUT_KILL elapsed_s={elapsed:.1f}', flush=True)
            proc.kill()
    if proc.poll() is not None:
        for line in proc.stdout:
            print(line, end='', flush=True)
        break
print(f'RUNNER_RETURN_CODE={proc.returncode}', flush=True)
raise SystemExit(proc.returncode)
