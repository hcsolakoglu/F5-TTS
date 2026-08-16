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
    '/content/f5tts_fg_l4_v2',
    '--samples',
    '1000',
    '--fetch-rows',
    '5000',
    '--epochs',
    '1',
    '--candidates',
    ','.join([
        'eager',
        'compiled_dit_blocks_dynamic',
        'compiled_dit_blocks_fullgraph_dynamic',
        'compiled_branchless_fullgraph_autodynamic',
        'compiled_branchless_dynamic_no_autotune_pw',
        'compiled_branchless_reduce_overhead_markstep',
    ]),
]
print('RUNNER_CMD=' + ' '.join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
selector = selectors.DefaultSelector()
selector.register(proc.stdout, selectors.EVENT_READ)
started = time.monotonic()
while True:
    events = selector.select(timeout=30)
    if events:
        for key, _ in events:
            stream = cast(TextIO, key.fileobj)
            line = stream.readline()
            if line:
                print(line, end='', flush=True)
    else:
        print(f'RUNNER_HEARTBEAT elapsed_s={time.monotonic() - started:.1f} returncode={proc.poll()}', flush=True)
    if proc.poll() is not None:
        for line in proc.stdout:
            print(line, end='', flush=True)
        break
print(f'RUNNER_RETURN_CODE={proc.returncode}', flush=True)
raise SystemExit(proc.returncode)
