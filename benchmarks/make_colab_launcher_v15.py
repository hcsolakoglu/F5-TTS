#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path

payload_path = Path("benchmarks/colab_real_trainer_bench.py")
launcher_path = Path("benchmarks/colab_real_trainer_launcher_v15.py")
payload = payload_path.read_bytes()
encoded = base64.b64encode(payload).decode("ascii")
launcher = f'''#!/usr/bin/env python3
from __future__ import annotations

import base64
import selectors
import subprocess
import sys
import time
from pathlib import Path

payload = base64.b64decode({encoded!r})
remote_script = Path("/content/colab_real_trainer_bench_v15_payload.py")
remote_script.write_bytes(payload)
print(f"PAYLOAD_WRITTEN={{remote_script}} bytes={{len(payload)}}", flush=True)
cmd = [sys.executable, str(remote_script)] + sys.argv[1:]
print("PAYLOAD_CMD=" + " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
sel = selectors.DefaultSelector()
sel.register(proc.stdout, selectors.EVENT_READ)
start = time.monotonic()
while True:
    events = sel.select(timeout=30)
    if events:
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                print(line, end="", flush=True)
    else:
        now = time.monotonic()
        print(f"PAYLOAD_HEARTBEAT elapsed_s={{now - start:.1f}} returncode={{proc.poll()}}", flush=True)
    if proc.poll() is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
        break
return_code = proc.returncode
print(f"PAYLOAD_RETURN_CODE={{return_code}}", flush=True)
raise SystemExit(return_code)
'''
launcher_path.write_text(launcher, encoding="utf-8")
print(launcher_path, len(payload), len(launcher))
