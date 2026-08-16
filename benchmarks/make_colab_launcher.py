#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path

payload_path = Path("benchmarks/colab_real_trainer_bench.py")
launcher_path = Path("benchmarks/colab_real_trainer_launcher_v6.py")
payload = payload_path.read_bytes()
encoded = base64.b64encode(payload).decode("ascii")
launcher = f'''#!/usr/bin/env python3
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

payload = base64.b64decode({encoded!r})
remote_script = Path("/content/colab_real_trainer_bench_v6_payload.py")
remote_script.write_bytes(payload)
print(f"PAYLOAD_WRITTEN={{remote_script}} bytes={{len(payload)}}", flush=True)
cmd = [sys.executable, str(remote_script)] + sys.argv[1:]
print("PAYLOAD_CMD=" + " ".join(cmd), flush=True)
raise SystemExit(subprocess.run(cmd).returncode)
'''
launcher_path.write_text(launcher, encoding="utf-8")
print(launcher_path, len(payload), len(launcher))
