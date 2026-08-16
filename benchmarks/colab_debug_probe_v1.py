#!/usr/bin/env python3
import subprocess, sys
cmd=['/usr/bin/python3','/content/f5tts_real_trainer_bench/outputs/correctness_probe.py']
print('RUN', cmd, flush=True)
p=subprocess.run(cmd, text=True, capture_output=True)
print('STDOUT:\n'+p.stdout, flush=True)
print('STDERR:\n'+p.stderr, flush=True)
raise SystemExit(p.returncode)
