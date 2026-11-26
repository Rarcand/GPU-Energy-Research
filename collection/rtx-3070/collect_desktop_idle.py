#!/usr/bin/env python3
"""Run the frozen grid only during a monitored quiet desktop session.

No applications are closed. A contaminated family/shard attempt is terminated,
quarantined and retried after another idle interval. Only owned child process
groups are terminated. WSL is recorded as a separate execution environment.
"""
import csv
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

SMI = '/usr/lib/wsl/lib/nvidia-smi'
POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
# WSL computation is accounted under system/VM processes. Retain every raw
# counter sample; these exclusions do not establish bare-metal exclusivity.
SYSTEM_PROCESSES = {'system','vmmem','vmmemwsl','wslhost','wslservice'}


def foreign_activity(sample):
    result=[]
    for active in sample.get('active',[]):
        name=str(active.get('process') or 'unknown').lower()
        if name in SYSTEM_PROCESSES:
            continue
        # Only exempt a VM worker whose GUID the host monitor independently
        # matched to a running WSL compute system. Other VMs remain foreign.
        if (name=='vmwp' and active.get('wsl_vm_id')
                and active['wsl_vm_id'] in sample.get('verified_wsl_vm_ids',[])):
            continue
        # Small desktop-compositor activity is unavoidable on a display GPU.
        if name=='dwm' and active['utilization']<=3:
            continue
        result.append(active)
    return result


def gpu_state():
    result=subprocess.run([SMI,
        '--query-gpu=utilization.gpu,memory.used,temperature.gpu,pstate',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr)
    fields=next(csv.reader(result.stdout.splitlines()))
    return dict(utilization=float(fields[0]),memory_used_mib=float(fields[1]),
                temperature=float(fields[2]),pstate=fields[3].strip())


def quiet(sample,state):
    # Windows retains display/driver buffers even with zero active GPU work.
    # Keep at least 6 GiB available rather than requiring an empty framebuffer.
    return not foreign_activity(sample) and state['utilization']<=5 and state['memory_used_mib']<=2048


class HostMonitor:
    def __init__(self,root):
        self.latest=None
        self.updated=0
        self.events=[]
        self.serial=0
        self.lock=threading.Lock()
        self.output=(root/'host-gpu-activity.jsonl').open('a',buffering=1)
        self.errors=(root/'host-monitor.err').open('a',buffering=1)
        script=(Path(__file__).with_name('windows_gpu_monitor.ps1')).read_text()
        self.process=subprocess.Popen([POWERSHELL,'-NoProfile','-NonInteractive',
            '-Command',script],stdout=subprocess.PIPE,stderr=self.errors,
            text=True,start_new_session=True)
        threading.Thread(target=self.consume,daemon=True).start()

    def consume(self):
        for line in self.process.stdout:
            try:
                sample=json.loads(line.strip())
                if not sample.get('sample_ok'):
                    continue
            except ValueError:
                continue
            with self.lock:
                self.serial+=1
                self.latest=sample
                self.updated=time.monotonic()
                self.output.write(json.dumps(dict(serial=self.serial,**sample))+'\n')
                if foreign_activity(sample):
                    self.events.append(self.serial)

    def check(self,since=None):
        with self.lock:
            if self.latest is None or time.monotonic()-self.updated>15:
                return False,'Windows GPU monitor is not current',self.serial
            if foreign_activity(self.latest):
                names=sorted(set(str(p.get('process') or 'unknown') for p in foreign_activity(self.latest)))
                return False,'Active Windows GPU apps: '+', '.join(names),self.serial
            if since is not None and any(i>since for i in self.events):
                return False,'Windows GPU activity occurred during measurement',self.serial
            return True,'quiet',self.serial


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
