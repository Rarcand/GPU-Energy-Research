"""Explicit low-clock idle admission; measurement monitoring is unchanged.

Used only by the separately versioned timing driver. Raw NVIDIA counters remain
raw. P8 utilization is relative to the current low clock, so qualify a quiet
baseline using independent Windows engines, power and clock constraints.
"""
import csv,subprocess
from collect_desktop_idle import SMI,foreign_activity,quiet


def gpu_state_extended():
    result=subprocess.run([SMI,'--query-gpu=utilization.gpu,memory.used,temperature.gpu,pstate,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    f=next(csv.reader(result.stdout.splitlines()))
    return dict(utilization=float(f[0]),memory_used_mib=float(f[1]),temperature=float(f[2]),pstate=f[3].strip(),power_draw_w=float(f[4]),sm_clock_mhz=float(f[5]))


def idle_profile(sample,state):
    if not sample.get('sample_ok') or foreign_activity(sample) or state['memory_used_mib']>2048:return None
    if (not sample.get('active') and state.get('pstate')=='P8'
        and state.get('power_draw_w',float('inf'))<=20
        and state.get('sm_clock_mhz',float('inf'))<=300
        and state['utilization']<=10):return 'p8-empty-host-engines-v1'
    return 'original-strict' if quiet(sample,state) else None
