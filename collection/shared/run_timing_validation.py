#!/usr/bin/env python3
"""Checkpointed full-grid driver; optional quiet-Windows monitoring.

Completed units are immutable and checksum-verified. Interrupted or contaminated
units are preserved under attempts/, never counted as published measurements.
"""
import argparse,fcntl,json,os,random,signal,statistics,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from timing_validation import registry,FAMILIES,atomic,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--device',choices=('A100','H100','RTX3070-WSL'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--passes',type=int,nargs='+',default=[0,1,2],choices=range(3))
    p.add_argument('--desktop',action='store_true');p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--max-hours',type=float,default=72)
    p.add_argument('--shards',type=int,nargs='+',choices=range(10),default=list(range(10)))
    a=p.parse_args()
    if a.device=='RTX3070-WSL' and not a.desktop:
        p.error('RTX 3070 collection requires --desktop for host monitoring')
    if a.desktop and a.device!='RTX3070-WSL':
        p.error('--desktop applies to the RTX 3070')
    worker=Path(__file__).resolve().with_name('timing_validation.py')
    kernels=worker.parent/'kernels'
    code_paths=[Path(__file__).resolve(),worker]
    def interrupted(signum,frame):raise KeyboardInterrupt('Timing validation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    a.root=a.root.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    source=registry(a.root,a.device)
    lock=(a.output/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.desktop:
        desktop_dir=worker.parents[1]/'rtx-3070'
        sys.path.insert(0,str(desktop_dir))
        code_paths.extend(desktop_dir/name for name in ('collect_desktop_idle.py','desktop_p8_idle.py','windows_gpu_monitor.ps1'))
        from collect_desktop_idle import HostMonitor,stop_child
        from desktop_p8_idle import gpu_state_extended as gpu_state,idle_profile
        (a.root/'results/desktop-runs').mkdir(parents=True,exist_ok=True)
        gpu_lock=(a.root/'results/desktop-runs/gpu-collector.lock').open('w')
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['PATH']='/usr/lib/wsl/lib:'+os.environ.get('PATH','')
        h=a.root/'.build/python-headers/usr/include'
        os.environ['C_INCLUDE_PATH']=':'.join(str(p) for p in (h/'python3.12',h,h/'x86_64-linux-gnu/python3.12'))
        os.environ['TRITON_CACHE_DIR']=str(a.root/'.cache/triton-collector')
        monitor=HostMonitor(a.output)
    else:monitor=None
    manifest=dict(device=a.device,started_utc=datetime.now(timezone.utc).isoformat(),passes=a.passes,
        shards=a.shards,expected_candidates_per_pass=sum(len(rows) for (_,shard),(_,rows) in source.items() if shard in a.shards),repetitions_per_protocol=50,
        protocols=['individual_launch','adaptive_cuda_graph'],
        source_files={str(p.relative_to(a.root)):digest(p) for p,_ in source.values()},
        code={str(p.relative_to(a.root)):digest(p) for p in code_paths},
        frozen_kernels={p.name:digest(p) for p in kernels.glob('*.py')},
        idle_admission='p8-empty-host-engines-v1: 30s fresh empty Windows engine samples, P8, <=20W, <=300MHz, <=10% raw utilization, <=2048MiB; strict fallback. Foreign-app and stale-monitor rejection unchanged.',
        environment='Windows/WSL2 monitored desktop' if a.desktop else 'Slurm allocated native CUDA',
        limitation='Three process-separated passes; not three independent physical GPUs. Desktop pauses for foreign GPU activity; no apps are killed.')
    if not a.desktop:
        manifest.pop('idle_admission')
    manifest_path=a.output/('smoke-manifest.json' if a.smoke_only else 'manifest-pass%s.json'%'-'.join(map(str,a.passes)))
    if a.device=='H100' and not a.smoke_only:
        manifest_path=a.output/('manifest-pass%s-shards%s.json'%('-'.join(map(str,a.passes)),'-'.join(map(str,a.shards))))
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        for k in ('source_files','code','frozen_kernels'):
            if manifest[k]!=old[k]:raise RuntimeError('Resume source changed: '+k)
    else:atomic(manifest_path,manifest)
    deadline=time.monotonic()+a.max_hours*3600
    def state(phase,**kw):atomic(a.output/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw))
    last_idle={}
    def idle():
        if monitor is None:return None
        since=None;util=[];profiles=[];baseline=[]
        while time.monotonic()<deadline:
            if monitor.process.poll() is not None:raise RuntimeError('Host GPU monitor exited')
            good,reason,serial=monitor.check();gpu=gpu_state()
            profile=idle_profile(monitor.latest or {},gpu) if good else None
            if profile:
                if profiles and profiles[-1]!=profile:since=None;util=[];profiles=[];baseline=[]
                since=time.monotonic() if since is None else since
                util.append(gpu['utilization']);profiles.append(profile);baseline.append(gpu)
                limit=10 if profile=='p8-empty-host-engines-v1' else 2
                if time.monotonic()-since>=30 and statistics.median(util)<=limit:
                    last_idle.clear();last_idle.update(profile=profile,seconds=time.monotonic()-since,samples=baseline,host_counter_serial=serial)
                    return serial
            else:since=None;util=[];profiles=[];baseline=[]
            state('waiting_idle',reason=reason if profile else ('Idle admission not met: '+reason),gpu=gpu,idle_profile=profile)
            time.sleep(2)
        raise TimeoutError('Collection window expired while waiting for idle')
    current=None
    try:
        passes=[0] if a.smoke_only else a.passes
        for pass_index in passes:
            units=[(f,0) for f in FAMILIES] if a.smoke_only else sorted(k for k in source if k[1] in a.shards)
            random.Random(771903+pass_index).shuffle(units)
            for family,shard in units:
                label='%s-%02d'%(family,shard)
                directory=a.output/('smoke' if a.smoke_only else 'pass-%02d'%pass_index)
                target=directory/(label+'.jsonl');summary=target.with_suffix('.summary.json')
                if summary.exists():
                    s=json.loads(summary.read_text())
                    expected=2 if a.smoke_only else len(source[(family,shard)][1])
                    if not s['complete'] or s['rows']!=expected or digest(target)!=s['output_sha256']:
                        raise RuntimeError('Corrupt completed unit: '+label)
                    continue
                while time.monotonic()<deadline:
                    serial=idle()
                    attempt=a.output/'attempts'/('%s-p%d-%d'%(label,pass_index,time.time_ns()))
                    attempt.mkdir(parents=True)
                    if monitor:atomic(attempt/'idle-admission.json',last_idle)
                    output=attempt/'measurements.jsonl'
                    for path,expected_hash in manifest['code'].items():
                        if digest(a.root/path)!=expected_hash:raise RuntimeError('Code changed during run')
                    command=[sys.executable,'-u',str(worker),
                        '--root',str(a.root),'--device',a.device,'--pass-index',str(pass_index),
                        '--family',family,'--shard',str(shard),'--output',str(output)]
                    if a.smoke_only:command+=['--limit','--repetitions','8']
                    state('running',pass_index=pass_index,unit=label,attempt=str(attempt))
                    print('Starting',pass_index,label,flush=True)
                    contaminated=None
                    with (attempt/'stdout.log').open('w') as stdout,(attempt/'stderr.log').open('w') as stderr:
                        current=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
                        while current.poll() is None:
                            if monitor:
                                good,reason,_=monitor.check(serial)
                                if not good:
                                    contaminated=reason;stop_child(current);break
                            time.sleep(1)
                        if monitor:
                            time.sleep(3);good,reason,_=monitor.check(serial)
                            if not good:contaminated=reason
                    if contaminated:
                        atomic(attempt/'QUARANTINED.json',dict(reason=contaminated))
                        print('Quarantined',label,contaminated,flush=True);continue
                    if current.returncode:
                        raise RuntimeError('Unit failed (%d), inspect %s'%(current.returncode,attempt/'stderr.log'))
                    if monitor:
                        protocol_path=output.with_suffix('.protocol.json')
                        protocol=json.loads(protocol_path.read_text());protocol['desktop_idle_admission']=dict(profile=last_idle['profile'],baseline_file=str(attempt/'idle-admission.json'),driver=Path(__file__).name)
                        atomic(protocol_path,protocol)
                    completed=json.loads(output.with_suffix('.summary.json').read_text())
                    expected=2 if a.smoke_only else len(source[(family,shard)][1])
                    if not completed['complete'] or completed['rows']!=expected:raise RuntimeError('Incomplete unit')
                    if a.smoke_only and completed['counts'].get('failed',0):raise RuntimeError('Smoke candidate failed: '+label)
                    directory.mkdir(parents=True,exist_ok=True)
                    for suffix in ('.jsonl','.protocol.json','.summary.json'):
                        src=output if suffix=='.jsonl' else output.with_suffix(suffix)
                        dst=target if suffix=='.jsonl' else target.with_suffix(suffix)
                        if dst.exists():raise FileExistsError(dst)
                        src.replace(dst)
                    print('Published',pass_index,label,completed['counts'],'seconds',round(completed['seconds'],1),flush=True)
                    break
                else:raise TimeoutError('Collection window expired')
        state('complete',passes=passes,shards=a.shards,smoke_only=a.smoke_only)
    except BaseException as exc:
        state('failed',error=str(exc));raise
    finally:
        if current is not None and current.poll() is None:
            if monitor:stop_child(current)
            else:
                os.killpg(current.pid,signal.SIGTERM);current.wait()
        if monitor:stop_child(monitor.process)

if __name__=='__main__':main()
