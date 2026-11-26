"""Single-GPU lab collection; stop and quarantine on foreign CUDA activity.

Best-effort monitoring does not provide scheduler-enforced exclusivity.
"""
import argparse,fcntl,json,os,random,signal,subprocess,sys,time
from pathlib import Path
SHARED = Path(__file__).resolve().parents[1] / 'shared'
sys.path.insert(0, str(SHARED))
from timing_validation import registry,atomic,digest

def query(option,fields):
    r=subprocess.run(['nvidia-smi','--query-'+option+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    return [line.split(', ') for line in r.stdout.splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uuid',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--pass-index',type=int,choices=range(3),required=True);a=p.parse_args()
    a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.uuid,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    units=registry(a.root,'RTXA5000');worker=SHARED/'timing_validation.py'
    sources={str(worker):digest(worker),str(Path(__file__)):digest(Path(__file__))}
    sources.update({str(f):digest(f) for f in (SHARED/'kernels').glob('*.py')})
    sources.update({str(f):digest(f) for f,rows in units.values()})
    manifest=dict(device='RTXA5000',uuid=a.uuid,expected_rows=14 if a.smoke else 30000,pass_index=a.pass_index,sources=sources,environment='Shared native Linux; best-effort foreign-compute monitor, no reservation',poll_seconds=1)
    old=a.output/'manifest.json'
    if old.exists() and json.loads(old.read_text())!=manifest:raise RuntimeError('Resume manifest changed')
    atomic(old,manifest)
    def foreign(allowed=None):
        return [r for r in query('compute-apps','gpu_uuid,pid') if r[0]==a.uuid and (allowed is None or int(r[1])!=allowed)]
    child=None
    def stop(*args):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        quiet_start=time.monotonic()
        while time.monotonic()-quiet_start<30:
            gpu=next(r for r in query('gpu','uuid,utilization.gpu,memory.used') if r[0]==a.uuid)
            if foreign() or float(gpu[1])>5 or float(gpu[2])>128:raise RuntimeError('Selected GPU is not idle; no work started')
            time.sleep(2)
        order=[k for k in sorted(units) if not a.smoke or k[1]==0];random.Random(771903+a.pass_index).shuffle(order)
        for family,shard in order:
            for f,h in sources.items():
                if digest(Path(f))!=h:raise RuntimeError('Source changed')
            label='%s-%02d'%(family,shard);target=a.output/'published'/(label+'.jsonl');summary=target.with_suffix('.summary.json')
            expected=2 if a.smoke else len(units[(family,shard)][1])
            if summary.exists():
                s=json.loads(summary.read_text())
                if not s['complete'] or s['rows']!=expected or digest(target)!=s['output_sha256']:raise RuntimeError('Corrupt checkpoint')
                continue
            if foreign():raise RuntimeError('GPU became occupied before unit')
            attempt=a.output/'attempts'/('%s-%d'%(label,time.time_ns()));attempt.mkdir(parents=True)
            output=attempt/'measurements.jsonl'
            cmd=[sys.executable,'-u',str(worker),'--root',str(a.root),'--device','RTXA5000','--pass-index',str(a.pass_index),'--family',family,'--shard',str(shard),'--output',str(output)]
            if a.smoke:cmd+=['--limit','--repetitions','8']
            atomic(a.output/'status.json',dict(phase='running',unit=label,attempt=str(attempt)))
            print('Starting',label,flush=True)
            with (attempt/'stdout.log').open('w') as out,(attempt/'stderr.log').open('w') as err:
                child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
                try:
                    while child.poll() is None:
                        if foreign(child.pid):raise RuntimeError('Foreign GPU process detected')
                        time.sleep(1)
                    if foreign():raise RuntimeError('Foreign GPU process at unit end')
                except BaseException as exc:
                    stop();atomic(attempt/'QUARANTINED.json',dict(reason=str(exc)));raise
            if child.returncode:raise RuntimeError('Worker failed; inspect '+str(attempt))
            s=json.loads(output.with_suffix('.summary.json').read_text())
            if not s['complete'] or s['rows']!=expected:raise RuntimeError('Incomplete worker output')
            if a.smoke and s['counts'].get('failed',0):raise RuntimeError('Readiness candidate failed; production blocked')
            target.parent.mkdir(exist_ok=True)
            for suffix in ('.jsonl','.summary.json','.protocol.json'):
                src=output if suffix=='.jsonl' else output.with_suffix(suffix);dst=target if suffix=='.jsonl' else target.with_suffix(suffix)
                if dst.exists():raise FileExistsError(dst)
                src.replace(dst)
            print('Published',label,s['rows'],s['counts'],flush=True)
        atomic(a.output/'status.json',dict(phase='complete',expected_rows=manifest['expected_rows']))
    except BaseException as exc:
        stop();atomic(a.output/'status.json',dict(phase='stopped',error=str(exc)));raise

if __name__=='__main__':main()
