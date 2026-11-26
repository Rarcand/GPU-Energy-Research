"""Separate Turing/FP16 experiment. Never pool compiler versions implicitly."""
import argparse,json,random,subprocess,sys,zipfile,shutil,tempfile,time
from pathlib import Path
SHARED = Path(__file__).resolve().parents[1] / 'shared'
sys.path.insert(0, str(SHARED))
from timing_validation import registry,atomic,digest

def checkpoint_unit(output, store, expected):
    """Commit archive then hash receipt; incomplete copies are never resumable."""
    summary=json.loads(output.with_suffix('.summary.json').read_text())
    if not summary['complete'] or summary['rows']!=expected or digest(output)!=summary['output_sha256']:
        raise RuntimeError('Invalid checkpoint output')
    label=output.stem
    store.mkdir(parents=True,exist_ok=True)
    archive=store/(label+'.zip')
    receipt=store/(label+'.complete.json')
    if receipt.exists():
        info=json.loads(receipt.read_text())
        if digest(archive)!=info['archive_sha256']:raise RuntimeError('Persistent checkpoint corrupted')
        return
    with tempfile.TemporaryDirectory() as td:
        local=Path(td)/(label+'.zip')
        with zipfile.ZipFile(local,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for suffix in ('.jsonl','.summary.json','.protocol.json'):
                path=output if suffix=='.jsonl' else output.with_suffix(suffix)
                z.write(path,path.name)
        h=digest(local)
        tmp=archive.with_suffix('.zip.uploading')
        shutil.copyfile(local,tmp)
        if digest(tmp)!=h:raise RuntimeError('Persistent write verification failed')
        tmp.replace(archive)
        if digest(archive)!=h:raise RuntimeError('Persistent archive verification failed')
        atomic(receipt,dict(archive_sha256=h,rows=expected,output_sha256=summary['output_sha256']))
    print('Persistent checkpoint verified:',archive,flush=True)


def restore_unit(output, store, expected):
    archive=store/(output.stem+'.zip');receipt=store/(output.stem+'.complete.json')
    if not receipt.exists():return False
    info=json.loads(receipt.read_text())
    if info['rows']!=expected or digest(archive)!=info['archive_sha256']:
        raise RuntimeError('Persistent checkpoint integrity failure')
    names={output.stem+s for s in ('.jsonl','.summary.json','.protocol.json')}
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=names or z.testzip() is not None:raise RuntimeError('Invalid archive members')
        for name in names:
            target=output.parent/name
            data=z.read(name)
            if target.exists() and target.read_bytes()!=data:raise RuntimeError('Local/persistent checkpoint conflict')
            target.write_bytes(data)
    s=json.loads(output.with_suffix('.summary.json').read_text())
    if not s['complete'] or s['rows']!=expected or digest(output)!=s['output_sha256']:
        raise RuntimeError('Restored checkpoint verification failed')
    return True


def main(*, repeat=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=SHARED.parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    if repeat:
        p.add_argument('--pass-index',type=int,choices=(1,2),required=True)
    else:
        p.set_defaults(pass_index=0)
    a=p.parse_args()
    root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
    info="import torch,triton,json; p=torch.cuda.get_device_properties(0); print(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,gpu=p.name,capability=[p.major,p.minor])))"
    env=json.loads(subprocess.check_output([sys.executable,'-c',info],text=True).strip().splitlines()[-1])
    if env['capability']!=[7,5] or 'T4' not in env['gpu']:raise RuntimeError('Expected a T4 runtime')
    if env['torch'].split('+')[0]!='2.6.0' or env['triton']!='3.2.0' or env['cuda']!='12.4':raise RuntimeError('Unexpected isolated T4 software stack')
    units=registry(root,'T4');worker=SHARED/'timing_validation.py'
    sources={str(p.relative_to(root)):digest(p) for p in [worker,Path(__file__)]}
    sources.update({str(p.relative_to(root)):digest(p) for p in (SHARED/'kernels').glob('*.py')})
    sources.update({str(p.relative_to(root)):digest(p) for p,rows in units.values()})
    expected=sum(sum(r['dtype']=='float16' for r in rows) for path,rows in units.values())
    assert expected==15000
    manifest=dict(comparison_group='turing-fp16-torch26-triton32',environment=env,expected_rows=14 if a.smoke else expected,excluded_dtype='bfloat16: unsupported native T4 arithmetic',sources=sources,pass_index=a.pass_index,cross_compiler_pooling_allowed=False,limitation='Managed Colab VM; technical repeat; compiler differs from main study. Same candidate keys and timing/reference helpers, recorded actual input hashes.')
    m=a.output/'manifest.json'
    if m.exists() and json.loads(m.read_text())!=manifest:raise RuntimeError('Resume manifest changed')
    atomic(m,manifest)
    store=a.checkpoint_dir.resolve();store.mkdir(parents=True,exist_ok=True)
    persistent_manifest=store/'manifest.json'
    if persistent_manifest.exists() and json.loads(persistent_manifest.read_text())!=manifest:
        raise RuntimeError('Persistent resume manifest changed')
    atomic(persistent_manifest,manifest)
    if json.loads(persistent_manifest.read_text())!=manifest:raise RuntimeError('Persistent manifest write failed')
    order=[k for k in sorted(units) if not a.smoke or k[1]==0];random.Random(771903+a.pass_index).shuffle(order)
    try:
        for family,shard in order:
            for path,h in sources.items():
                if digest(root/path)!=h:raise RuntimeError('Source changed')
            output=a.output/('%s-%02d.jsonl'%(family,shard));summary=output.with_suffix('.summary.json')
            n=2 if a.smoke else sum(r['dtype']=='float16' for r in units[(family,shard)][1])
            if not summary.exists():restore_unit(output,store,n)
            if summary.exists():
                s=json.loads(summary.read_text())
                if not s['complete'] or s['rows']!=n or s['output_sha256']!=digest(output):raise RuntimeError('Corrupt completed checkpoint')
                checkpoint_unit(output,store,n)
                continue
            if output.exists():
                partial=a.output/'partial-attempts'/('%s-%d'%(output.stem,time.time_ns()));partial.mkdir(parents=True)
                for suffix in ('.jsonl','.protocol.json'):
                    old=output if suffix=='.jsonl' else output.with_suffix(suffix)
                    if old.exists():old.replace(partial/old.name)
            atomic(a.output/'status.json',dict(phase='running',family=family,shard=shard))
            cmd=[sys.executable,'-u',str(worker),'--root',str(root),'--device','T4','--pass-index',str(a.pass_index),'--family',family,'--shard',str(shard),'--output',str(output)]
            if a.smoke:cmd+=['--limit','--repetitions','8']
            subprocess.run(cmd,check=True)
            s=json.loads(summary.read_text());n=2 if a.smoke else sum(r['dtype']=='float16' for r in units[(family,shard)][1])
            if not s['complete'] or s['rows']!=n:raise RuntimeError('Incomplete unit')
            if a.smoke and s['counts'].get('failed',0):raise RuntimeError('T4 smoke failed; production blocked')
            checkpoint_unit(output,store,n)
            print('Published',family,shard,s['rows'],s['counts'],flush=True)
        atomic(a.output/'status.json',dict(phase='complete',expected_rows=manifest['expected_rows']))
        atomic(store/'status.json',dict(phase='complete',expected_rows=manifest['expected_rows']))
    except BaseException as exc:
        atomic(a.output/'status.json',dict(phase='failed',error=str(exc)));raise

if __name__=='__main__':main()
