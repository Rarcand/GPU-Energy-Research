#!/usr/bin/env python3
"""Isolated, staged H100 probes. Never changes devices or terminates other users.

Only run main inside a gpu-h100 allocation. Driver probes retain/release their
own primary context and 4 KiB allocation. A healthy allocated UUID is selected
explicitly; import timeouts do not become false hardware diagnoses.
"""
import argparse,csv,ctypes,hashlib,json,os,signal,socket,stat,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def inherited_devices(devices,visibility):
    if visibility is None:return devices
    tokens=[v.strip() for v in visibility.split(',') if v.strip()]
    return [d for d in devices if any(t==d['index'] or d['uuid'].startswith(t) for t in tokens)]


def phase(stage_name,**kw):print(json.dumps(dict(stage=stage_name,**kw)),flush=True)


def child(mode):
    if mode=='driver':
        phase('load_libcuda')
        cuda=ctypes.CDLL('libcuda.so.1')
        def check(name,*args):
            code=getattr(cuda,name)(*args)
            p=ctypes.c_char_p();cuda.cuGetErrorName(code,ctypes.byref(p))
            phase(name,code=code,error=p.value.decode() if p.value else '')
            if code:raise RuntimeError(name+' failed: '+str(code))
        check('cuInit',0)
        device=ctypes.c_int();check('cuDeviceGet',ctypes.byref(device),0)
        ctx=ctypes.c_void_p();check('cuDevicePrimaryCtxRetain',ctypes.byref(ctx),device)
        try:
            check('cuCtxSetCurrent',ctx)
            pointer=ctypes.c_uint64();check('cuMemAlloc_v2',ctypes.byref(pointer),ctypes.c_size_t(4096))
            check('cuMemFree_v2',pointer)
        finally:check('cuDevicePrimaryCtxRelease_v2' if hasattr(cuda,'cuDevicePrimaryCtxRelease_v2') else 'cuDevicePrimaryCtxRelease',device)
        phase('driver_probe_pass')
    else:
        phase('import_torch')
        import torch
        phase('torch_imported',version=torch.__version__,cuda=torch.version.cuda,torch_file=torch.__file__)
        if mode=='import':return
        phase('cuda_init');torch.cuda.init();torch.cuda.set_device(0)
        props=torch.cuda.get_device_properties(0)
        phase('device_initialized',name=props.name,capability=[props.major,props.minor])
        if 'H100' not in props.name:raise ValueError('Expected H100')
        torch.set_num_threads(1)
        for name in ('float16','bfloat16'):
            a=torch.ones((64,64),device='cuda',dtype=getattr(torch,name));b=a@a
            torch.cuda.synchronize();torch.testing.assert_close(b,torch.full_like(b,64),atol=0,rtol=0)
            phase('dtype_pass',dtype=name)
        phase('torch_probe_pass',free_total=list(torch.cuda.mem_get_info()))
    if Path('/proc/self/maps').exists():
        phase('loaded_cuda_libraries',paths=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines()
            if '/' in l and any(s in l for s in ('libcuda.so','libcudart.so','libnvidia-ptxjitcompiler'))}))


def run_child(python,mode,out,env,timeout):
    stdout=out.with_suffix('.out');stderr=out.with_suffix('.err')
    command=[python,'-u',str(Path(__file__).resolve()),'--child',mode]
    tick=time.monotonic();timed_out=False
    with stdout.open('w') as a,stderr.open('w') as b:
        process=subprocess.Popen(command,stdout=a,stderr=b,env=env,start_new_session=True)
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    stages=[]
    for line in stdout.read_text().splitlines():
        try:stages.append(json.loads(line))
        except ValueError:pass
    return dict(returncode=process.returncode,timed_out=timed_out,seconds=time.monotonic()-tick,
                stages=stages,stderr_tail=stderr.read_text()[-6000:],stdout_file=str(stdout),stderr_file=str(stderr))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',choices=('import','driver','torch'))
    p.add_argument('--python');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.child:child(a.child);return
    if not a.python or not a.output:p.error('--python and --output required')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='gpu-h100':
        p.error('Must run inside allocated gpu-h100 batch job')
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inherited_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        environment={k:os.environ.get(k) for k in ('LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME','VIRTUAL_ENV')},probes=[])
    def state(stage):
        phase(stage);report['stage']=stage;atomic(a.output/'health.json',report)
    state('nvidia_smi_inventory')
    smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_mode,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
    report['nvidia_smi']=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr)
    if smi.returncode:raise RuntimeError('nvidia-smi inventory failed')
    names=('index','uuid','name','driver','compute_mode','memory_used_mb','memory_total_mb','utilization_pct')
    devices=[dict(zip(names,[v.strip() for v in row])) for row in csv.reader(smi.stdout.splitlines())]
    devices=inherited_devices(devices,report['inherited_visibility'])
    for label,command in [('nvidia_smi_details',['nvidia-smi','-q']),
                          ('compute_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'])]:
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            report[label]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        except subprocess.TimeoutExpired:report[label]=dict(timed_out=True)
    report['allocated_visible_devices']=devices
    report['device_access']={}
    for path in ('/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools',*('/dev/nvidia'+d['index'] for d in devices)):
        try:
            st=os.stat(path);report['device_access'][path]=dict(mode=oct(stat.S_IMODE(st.st_mode)),uid=st.st_uid,gid=st.st_gid,
                readable=os.access(path,os.R_OK),writable=os.access(path,os.W_OK))
        except OSError as e:report['device_access'][path]=dict(error=str(e))
    state('python_import_preflight')
    imported=run_child(a.python,'import',a.output/'import',dict(os.environ),600)
    report['import_probe']=imported
    if imported['returncode']:
        report['failure_class']='python_import_timeout' if imported['timed_out'] else 'python_import_failure'
        state('failed');raise SystemExit(40)
    for device in devices:
        if 'H100' not in device['name']:continue
        env={**os.environ,'CUDA_VISIBLE_DEVICES':device['uuid'],'CUDA_LAUNCH_BLOCKING':'1'}
        state('driver_probe_gpu_'+device['index'])
        driver=run_child(a.python,'driver',a.output/('gpu-'+device['index']+'-driver'),env,90)
        probe=dict(device=device,driver=driver)
        state('torch_probe_gpu_'+device['index'])
        probe['torch']=run_child(a.python,'torch',a.output/('gpu-'+device['index']+'-torch'),env,300)
        # Application-level FP16/BF16 success certifies usability even if an
        # optional ctypes diagnostic lacks a driver API symbol.
        probe['healthy']=probe['torch']['returncode']==0
        stages=[s['stage'] for s in probe['torch']['stages']]
        if probe['healthy']:probe['classification']='verified_fp16_bf16_cuda'
        elif 'TypeError:' in probe['torch']['stderr_tail']:
            probe['classification']='diagnostic_python_failure'
        elif 'cuda_init' not in stages:
            probe['classification']='python_import_timeout' if probe['torch']['timed_out'] else 'python_import_failure'
        else:probe['classification']='cuda_probe_timeout' if probe['torch']['timed_out'] else 'cuda_probe_failure'
        report['probes'].append(probe);state('probed_gpu_'+device['index'])
    healthy=[p for p in report['probes'] if p['healthy']]
    if not healthy:
        report['failure_class']='no_verified_usable_allocated_h100';state('failed');raise SystemExit(42)
    selected=healthy[0]['device'];report['selected_device']=selected
    report['selection_reason']='First healthy inherited allocated H100; failed probes remain in health report.'
    (a.output/'selected-uuid.txt').write_text(selected['uuid']+'\n')
    state('ready')

if __name__=='__main__':main()
