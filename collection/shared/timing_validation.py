#!/usr/bin/env python3
"""Full-grid, separate-regime timing validation. GPU execution requires allocation.

Every supported source key is attempted, including previously rejected keys.
The T4 runs the FP16 subset of the matched registry.
Inputs are regenerated from a documented per-workload seed, shared across
configurations and passes. Original CSVs and historical latency are immutable.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

FAMILIES=('gemm','softmax','cross_entropy','conv2d','attention','gather','residual_layernorm')
EXPECTED=dict(zip(FAMILIES,(7200,3408,3408,4992,4000,4000,2992)))


GPU_NAMES = {
    "A100": "A100", "H100": "H100", "RTX3070-WSL": "RTX 3070",
    "RTXA5000": "RTX A5000", "T4": "T4",
}
# The H100, A5000 and T4 studies reused the A100 candidate registry. Its
# validity and timing columns describe the source row, never the new GPU.
REUSED_REGISTRY = {"H100", "RTXA5000", "T4"}
BUFFER_BYTES = 1024 * 1024


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def registry(root,device):
    if device not in GPU_NAMES:
        raise ValueError(f"Unknown GPU: {device}")
    if device == "RTX3070-WSL":
        slug = "nvidia_geforce_rtx_3070_wsl"
        pattern = "*-matched30k-rtx3070-wsl-v2-*.csv"
    else:
        slug = "nvidia_a100-pcie-40gb"
        pattern = "*-matched30k-*.csv"
    paths = sorted((root / "results" / slug).glob(pattern))
    if not paths:
        # A clean checkout carries only the matched grid; partial production
        # directories must fail validation rather than mix registry sources.
        paths = sorted((root / "collection" / "grid").glob("*-matched30k-*.csv"))
    if len(paths) != 70:
        raise ValueError("Require all 70 original production files")
    units = {}
    counts = defaultdict(int)
    keys = set()
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty registry file: {path}")
        family = rows[0]["kernel_family"]
        shard = int(path.stem.rsplit("-", 1)[1])
        if family not in FAMILIES or shard not in range(10) or (family, shard) in units:
            raise ValueError(f"Unexpected or duplicate family/shard: {path}")
        for row in rows:
            key = (row["kernel_family"], row["workload_id"], row["config_id"])
            if key in keys or key[0] != family:
                raise ValueError(f"Duplicate key or mixed family in {path}")
            keys.add(key)
            counts[family] += 1
        units[family, shard] = path, rows
    if dict(counts) != EXPECTED:
        raise ValueError(f"Unexpected family counts: {dict(counts)}")
    return units


def seeded(workload):
    return int.from_bytes(hashlib.sha256(('timing-validation-v1:'+workload).encode()).digest()[:4],'big')


def samples(launch,torch,n):
    values=[]
    for _ in range(n):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();launch();end.record();end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def describe(values):
    med=statistics.median(values)
    return dict(median_ms=med,mean_ms=statistics.mean(values),
                cv=statistics.stdev(values)/statistics.mean(values),
                half_median_drift=abs(statistics.median(values[len(values)//2:])/
                                      statistics.median(values[:len(values)//2])-1),samples_ms=values)


def graph_samples(launch,torch,n,target_ms=.75):
    # Increase captured work until the measured replay lasts at least 0.75ms,
    # reducing sensitivity to Python dispatch gaps around the replay call.
    calibration=[]
    for batch in (8,32,128,512,2048):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(batch):launch()
        for _ in range(5):graph.replay()
        torch.cuda.synchronize()
        pilot=samples(graph.replay,torch,3)
        elapsed=statistics.median(pilot)
        calibration.append(dict(batch=batch,replay_median_ms=elapsed))
        if elapsed>=target_ms or batch==2048:break
        del graph
    for _ in range(10):graph.replay()
    torch.cuda.synchronize()
    values=[v/batch for v in samples(graph.replay,torch,n)]
    return dict(**describe(values),captured_launches=batch,calibration=calibration,
                target_replay_ms=target_ms,target_reached=elapsed>=target_ms)


def workload(row,torch,gemm,reductions,extra,cases):
    family=row['kernel_family'];dtype=getattr(torch,row['dtype'])
    shape=json.loads(row['workload_features_json']);seed=seeded(row['workload_id'])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    generator=torch.Generator().manual_seed(seed)
    def randn(shape):return torch.randn(shape,generator=generator).to(device='cuda',dtype=dtype)
    if family=='gemm':
        a=randn((int(shape['m']),int(shape['k'])))
        b=randn((int(shape['k']),int(shape['n'])))
        output=torch.empty((a.shape[0],b.shape[1]),device='cuda',dtype=dtype)
        strict=a@b
        reference=(a.double()@b.double()).to(dtype)
        tolerance=.05 if dtype==torch.bfloat16 else .02
        torch.testing.assert_close(strict,reference,atol=tolerance,rtol=tolerance)
        def launch(config):gemm.launch(a,b,output,tuple(int(config[k]) for k in
                              ('block_m','block_n','block_k','num_warps','num_stages')))
        atol=rtol=tolerance
    elif family in ('softmax','cross_entropy'):
        rows,width=int(shape['rows']),int(shape['reduction_width'])
        x=randn((rows,width))
        target=torch.randint(0,width,(rows,),generator=generator).to('cuda')
        output=torch.empty_like(x) if family=='softmax' else torch.empty((rows,),device='cuda',dtype=torch.float32)
        reference=torch.softmax(x,dim=1) if family=='softmax' else torch.nn.functional.cross_entropy(x.float(),target,reduction='none')
        def launch(config):reductions.launch(family,x,output,target,tuple(int(config[k]) for k in
                                            ('block_multiplier','num_warps','num_stages')))
        atol=rtol=.02
    else:
        fields,_,knobs,_=cases(family,expanded=True)
        dimensions=tuple(shape[k] if k=='locality' else bool(shape[k]) if k=='causal' else int(shape[k]) for k in fields)
        execute,output,reference=extra.make_workload(family,dimensions,dtype,torch.Generator().manual_seed(seed))
        def launch(config):execute(*(int(config[k]) for k in knobs))
        atol,rtol=.02,.03
    incoming=execute if family not in ('gemm','softmax','cross_entropy') else launch
    signature=hashlib.sha256()
    for name,value in zip(incoming.__code__.co_freevars,incoming.__closure__ or ()):
        tensor=value.cell_contents
        if name!='output' and isinstance(tensor,torch.Tensor):
            signature.update((name+str(tensor.shape)+str(tensor.dtype)).encode())
            signature.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return launch,output,reference,atol,rtol,signature.hexdigest()


def select_rows(rows, device, smoke):
    """Choose the measured precision grid and, optionally, the original smoke panel."""
    if device == "T4":
        rows = [r for r in rows if r["dtype"] == "float16"]
        if smoke:
            def config_weight(row):
                knobs = json.loads(row["config_features_json"]).values()
                return sum(float(v) for v in knobs if isinstance(v, (float, int)))

            workloads = sorted({r["workload_id"] for r in rows})[:2]
            return [min((r for r in rows if r["workload_id"] == w), key=config_weight)
                    for w in workloads]
    elif smoke:
        return [next(r for r in rows if r["dtype"] == dtype)
                for dtype in ("float16", "bfloat16")]
    return rows


def device_protocol(device):
    if device not in REUSED_REGISTRY:
        return {}
    fields = dict(grid_source_device="A100", output_buffer_bytes=BUFFER_BYTES,
                  selected_device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if device == "RTXA5000":
        fields["environment"] = "Shared lab native Linux; monitored, not scheduler-exclusive"
    elif device == "T4":
        fields.update(environment="Google Colab T4; managed cloud VM",
                      comparison_group="turing-fp16-torch26-triton32",
                      excluded_dtype="bfloat16: no native T4 BF16",
                      cross_compiler_pooling_allowed=False)
    return fields


def source_fields(device, row):
    if device in REUSED_REGISTRY:
        return dict(registry_source_valid=row.get("is_valid"),
                    registry_source_latency_median_ms=row.get("latency_median_ms", ""))
    return dict(original_valid=row.get("is_valid"),
                original_latency_median_ms=row.get("latency_median_ms", ""))


def unit(args):
    units = registry(args.root, args.device)
    source, rows = units[args.family, args.shard]
    rows = select_rows(rows, args.device, args.limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summary_path=args.output.with_suffix('.summary.json')
    if args.output.exists() or summary_path.exists():raise FileExistsError(args.output)
    study=Path(__file__).resolve().parent / 'kernels'
    sys.path.insert(0,str(study))
    import torch,triton
    import benchmark_matmul as gemm
    import benchmark_reductions as reductions
    import extra_triton_kernels as extra
    from benchmark_extra_kernels import cases
    from measurement_utils import architecture_metadata,dynamic_telemetry
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    arch=architecture_metadata()
    expected = GPU_NAMES[args.device]
    if expected not in torch.cuda.get_device_name():raise ValueError('Unexpected GPU '+torch.cuda.get_device_name())
    manifest=dict(device=args.device,pass_index=args.pass_index,family=args.family,shard=args.shard,
        source=str(source),source_sha256=digest(source),rows=len(rows),torch=torch.__version__,triton=triton.__version__,
        python=sys.version,architecture=arch,hostname=platform.node(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID',''),
        script_sha256=digest(Path(__file__)),kernel_sources={p.name:digest(p) for p in study.glob('*.py')},
        input_policy='CPU-generated fixed per-workload SHA256 seed, shared configs/devices/passes; actual GPU input hashes stored. Not historical input reconstruction.',
        timing_policy='Separate individual-launch and adaptive CUDA-graph regimes, counterbalanced order. Graph values are amortized per captured launch.',
        repetitions=args.repetitions,graph_target_ms=.75,workload_warmup_seconds=.1,
        **device_protocol(args.device))
    atomic(args.output.with_suffix('.protocol.json'),manifest)
    groups=defaultdict(list)
    for row in rows:groups[row['workload_id']].append(row)
    rng=random.Random(81731+args.pass_index);order=sorted(groups);rng.shuffle(order)
    counts=defaultdict(int);started=time.monotonic()
    with args.output.open('x', buffering=manifest.get('output_buffer_bytes', 1)) as stream:
        for work_id in order:
            members=groups[work_id];rng.shuffle(members)
            execute,output,reference,atol,rtol,input_hash=workload(members[0],torch,gemm,reductions,extra,cases)
            telemetry=dynamic_telemetry()
            for member_index,row in enumerate(members):
                config=json.loads(row['config_features_json'])
                launch=lambda:execute(config)
                record=dict(workload_id=work_id,config_id=row['config_id'],kernel_family=args.family,dtype=row['dtype'],
                    workload_features_json=row['workload_features_json'],config_features_json=row['config_features_json'],
                    input_seed=seeded(work_id),input_sha256=input_hash,
                    **source_fields(args.device, row),
                    pass_index=args.pass_index,telemetry=telemetry,correctness_pass=False,timing_status='not_run')
                stage='compile_or_first_launch'
                try:
                    tick=time.monotonic();launch();torch.cuda.synchronize()
                    record['compile_ms']=(time.monotonic()-tick)*1000;stage='correctness'
                    torch.testing.assert_close(output.float(),reference.float(),atol=atol,rtol=rtol)
                    record['correctness_pass']=True
                    if member_index==0:
                        warm_until=time.monotonic()+.1
                        while time.monotonic()<warm_until:
                            for _ in range(16):launch()
                            torch.cuda.synchronize()
                    stream_warm=torch.cuda.Stream();stream_warm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream_warm):
                        for _ in range(10):launch()
                    torch.cuda.current_stream().wait_stream(stream_warm);torch.cuda.synchronize()
                    modes=['individual','graph'] if (seeded(work_id+row['config_id'])+args.pass_index)%2 else ['graph','individual']
                    record['protocol_order']=modes
                    for mode in modes:
                        stage='timing_'+mode
                        if mode=='individual':
                            for _ in range(10):launch()
                            torch.cuda.synchronize()
                            record[mode]=describe(samples(launch,torch,args.repetitions))
                        else:record[mode]=graph_samples(launch,torch,args.repetitions)
                    stage='post_timing_correctness'
                    torch.testing.assert_close(output.float(),reference.float(),atol=atol,rtol=rtol)
                    record['timing_status']='ok';counts['successful']+=1
                except Exception as exc:
                    record.update(failure_stage=stage,failure_type=type(exc).__name__,failure_message=str(exc)[:3000],timing_status='failed')
                    if stage=='post_timing_correctness':record['correctness_pass']=False
                    counts['failed']+=1
                    # A damaged CUDA context must not turn all later rows into invalid candidates.
                    if 'illegal memory' in str(exc).lower() or 'device-side assert' in str(exc).lower():raise
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                if sum(counts.values())%25==0:print(args.family,args.shard,'rows',sum(counts.values()),dict(counts),flush=True)
            del execute,launch,output,reference
            torch.cuda.empty_cache()
    atomic(summary_path,dict(complete=True,rows=sum(counts.values()),counts=dict(counts),seconds=time.monotonic()-started,
        source_sha256=digest(source),output_sha256=digest(args.output)))
    print('Completed',args.output,dict(counts),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--device',choices=tuple(GPU_NAMES),required=True)
    p.add_argument('--pass-index',type=int,choices=range(3),required=True)
    p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--shard',type=int,choices=range(10),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repetitions',type=int,default=50)
    p.add_argument('--limit',action='store_true')
    a=p.parse_args()
    if a.repetitions<4:raise ValueError('Require >=4 timing samples')
    unit(a)

if __name__=='__main__':main()
