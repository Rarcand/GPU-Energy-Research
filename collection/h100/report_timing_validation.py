#!/usr/bin/env python3
"""Report protocol sensitivity and repeated-pass stability without filtering noise.

Use --input for one device's validation root; --peer optionally adds the other.
Incomplete runs produce coverage only, never final stability claims.
"""
import argparse,json,statistics
from collections import defaultdict,Counter
from pathlib import Path
import sys
SHARED = Path(__file__).resolve().parents[1] / 'shared'
sys.path.insert(0, str(SHARED))
from timing_validation import digest,atomic,EXPECTED


def read(root):
    rows={};units=Counter()
    for summary in sorted(root.rglob('*.summary.json')):
        if summary.parent.name not in ('pass-00','pass-01','pass-02'):continue
        path=summary.with_name(summary.name.replace('.summary.json','.jsonl'))
        info=json.loads(summary.read_text())
        if not info['complete'] or digest(path)!=info['output_sha256']:raise ValueError('Corrupt completed output '+str(path))
        protocol=json.loads(path.with_suffix('.protocol.json').read_text())
        device=protocol['device'];p=protocol['pass_index'];units[(device,p)]+=1
        with path.open() as stream:
            for line in stream:
                r=json.loads(line);key=(device,p,r['kernel_family'],r['workload_id'],r['config_id'])
                if key in rows:raise ValueError('Duplicate published candidate-pass')
                for mode in ('individual','graph'):
                    if mode in r:r[mode]={k:v for k,v in r[mode].items() if k not in ('samples_ms','calibration')}
                rows[key]=r
    return rows,units


def assess(rows,device):
    by_pass={p:{k[2:]:v for k,v in rows.items() if k[0]==device and k[1]==p} for p in range(3)}
    coverage={str(p):dict(rows=len(rs),successful=sum(r['timing_status']=='ok' for r in rs.values()),
                           families=dict(Counter(k[0] for k in rs))) for p,rs in by_pass.items()}
    complete=all(len(rs)==30000 and dict(Counter(k[0] for k in rs))==EXPECTED for rs in by_pass.values())
    out=dict(complete=complete,coverage=coverage)
    if not complete:return out
    keys=set(by_pass[0])
    if any(set(rs)!=keys for rs in by_pass.values()):raise ValueError('Candidate sets differ across passes')
    mismatch=sum(len({by_pass[p][k]['input_sha256'] for p in range(3)})!=1 for k in keys)
    out['input_hash_mismatches_across_passes']=mismatch
    good={k for k in keys if all(by_pass[p][k]['timing_status']=='ok' and by_pass[p][k]['correctness_pass'] for p in range(3))}
    out['fully_measured_correct_candidates']=len(good)
    out['failed_candidate_keys']=len(keys-good)
    out['families']={}
    for family in EXPECTED:
        selected=sorted(k for k in good if k[0]==family)
        if not selected:continue
        stats={}
        for mode in ('individual','graph'):
            spread=[];cv=[];drift=[];tasks=defaultdict(list)
            for k in selected:
                values=[by_pass[p][k][mode]['median_ms'] for p in range(3)]
                spread.append((max(values)-min(values))/statistics.median(values))
                for p in range(3):
                    cv.append(by_pass[p][k][mode]['cv']);drift.append(by_pass[p][k][mode]['half_median_drift'])
                tasks[k[1]].append(values)
            stable=[]
            for candidates in tasks.values():
                best=min(range(len(candidates)),key=lambda i:candidates[i][0])
                stable.extend(candidates[best][p]<=1.05*min(v[p] for v in candidates) for p in (1,2))
            stats[mode]=dict(candidates=len(selected),tasks=len(tasks),median_cross_pass_rel_spread=statistics.median(spread),
                cross_pass_spread_over_5pct_fraction=sum(v>.05 for v in spread)/len(spread),
                median_within_pass_cv=statistics.median(cv),within_pass_half_drift_over_5pct_fraction=sum(v>.05 for v in drift)/len(drift),
                pass0_best_within5pct_on_later_passes=sum(stable)/len(stable))
        ratios=[];tasks=defaultdict(list)
        for k in selected:
            individual=statistics.median(by_pass[p][k]['individual']['median_ms'] for p in range(3))
            graph=statistics.median(by_pass[p][k]['graph']['median_ms'] for p in range(3))
            ratios.append(individual/graph);tasks[k[1]].append((individual,graph))
        stable=[]
        for candidates in tasks.values():
            best=min(range(len(candidates)),key=lambda i:candidates[i][0])
            stable.append(candidates[best][1]<=1.05*min(v[1] for v in candidates))
        stats['individual_over_graph_median_ratio']=statistics.median(ratios)
        stats['individual_selected_within5pct_of_graph_best']=sum(stable)/len(stable)
        out['families'][family]=stats
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--peer',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows,units=read(a.input)
    if a.peer:
        other,ou=read(a.peer)
        if set(rows)&set(other):raise ValueError('Overlapping roots')
        rows.update(other);units.update(ou)
    devices=sorted({k[0] for k in rows})
    report=dict(complete_for_supplied_devices=bool(devices),
        caveats=['Three passes are technical repeats, not independent GPU samples.',
                 'Graph and individual protocols define separate latency targets; original labels are unchanged.',
                 'Stability metrics condition on keys successfully measured and correct in all three passes; coverage reports failures separately.',
                 'Fixed seeded CPU generation is checked with actual input hashes, not assumed bitwise portable.'],
        devices={d:assess(rows,d) for d in devices})
    report['complete_for_supplied_devices']=bool(devices) and all(v['complete'] for v in report['devices'].values())
    if len(devices)==2 and report['complete_for_supplied_devices']:
        akeys={k[2:]:v['input_sha256'] for k,v in rows.items() if k[0]==devices[0] and k[1]==0}
        bkeys={k[2:]:v['input_sha256'] for k,v in rows.items() if k[0]==devices[1] and k[1]==0}
        if set(akeys)!=set(bkeys):raise ValueError('Device candidate grids differ')
        report['cross_device_input_hash_mismatches']=sum(akeys[k]!=bkeys[k] for k in akeys)
    atomic(a.output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
