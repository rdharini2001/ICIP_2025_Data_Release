#!/usr/bin/env python3
"""Verify published AVID numbers against bundled numeric CSVs (not raw images)."""
import csv, json, math
from pathlib import Path
from statistics import median,mean,pstdev
R=Path(__file__).resolve().parents[2]
D=R/'reproducibility_data'
def rows(x):
 with (D/x).open(newline='') as f:return list(csv.DictReader(f))
def check(name,actual,expected,tol=1e-6):
 if isinstance(expected,(int,float)):
  assert math.isclose(actual,expected,rel_tol=0,abs_tol=tol),(name,actual,expected)
 else: assert actual==expected,(name,actual,expected)
 results.append((name,actual,expected))
results=[]
cor=rows('dataset_corpus_stats.csv')
check('Train pairs',int(cor[0]['n_pairs']),28790);check('Val pairs',int(cor[1]['n_pairs']),3250)
check('Total pairs',sum(int(x['n_pairs']) for x in cor),32040)
check('Videos',sum(int(x['n_videos']) for x in cor),115)
check('Bird instances',sum(int(x['bird_instances']) for x in cor),15471)
check('Drone instances',sum(int(x['drone_instances']) for x in cor),20481)
check('Total instances',sum(int(x['bird_instances'])+int(x['drone_instances']) for x in cor),35952)
check('COCO small',sum(int(x['bbox_small']) for x in cor),35581)
check('COCO medium',sum(int(x['bbox_medium']) for x in cor),371)
hist=rows('bbox_scale_histogram.csv'); bins={}
for x in hist: bins[x['bucket_px']]=bins.get(x['bucket_px'],0)+int(x['count'])
check('Histogram total',sum(bins.values()),35952)
check('0-8px',bins['0-4px']+bins['4-8px'],103)
check('12-16px',bins['12-16px'],17706)
check('8-16px',bins['8-12px']+bins['12-16px'],24489)
check('8-16 fraction percent',100*24489/35952,68.1,tol=.05)
check('COCO small percent',100*35581/35952,99.0,tol=.05)
shifts=rows('real_alignment_sample_300pairs.csv');ds=[float(x['alignment_shift_px']) for x in shifts]
check('N real shifts',len(ds),300)
check('Mean real shift',mean(ds),4.66,tol=.006)
check('Median real shift',median(ds),1.0)
check('zero shift fraction',100*sum(x==0 for x in ds)/300,46.0,tol=.05)
check('>2px fraction',100*sum(x>2 for x in ds)/300,36.0,tol=.05)
check('>5px fraction',100*sum(x>5 for x in ds)/300,32.3,tol=.05)
check('Search radial max',12*math.sqrt(2),16.97,tol=.0057)
matched=rows('object_relative_misalignment_rows.csv'); rhos=[float(x['rho']) for x in matched]
check('N matched',len(matched),264)
raw={(x['video_id'],x['frame_idx']):float(x['alignment_shift_px']) for x in shifts}
for x in matched:
 k=(x['video_id'],x['frame_idx']);assert k in raw
 assert math.isclose(raw[k],float(x['alignment_shift_px']),abs_tol=1e-5)
 assert math.isclose(float(x['rho']),float(x['alignment_shift_px'])/float(x['median_target_side_px']),abs_tol=1e-5)
check('median target side matched',median([float(x['median_target_side_px']) for x in matched]),14.18,tol=.005)
check('mean rho',mean(rhos),.297,tol=.0005)
check('median rho',median(rhos),0)
for t,n in [(.25,72),(.5,69),(1.,43)]:
 check(f'Count rho>{t}',sum(x>t for x in rhos),n)
check('B1 of 264',mean(min(x,1) for x in rhos),.265,tol=.0005)
cr=rows('bayes_bound_tightness_and_kde_crosscheck.csv');bounds={x['modality']:x for x in cr}
for key,val in [('rgb',.1484),('ir',.1979),('fusion',.0457)]:
 check(f'Bhat {key}',float(bounds[key]['bhattacharyya_bound_s0.5']),val,tol=.0001)
for key,val in [('rgb',.1314),('ir',.1944),('fusion',.0396)]:
 check(f'Chernoff {key}',float(bounds[key]['chernoff_bound_optimized']),val,tol=.0001)
for key,val in [('rgb',.1650),('ir',.1774),('fusion',.0472)]:
 check(f'KDE {key}',float(bounds[key]['kde_nonparametric_bound']),val,tol=.0001)
check('Fusion gap 12-16',float(bounds['rgb']['bhattacharyya_bound_s0.5'])-float(bounds['fusion']['bhattacharyya_bound_s0.5']),.103,tol=.0005)
dim=rows('dimension_control_12_16.csv')
check('Dimension 4+4 bound',float(next(x for x in dim if x['representation'].lower().startswith('rgb+ir balanced'))['bound']),.1396,tol=.0001)
co=rows('corruption_sweep.csv');sev5=[x for x in co if x['severity']=='5'];check('Severity-5 observations',len(sev5),14)
for modality,base,inc,mult in [('rgb',.147694,.132477,1.896968),('ir',.162318,.051517,1.317384)]:
 clean=float(next(x['bayes_error_bound'] for x in co if x['modality']==modality and x['corruption_family']=='clean'))
 arr=[float(x['bayes_error_bound']) for x in sev5 if x['modality']==modality]
 check(f'{modality} clean bound',clean,base,tol=1e-5)
 check(f'{modality} avg severity5 penalty',mean(arr)-clean,inc,tol=.00001)
 check(f'{modality} avg severity5 ratio',mean(arr)/clean,mult,tol=.00002)
check('severity5 speckle RGB',float(next(x['bayes_error_bound'] for x in sev5 if x['modality']=='rgb' and x['corruption_family']=='speckle_noise')),.3752,tol=.0001)
check('severity5 camera IR',float(next(x['bayes_error_bound'] for x in sev5 if x['modality']=='ir' and x['corruption_family']=='camera_instability')),.2240,tol=.0001)
noise=rows('noise_floor_check.csv'); nf={x['modality']:x for x in noise}
for mod,clean,ratio in [('RGB',.301893,16.6),('IR',.638350,7.8)]:
 n=float(nf[mod]['clean_full_frame_median_sigma']);check(f'{mod} clean noise',n,clean,tol=.00001)
 check(f'{mod} noise ratio',5/n,ratio,tol=.05)
sweep=rows('alignment_bfg_sweep.csv');p0=next(x for x in sweep if x['injected_shift_px']=='0.0');p12=next(x for x in sweep if x['injected_shift_px']=='12.0')
check('Fusion gap shift0',float(p0['bayes_fusion_gain']),.1107,tol=.0001)
check('Fusion gap shift12',float(p12['bayes_fusion_gain']),.0935,tol=.0001)
check('Pair bound shift0',float(p0['fusion_bayes_bound']),.0370,tol=.0001)
check('Pair bound shift12',float(p12['fusion_bayes_bound']),.0542,tol=.0001)
check('Fusion gap retention shift12',float(p12['bayes_fusion_gain'])/float(p0['bayes_fusion_gain']),.844,tol=.0005)
print(json.dumps({'checks':len(results),'status':'PASS','selected':{k:v for k,v,e in results if k in ['Total pairs','Total instances','N real shifts','N matched','B1 of 264','Fusion gap retention shift12','8-16px']}},indent=2))
