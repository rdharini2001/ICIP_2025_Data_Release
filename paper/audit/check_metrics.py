from pathlib import Path
import json,csv,numpy as np
root=Path(__file__).resolve().parents[2];d=root/'reproducibility_data'
def rows(filename):
 with (d/filename).open() as f:return list(csv.DictReader(f))
base=rows('dataset_corpus_stats.csv');rs=rows('object_relative_misalignment_rows.csv');ds=rows('real_alignment_sample_300pairs.csv')
assert sum(int(r['n_pairs']) for r in base)==32040
assert sum(int(r['n_videos']) for r in base)==115
assert sum(int(r['bird_instances'])+int(r['drone_instances']) for r in base)==35952
v=np.array([float(r['rho']) for r in rs]);s=np.array([float(r['alignment_shift_px']) for r in ds])
assert len(v)==264 and len(s)==300
assert (s>5).sum()==97
assert [(v>t).sum() for t in [.25,.5,1]]==[72,69,43]
result={'paired_frames':32040,'videos':115,'instances':35952,'n_pixel_shifts':len(s),'fraction_shift_gt_5':(s>5).mean(),'n_annotated_relative':len(v),'missing_label_join':36,'fraction_rho_gt_025':(v>.25).mean(),'fraction_rho_gt_05':(v>.5).mean(),'fraction_rho_gt_1':(v>1).mean(),'burden_unit':np.minimum(v,1).mean(),'median_rho':np.median(v).item(),'median_target_side_px':float(np.median([float(r['median_target_side_px']) for r in rs]))}
(root/'paper/audit/metrics_verified.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
