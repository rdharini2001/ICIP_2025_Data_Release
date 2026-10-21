from pathlib import Path
import csv,json
root=Path(__file__).resolve().parents[1];d=root/'reproducibility_data';a=root/'docs/assets'
def load(name):
 with (d/name).open() as f:return list(csv.DictReader(f))
def cast(row):
 o={}
 for k,v in row.items():
  try:o[k]=float(v) if '.' in v or 'e-' in v else int(v)
  except (TypeError,ValueError):o[k]=v
 return o
rows=[cast(r) for r in load('object_relative_misalignment_rows.csv')]
shifts=[cast(r) for r in load('real_alignment_sample_300pairs.csv')]
h={}
for x in load('bbox_scale_histogram.csv'):h[x['bucket_px']]=h.get(x['bucket_px'],0)+int(x['count'])
source={'rho':[v['rho'] for v in rows], 'shifts':[v['alignment_shift_px'] for v in shifts], 'size':h,'corruption':[cast(r) for r in load('corruption_sweep.csv')], 'fusion':[cast(r) for r in load('alignment_bfg_sweep.csv')], 'cross':[cast(r) for r in load('bayes_bound_tightness_and_kde_crosscheck.csv')]}
(a/'data.js').write_text('window.AVID='+json.dumps(source,separators=(',',':'))+';\n')
(a/'data.json').write_text(json.dumps(source,indent=1)+'\n')
