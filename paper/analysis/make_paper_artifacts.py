from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DATA = ROOT / 'reproducibility_data'
FIG.mkdir(exist_ok=True)

# Palette sampled from the supplied reference paper. The paper repeatedly uses
# a dark navy primary, medium blue, muted cyan, warm gold, and sparse red/green
# accents on a white background.
NAVY = '#0E3D86'
BLUE = '#356CAC'
CYAN = '#80BEC8'
DEEP = '#08064A'
GOLD = '#C7AA62'
PALEGOLD = '#E7D786'
GREEN = '#78BA49'
RED = '#D41010'
GRAY = '#6F6F6F'
LIGHTGRAY = '#D9D9D9'
PALEBLUE = '#D1DFEB'
PALEGREEN = '#E4F0DD'
PALEYELLOW = '#F0E6C9'
LAVENDER = '#E6E5F2'
INK = '#202020'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.3,
    'axes.labelsize': 8.3,
    'axes.titlesize': 8.5,
    'legend.fontsize': 7.2,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

def save(fig, name, dpi=320):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(FIG / f'{name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

def panel_label(ax, label, x=-0.10, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='top', ha='left', color='black')

def light_grid(ax):
    ax.grid(axis='y', color='#E7E7E7', linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

# -----------------------------------------------------------------------------
# Figure 1: benchmark overview. Scientific schematic rather than a decorative
# infographic: four modules, compact labels, restrained pale fills, and the same
# navy/blue/cyan/gold/green hierarchy as the reference paper.
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.2, 4.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

def module(x, y, w, h, edge, fill, title, lines, letter):
    ax.text(x-0.015, y+h+0.035, letter, fontsize=12, fontweight='bold', va='bottom')
    box = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.008,rounding_size=0.018',
                         facecolor=fill, edgecolor=edge, linewidth=1.15)
    ax.add_patch(box)
    ax.add_patch(Rectangle((x+0.018, y+h-0.098), 0.042, 0.042,
                           facecolor=edge, edgecolor='none', alpha=0.95))
    ax.text(x+0.075, y+h-0.055, title, fontsize=10.2, fontweight='bold', va='center', color=INK)
    ax.plot([x+0.02, x+w-0.02], [y+h-0.12, y+h-0.12], color=edge, lw=0.55, alpha=0.55)
    yy = y+h-0.175
    for line in lines:
        ax.text(x+0.035, yy, u'\u2022', color=edge, fontsize=10, va='center')
        ax.text(x+0.055, yy, line, fontsize=8.25, va='center', color='#333333')
        yy -= 0.078

def arrow(p1, p2):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=12,
                                connectionstyle='arc3,rad=0.08', linewidth=1.25,
                                color='#727272'))

module(0.025,0.20,0.25,0.62,NAVY,'#F2F6FA','AVID dataset',[
    '32,040 paired RGB--IR frames',
    '115 short surveillance videos',
    '35,952 bird / drone instances',
    '68.1% of targets are 8--16 px',
], 'a')
module(0.365,0.55,0.28,0.27,GOLD,'#FBF8EE','Controlled perturbations',[
    '7 corruption families x 5 severities',
    'weak-alignment stress tests',
], 'b')
module(0.365,0.18,0.28,0.27,GREEN,'#F3F8F0','Measured data properties',[
    'object scale and class balance',
    'RGB--IR shift relative to target size',
], 'c')
module(0.735,0.20,0.24,0.62,DEEP,'#F5F4FA','Data-intrinsic readouts',[
    'class-overlap error bounds',
    'fusion-gap and corruption penalties',
    'alignment burden curve',
    'mutual-information sensitivity',
], 'd')
arrow((0.275,0.56),(0.36,0.68)); arrow((0.275,0.45),(0.36,0.31))
arrow((0.65,0.68),(0.73,0.58)); arrow((0.65,0.31),(0.73,0.45))
ax.text(0.50,0.075,'Dataset release  \u2192  controlled changes  \u2192  scale-aware, model-free diagnostics',
        ha='center', va='center', fontsize=9, color='#333333')
save(fig,'fig_overview')

# -----------------------------------------------------------------------------
# Figure 2: representative real pairs. Recolor source annotation boxes from
# green to the sparse red annotation color used in the reference paper, remove
# the old super-title, and preserve the actual images.
# -----------------------------------------------------------------------------
src = np.array(Image.open(ROOT/'analysis'/'fig_real_samples_source.png').convert('RGB'))
mask = (src[:,:,1] > 180) & (src[:,:,0] < 100) & (src[:,:,2] < 100)
src[mask] = np.array([212,16,16], dtype=np.uint8)
# Crop away the old heading, while keeping the four panel titles.
src = src[95:930,:,:]
Image.fromarray(src).save(FIG/'fig_real_samples.png')

# diversity montage: same annotation recolor, remove old heading only.
div = np.array(Image.open(ROOT/'analysis'/'fig_diversity_source.png').convert('RGB'))
mask = (div[:,:,1] > 180) & (div[:,:,0] < 100) & (div[:,:,2] < 100)
div[mask] = np.array([212,16,16], dtype=np.uint8)
div = div[55:,:,:]
Image.fromarray(div).save(FIG/'fig_diversity_montage.png')

# -----------------------------------------------------------------------------
# Figure 3: scale + alignment distribution + new alignment burden curve.
# -----------------------------------------------------------------------------
bbox = pd.read_csv(DATA/'bbox_scale_histogram.csv')
order=['0-4px','4-8px','8-12px','12-16px','16-20px','20-24px','24-28px','28-32px','32-48px']
b = bbox[bbox.bucket_px.isin(order)].pivot(index='bucket_px',columns='split',values='count').fillna(0).reindex(order)
combined=b.sum(axis=1)
rel = pd.read_csv(DATA/'real_alignment_sample_300pairs.csv')
burden = pd.read_csv(DATA/'alignment_burden_curve.csv')

fig, axs = plt.subplots(1,3,figsize=(11.1,3.15),gridspec_kw={'width_ratios':[1.2,1.0,1.05]})
ax=axs[0]
ax.bar(np.arange(len(order)),combined.values,color=NAVY,width=.68,zorder=3)
ax.set_xticks(range(len(order))); ax.set_xticklabels([x.replace('px','') for x in order],rotation=42,ha='right')
ax.set_ylabel('annotated instances'); ax.set_xlabel('geometric-mean box side (px)')
ax.axvspan(1.5,3.5,color=CYAN,alpha=.15,lw=0)
ax.text(.98,.94,'68.1% at 8--16 px',transform=ax.transAxes,ha='right',va='top',fontsize=7.7,color=NAVY)
panel_label(ax,'a'); light_grid(ax)

ax=axs[1]
bins=np.arange(-.5,18.5,1)
ax.hist(rel.alignment_shift_px,bins=bins,color=BLUE,edgecolor='white',linewidth=.45,zorder=3)
ax.axvline(5,color=GOLD,ls='--',lw=1.15)
ax.set_xlabel('estimated RGB--IR shift (px)'); ax.set_ylabel('sampled frame pairs')
ax.text(.97,.94,'median 1 px\n32.3% exceed 5 px',transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color=BLUE)
ax.text(.97,.72,'search: +/-12 px per axis',transform=ax.transAxes,ha='right',fontsize=6.9,color=GRAY)
panel_label(ax,'b'); light_grid(ax)

ax=axs[2]
thresholds=[0.25, 0.50, 1.00]
matched=pd.read_csv(DATA/'object_relative_misalignment_rows.csv')
r=matched['rho'].values
values=[100*np.mean(r>t) for t in thresholds]
ax.bar(np.arange(3), values, color=BLUE, width=.64, zorder=3)
ax.set_xticks(range(3)); ax.set_xticklabels(['0.25','0.50','1.00'])
ax.set_ylim(0, 35)
ax.set_xlabel('target sides'); ax.set_ylabel('pairs exceeding threshold (%)')
for i,v in enumerate(values): ax.text(i,v+0.75,f'{v:.1f}%',ha='center',fontsize=7.3,color=NAVY)
unit=np.minimum(r,1).mean()
ax.text(.97,.94,f'{len(r)} matched pairs\nBurden score: {unit:.3f}',
        transform=ax.transAxes,ha='right',va='top',fontsize=7.2,color=DEEP)
panel_label(ax,'c'); light_grid(ax)
fig.tight_layout(w_pad=1.6)
save(fig,'fig_dataset_statistics')

# -----------------------------------------------------------------------------
# Figure 4: mathematical framework and interpretation.
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1,4,figsize=(11.1,2.85),gridspec_kw={'width_ratios':[1.0,1.1,1.1,1.15]})
# a: fixed patch representation
ax=axs[0]; ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
panel_label(ax,'a',x=-.02,y=1.02)
ax.text(.50,.96,'Fixed representation',ha='center',fontweight='bold',fontsize=8.8)
for y,c,lab in [(.70,BLUE,'RGB crop'),(.43,GOLD,'IR crop')]:
    ax.add_patch(Rectangle((.05,y-.10),.22,.18,facecolor=c,alpha=.18,edgecolor=c,lw=1))
    # tiny-object mark
    ax.add_patch(Rectangle((.135,y-.02),.045,.035,facecolor='none',edgecolor=RED,lw=1.1))
    ax.text(.16,y-.14,lab,ha='center',fontsize=7.2)
    ax.add_patch(FancyArrowPatch((.29,y),(.48,y),arrowstyle='-|>',mutation_scale=10,color=GRAY,lw=1))
    ax.add_patch(Rectangle((.51,y-.085),.17,.15,facecolor='#F5F5F5',edgecolor=LIGHTGRAY,lw=.9))
    ax.text(.595,y,'24 x 24\ngrayscale',ha='center',va='center',fontsize=6.8)
    ax.add_patch(FancyArrowPatch((.70,y),(.85,y),arrowstyle='-|>',mutation_scale=10,color=GRAY,lw=1))
    ax.text(.91,y,'PCA',ha='center',va='center',fontsize=7.2,fontweight='bold')
ax.text(.50,.10,'PCA is fitted once on clean pooled patches\nand reused across each perturbation sweep.',ha='center',fontsize=6.7,color=GRAY)

# b: overlap bound
ax=axs[1]
panel_label(ax,'b')
x=np.linspace(-3.5,4,450)
def norm(x,mu,s): return np.exp(-.5*((x-mu)/s)**2)/(s*np.sqrt(2*np.pi))
y0=norm(x,-.85,1.0); y1=norm(x,1.05,1.18)
ax.fill_between(x,0,y0,color=CYAN,alpha=.38); ax.plot(x,y0,color=BLUE,lw=1.4)
ax.fill_between(x,0,y1,color=PALEGOLD,alpha=.50); ax.plot(x,y1,color=GOLD,lw=1.4)
ax.fill_between(x,0,np.minimum(y0,y1),color=GREEN,alpha=.22)
ax.set_xticks([]); ax.set_yticks([]); ax.spines[['left','bottom']].set_visible(False)
ax.set_title('Class-conditional overlap',fontweight='bold',pad=5)
ax.text(.50,.08,r'larger overlap $\Rightarrow$ larger error upper bound',transform=ax.transAxes,ha='center',fontsize=6.8)
ax.text(.50,.82,r'$P_e\leq\sqrt{\pi_0\pi_1}\,e^{-D_B}$',transform=ax.transAxes,ha='center',fontsize=8.2,color=DEEP)

# c: object-relative alignment
ax=axs[2]; ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
panel_label(ax,'c',x=-.02,y=1.02)
ax.text(.50,.96,'Alignment in object units',ha='center',fontweight='bold',fontsize=8.8)
ax.add_patch(Rectangle((.13,.48),.36,.22,facecolor=PALEBLUE,edgecolor=BLUE,lw=1.0))
ax.add_patch(Rectangle((.26,.545),.10,.075,facecolor='none',edgecolor=RED,lw=1.2))
ax.annotate('',xy=(.36,.46),xytext=(.26,.46),arrowprops=dict(arrowstyle='<->',color=RED,lw=1))
ax.text(.31,.405,r'$s_{target}$',ha='center',fontsize=7.4,color=RED)
ax.add_patch(Rectangle((.47,.36),.36,.22,facecolor='#F6F0DD',edgecolor=GOLD,lw=1.0))
ax.add_patch(Rectangle((.65,.425),.10,.075,facecolor='none',edgecolor=RED,lw=1.2))
ax.annotate('',xy=(.65,.67),xytext=(.36,.67),arrowprops=dict(arrowstyle='<->',color=DEEP,lw=1.15))
ax.text(.505,.705,r'$d_{align}$',ha='center',fontsize=7.4,color=DEEP)
ax.text(.50,.21,r'$\rho=d_{align}/s_{target}$',ha='center',fontsize=9.2,color=DEEP)
ax.text(.50,.09,r'$\hat A(\tau)=n^{-1}\sum_i \mathbf{1}[\rho_i>\tau]$',ha='center',fontsize=8.0,color=GREEN)

# d: derived diagnostics
ax=axs[3]; ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
panel_label(ax,'d',x=-.02,y=1.02)
ax.text(.50,.96,'Perturbation readouts',ha='center',fontweight='bold',fontsize=8.8)
items=[
    (NAVY,r'$\Delta_{fusion}=\min(\hat B_{RGB},\hat B_{IR})-\hat B_{RGB+IR}$','paired-feature bound advantage'),
    (BLUE,r'$\Delta_c(s)=\hat B_{c,s}-\hat B_{clean}$','corruption separability penalty'),
    (GREEN,r'$R_{fusion}(a)=\Delta_{fusion}(a)/\Delta_{fusion}(0)$','fusion-gap retention'),
    (GOLD,r'$I(Y;Z_{RGB},Z_{IR})$','label-information sensitivity'),
]
y=.76
for col,form,desc in items:
    ax.add_patch(Circle((.07,y),.020,facecolor=col,edgecolor='none'))
    ax.text(.12,y+.025,form,fontsize=7.6,va='center',color=INK)
    ax.text(.12,y-.045,desc,fontsize=6.6,va='center',color=GRAY)
    y-=.19
save(fig,'fig_intrinsic_framework')

# -----------------------------------------------------------------------------
# Figure 5: scale separability, cross-checks, and representation sensitivity.
# -----------------------------------------------------------------------------
sc=pd.read_csv(DATA/'scale_sweep.csv')
# pandas can parse booleans as strings depending on environment
sc=sc[~sc['skipped'].astype(str).str.lower().eq('true')].copy()
scale_order=['0-8','8-12','12-16','16-20','20-24','24-28','28-32','32-48']
fig, axs=plt.subplots(1,4,figsize=(11.3,3.15),gridspec_kw={'width_ratios':[1.23,.90,1.05,1.15]})
ax=axs[0]
for mod,color,marker,label in [('rgb',BLUE,'o','RGB'),('ir',GOLD,'s','IR'),('fusion',NAVY,'^','RGB+IR (8+8)')]:
    d=sc[sc.modality==mod].copy(); d['idx']=d.bucket_px.map({k:i for i,k in enumerate(scale_order)}); d=d.dropna(subset=['idx']).sort_values('idx')
    ax.plot(d.idx,d.bayes_error_bound,marker=marker,lw=1.55,ms=4.0,label=label,color=color)
ax.set_xticks(range(6)); ax.set_xticklabels(scale_order[:6],rotation=40,ha='right')
ax.set_xlabel('object side length (px)'); ax.set_ylabel('Bhattacharyya error upper bound')
ax.legend(frameon=False,ncol=1,loc='upper right',borderaxespad=.2)
ax.text(.03,.07,'0--8 px has 103 instances',transform=ax.transAxes,fontsize=6.6,color=GRAY)
panel_label(ax,'a'); light_grid(ax)

piv=sc.pivot(index='bucket_px',columns='modality',values='bayes_error_bound')
piv['gap']=piv[['rgb','ir']].min(axis=1)-piv['fusion']
piv=piv.reindex(scale_order[:6])
ax=axs[1]
ax.bar(np.arange(len(piv)),piv.gap,color=NAVY,width=.68,zorder=3)
ax.axhline(0,color=GRAY,lw=.65)
ax.set_xticks(np.arange(len(piv))); ax.set_xticklabels(piv.index,rotation=42,ha='right')
ax.set_xlabel('object side length (px)'); ax.set_ylabel('fusion-gap diagnostic')
panel_label(ax,'b'); light_grid(ax)

cross=pd.read_csv(DATA/'bayes_bound_tightness_and_kde_crosscheck.csv').set_index('modality')
mods=['rgb','ir','fusion']; labels=['RGB','IR','RGB+IR']; xx=np.arange(3); w=.22
ax=axs[2]
ax.bar(xx-w,cross.loc[mods,'bhattacharyya_bound_s0.5'],w,label='Bhattacharyya',color=BLUE)
ax.bar(xx,cross.loc[mods,'chernoff_bound_optimized'],w,label='Chernoff',color=NAVY)
ax.bar(xx+w,cross.loc[mods,'kde_nonparametric_bound'],w,label='KDE',color=CYAN)
ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylabel('error upper bound / estimate')
ax.set_title('12--16 px cross-check',pad=4)
ax.legend(frameon=False,fontsize=6.6,loc='upper left')
panel_label(ax,'c'); light_grid(ax)

sens=pd.read_csv(DATA/'representation_sensitivity_12_16.csv')
ax=axs[3]
ax.plot(sens.n_components_per_modality,sens.rgb_bound,color=BLUE,marker='o',ms=3.6,lw=1.45,label='RGB')
ax.plot(sens.n_components_per_modality,sens.ir_bound,color=GOLD,marker='s',ms=3.6,lw=1.45,label='IR')
ax.plot(sens.n_components_per_modality,sens.fusion_bound,color=NAVY,marker='^',ms=3.8,lw=1.6,label='RGB+IR')
ax.set_xlabel('PCA components per modality'); ax.set_ylabel('error upper bound')
ax.set_title('representation sensitivity',pad=4)
ax.legend(frameon=False,fontsize=6.6)
ax.text(.98,.08,'joint dimension = 2k',transform=ax.transAxes,ha='right',fontsize=6.5,color=GRAY)
panel_label(ax,'d'); light_grid(ax)
fig.tight_layout(w_pad=1.25)
save(fig,'fig_scale_analysis')

# -----------------------------------------------------------------------------
# Figure 6: corruption response plus two validation summaries.
# -----------------------------------------------------------------------------
cor=pd.read_csv(DATA/'corruption_sweep.csv')
clean={m:float(cor[(cor.modality==m)&(cor.corruption_family=='clean')].iloc[0].bayes_error_bound) for m in ['rgb','ir']}
families=['gaussian_noise','speckle_noise','salt_and_pepper','gaussian_blur','uneven_illumination','motion_blur','camera_instability']
pretty={'gaussian_noise':'Gaussian noise','speckle_noise':'Speckle noise','salt_and_pepper':'Salt-and-pepper','gaussian_blur':'Gaussian blur','uneven_illumination':'Uneven illumination','motion_blur':'Motion blur','camera_instability':'Camera instability'}
fig, axs=plt.subplots(3,3,figsize=(10.2,7.0),sharey=False); axs=axs.ravel()
for i,fam in enumerate(families):
    ax=axs[i]
    for mod,color,mark,label in [('rgb',BLUE,'o','RGB'),('ir',GOLD,'s','IR')]:
        d=cor[(cor.modality==mod)&(cor.corruption_family==fam)].sort_values('severity')
        x=np.r_[0,d.severity.values]; y=np.r_[clean[mod],d.bayes_error_bound.values]
        ax.plot(x,y,color=color,marker=mark,ms=3.2,lw=1.35,label=label)
    ax.set_title(pretty[fam],pad=3)
    ax.set_xticks([0,1,2,3,4,5]); ax.set_xlabel('severity')
    ax.set_ylim(0.12,0.40)
    if i%3==0: ax.set_ylabel('error upper bound')
    panel_label(ax,chr(ord('a')+i),x=-.02,y=1.10); light_grid(ax)
axs[0].legend(frameon=False,fontsize=6.8,loc='upper left')

# h: average severity-5 penalty
ax=axs[7]
summ=pd.read_csv(DATA/'corruption_severity5_summary.csv').set_index('modality')
vals=[summ.loc['rgb','mean_severity5_penalty'],summ.loc['ir','mean_severity5_penalty']]
ax.bar([0,1],vals,color=[BLUE,GOLD],width=.52,zorder=3)
ax.set_xticks([0,1]); ax.set_xticklabels(['RGB','IR']); ax.set_ylabel('mean severity-5 penalty')
ax.set_title('Across seven families',pad=3)
ax.text(.5,.91,'1.90x vs 1.32x clean bound',transform=ax.transAxes,ha='center',fontsize=6.8,color=GRAY)
panel_label(ax,'h',x=-.02,y=1.10); light_grid(ax)

# i: noise-floor validation for Gaussian severity 1
ax=axs[8]
nf=pd.read_csv(DATA/'noise_floor_check.csv').set_index('modality')
med=[nf.loc['RGB','clean_full_frame_median_sigma'],nf.loc['IR','clean_full_frame_median_sigma']]
xx=np.array([0,1]);
ax.bar(xx-.16,med,width=.32,color=[BLUE,GOLD],label='clean median')
ax.bar(xx+.16,[5,5],width=.32,color=[CYAN,PALEGOLD],edgecolor=[BLUE,GOLD],linewidth=.7,label='AVID-C severity 1')
ax.set_yscale('log'); ax.set_xticks(xx); ax.set_xticklabels(['RGB','IR']); ax.set_ylabel('estimated / injected sigma (0--255)')
ax.set_title('Noise-floor check',pad=3)
ax.text(.02,.89,'16.6x',transform=ax.transAxes,color=BLUE,fontsize=7.1)
ax.text(.54,.89,'7.8x',transform=ax.transAxes,color=GOLD,fontsize=7.1)
ax.legend(frameon=False,fontsize=6.1,loc='lower right')
panel_label(ax,'i',x=-.02,y=1.10); light_grid(ax)
fig.tight_layout(w_pad=1.1,h_pad=1.2)
save(fig,'fig_corruption_analysis')

# -----------------------------------------------------------------------------
# Figure 7: controlled weak-alignment stress, including normalized retention.
# -----------------------------------------------------------------------------
al=pd.read_csv(DATA/'alignment_sensitivity_normalized.csv')
al=al[al.injected_shift_px<=12].copy()
fig,axs=plt.subplots(1,3,figsize=(9.8,3.05))
ax=axs[0]
ax.plot(al.injected_shift_px,al.bayes_fusion_gain,color=NAVY,marker='o',lw=1.65,ms=4)
ax.set_xlabel('per-axis shift budget (px)'); ax.set_ylabel('fusion-gap diagnostic')
ax.annotate('0.111',xy=(0,al.iloc[0].bayes_fusion_gain),xytext=(1.3,.113),fontsize=6.8,color=NAVY)
ax.annotate('0.093',xy=(12,al.iloc[-1].bayes_fusion_gain),xytext=(8.2,.089),fontsize=6.8,color=NAVY)
panel_label(ax,'a'); light_grid(ax)

ax=axs[1]
ax.plot(al.injected_shift_px,al.fusion_gap_retention,color=GREEN,marker='o',lw=1.65,ms=4)
ax.axhline(1,color=LIGHTGRAY,lw=.7)
ax.set_xlabel('per-axis shift budget (px)'); ax.set_ylabel('fusion-gap retention')
ax.set_ylim(.80,1.02)
ax.text(.97,.10,'84.4% retained at 12 px',transform=ax.transAxes,ha='right',fontsize=7.0,color=GREEN)
panel_label(ax,'b'); light_grid(ax)

ax=axs[2]
ax.plot(al.injected_shift_px,al.mutual_information_nats,color=BLUE,marker='o',lw=1.65,ms=4)
ax.set_xlabel('per-axis shift budget (px)'); ax.set_ylabel('estimated label information (nats)')
ax.set_ylim(.63,.73)
ax.text(.97,.10,'9.5% lower at 12 px',transform=ax.transAxes,ha='right',fontsize=7.0,color=BLUE)
panel_label(ax,'c'); light_grid(ax)
fig.tight_layout(w_pad=1.6)
save(fig,'fig_alignment_stress')

print('Wrote all paper figures.')
