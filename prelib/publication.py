"""Journal-width plates with panel letters and separate, complete captions."""
from __future__ import annotations
from pathlib import Path
import os,json,string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator,FixedLocator,ScalarFormatter,NullFormatter
from PIL import Image
from .common import run_dir,shared_dir,domain_config
from .graphs import network_data
from .metrics import ccdf,cdf,mixing

FIELDS=['#24658B','#C36C35','#43836B','#8B5C94']
MODELS={'PA_Age':('#6D737A','o'),'Geometry':('#347F9A','s'),'Additive':('#43836B','^')}
MODEL_LABELS={'PA_Age':'Degree + age','Geometry':'+ Semantic cost','Additive':'+ Community cost'}
LINE_STYLES=['-','--',':','-.']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
      'axes.labelsize':9.5,'axes.titlesize':10.5,'xtick.labelsize':9,'ytick.labelsize':9,
      'legend.fontsize':9,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
      'axes.linewidth':.8,'lines.linewidth':1.35,'pdf.fonttype':42,'ps.fonttype':42,
      'svg.fonttype':'none','savefig.facecolor':'white','axes.formatter.use_mathtext':True})


def plate(rows=2,cols=2,height=None):
    fig,axes=plt.subplots(rows,cols,figsize=(7.05,height or 2.48*rows+.25),layout='constrained',squeeze=False)
    for i,ax in enumerate(axes.flat):
        ax.set_title(f'({string.ascii_lowercase[i]})',loc='left',fontweight='bold',pad=5)
        ax.tick_params(direction='out',length=3,width=.7)
    return fig,axes


def export(fig,dest,name,caption,cfg):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);caps=dest/'captions';caps.mkdir(exist_ok=True)
    if cfg.get('synthetic_data'):
        fig.text(.995,.995,'SYNTHETIC DATA',ha='right',va='top',fontsize=9,color='#A43C36',bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    formats=cfg.get('figure_formats',['pdf','svg','png'])
    try:
        for ext in formats:
            target=dest/f'{name}.{ext}';tmp=dest/f'_{name}.{ext}'
            try:
                # Windows commit requires write access. Keep the writable
                # stream open through save/flush/fsync, then close it before
                # checking or replacing the output.
                with tmp.open('w+b') as fh:
                    fig.savefig(fh,format=ext,dpi=cfg.get('figure_dpi',600),bbox_inches='tight',pad_inches=.04)
                    fh.flush()
                    os.fsync(fh.fileno())
                raw=tmp.read_bytes()
                if ext=='pdf' and b'%%EOF' not in raw[-1024:]:
                    raise OSError(f'Incomplete PDF export: {name}')
                if ext=='svg' and not raw.rstrip().endswith(b'</svg>'):
                    raise OSError(f'Incomplete SVG export: {name}')
                if ext=='png':
                    with Image.open(tmp) as im:im.verify()
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
        (caps/f'{name}.txt').write_text(caption+'\n',encoding='utf-8')
    finally:
        plt.close(fig)
    return dest/f'{name}.png'


def _network(ax,out,nodes):
    with np.load(out/'data/network_view.npz') as z:
        idx=z['nodes'];xy=z['xy'];e=z['edges']
    ax.add_collection(LineCollection(xy[e],colors='#88949A',linewidths=.22,alpha=.23,rasterized=True))
    c=nodes.community.to_numpy()[idx]
    color=[plt.colormaps['tab20'](int(i)%20) if i>=0 else (.65,.65,.65,1) for i in c]
    ax.scatter(xy[:,0],xy[:,1],c=color,s=2+2*np.log1p(nodes.in_degree.to_numpy()[idx]),linewidths=0,rasterized=True)
    ax.autoscale();ax.margins(.06);ax.set_axis_off()


def _band(ax,g,x,y,low,high,color,label=None,ls='-'):
    g=g.sort_values(x);xx=g[x].to_numpy(float);yy=g[y].to_numpy(float)
    ax.plot(xx,yy,color=color,label=label,ls=ls)
    ax.fill_between(xx,g[low].to_numpy(float),g[high].to_numpy(float),color=color,alpha=.14,linewidth=0)


def _domains(cfg):
    out=[]
    for i,spec in enumerate(cfg['domains']):
        path=run_dir(domain_config(cfg,spec['id']));nodes=pd.read_parquet(path/'data/node_statistics.parquet')
        edges=pd.read_parquet(path/'data/network_edges.parquet');out.append((spec,path,nodes,edges,FIELDS[i%len(FIELDS)]))
    return out


def atlas(cfg,domains,dest):
    if len(domains)!=2:raise ValueError('Publication layouts compare exactly two domains; set two CPC domains per run.')
    fig,a=plate(3);summary=pd.read_csv(shared_dir(cfg)/'tables/topology_summary.csv').set_index('domain')
    for i,(spec,out,nodes,edges,color) in enumerate(domains):
        _network(a[0,i],out,nodes)
        annual=pd.read_csv(out/'tables/annual_snapshots.csv')
        a[1,0].plot(annual.year,annual.new_nodes,color=color,ls=LINE_STYLES[i],label=spec['id'])
        stat=summary.loc[spec['id']];a[1,1].bar(np.array([0,1])+i*.32-.16,
            [stat.isolates/stat.nodes,stat.giant_fraction],width=.30,color=color,label=spec['id'])
        for key,ls in [('in_degree','-'),('out_degree','--')]:
            x,p=ccdf(nodes[key],positive=True);a[2,0].loglog(x,p,color=color,ls=ls,label=f"{spec['id']} {'in' if key=='in_degree' else 'out'}")
        k=np.sort(nodes.in_degree.to_numpy(float));a[2,1].plot(np.r_[0,np.arange(1,len(k)+1)/len(k)],np.r_[0,np.cumsum(k)/max(k.sum(),1)],color=color,ls=LINE_STYLES[i],label=spec['id'])
    a[1,0].set(xlabel='Grant year',ylabel='Annual patents');a[1,0].legend()
    a[1,1].set(xticks=[0,1],xticklabels=['Isolates','Largest component'],ylabel='Fraction of all patents',ylim=(0,1));a[1,1].legend()
    a[2,0].set(xlabel='Positive degree',ylabel='Conditional CCDF');a[2,0].legend(fontsize=8.5)
    a[2,1].plot([0,1],[0,1],':',color='.55');a[2,1].set(xlabel='Fraction of patents',ylabel='Fraction of internal citations')
    return export(fig,dest,'Main_Fig01_Networks',
      '(a,b) Neural and laser largest-component BFS views, respectively; sampled vertices and edges illustrate the connected cores. '
      '(c) Annual patent arrivals. (d) Isolate and largest-component fractions of all selected patents. '
      '(e) Positive in/out-degree CCDFs. (f) Incoming-citation Lorenz curves, including zero receivers. Numerical summaries use full networks.',cfg)


def topology_null(cfg,domains,dest):
    fig,a=plate(2);metrics=[('transitive_closure','Transitive closure'),('path_degeneracy','Two-step route pairs'),
        ('cross_edge_share','Cross-community fraction'),('mean_edge_distance','Mean semantic distance')]
    for ax,(metric,label) in zip(a.flat,metrics):
        for i,(spec,out,_,_,color) in enumerate(domains):
            q=pd.read_csv(out/'tables/null_summary.csv');q=q.loc[q.null_model.eq('Topology')&q.statistic.eq(metric)].iloc[0]
            draws=pd.read_csv(out/'tables/null_draws.csv');v=draws.loc[draws.null_model.eq('Topology'),metric].dropna()
            lo,hi=v.quantile([.025,.975]);ax.plot([i,i],[lo,hi],color=color,lw=2.0)
            ax.scatter(i,q.null_mean,marker='s',s=28,facecolor='white',edgecolor=color,zorder=3,label='Null mean' if i==0 else None)
            ax.scatter(i+.15,q.observed,marker='D',s=29,color=color,zorder=4,label='Observed' if i==0 else None)
        ax.set(xticks=[.075,1.075],xticklabels=[d[0]['id'] for d in domains],ylabel=label,xlim=(-.4,1.5))
        if metric=='path_degeneracy':ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3))
    a[0,0].legend(loc='best')
    return export(fig,dest,'Main_Fig02_Constrained_Ensemble',
      '(a) Closed fraction of directed two-step paths. (b) Pairs of two-step routes sharing endpoints. '
      '(c) Cross-community fraction on mapped edges. (d) Mean semantic distance on valid-vector edges. '
      'Diamonds are observations; squares and vertical intervals show the mean and central 95% range of 199 temporal-ensemble states (or the configured draw count). '
      'The four induced DAG triad classes share one fixed-degree closure coordinate and are not plotted as independent tests.',cfg)


def attachment(cfg,domains,dest):
    fig,a=plate(3)
    for col,(spec,out,_,_,color) in enumerate(domains):
        table=pd.read_csv(out/'tables/opportunity_enrichment.csv')
        for row,feature in enumerate(['degree','age','distance']):
            ax=a[row,col];q=table.loc[table.feature.eq(feature)&table.split.eq('test')]
            if feature=='distance':
                for b,ls in [(0,'-'),(1,'--')]:
                    g=q.loc[q.cross_community.eq(b)];_band(ax,g,'center','enrichment','low','high',color,'Within' if b==0 else 'Cross',ls)
                ax.set_xlabel('Semantic distance');ax.legend()
            else:
                q=q.sort_values('center');_band(ax,q,'center','enrichment','low','high',color)
                ax.set_xscale('log');ax.set_xlabel('Target in-degree + 1' if feature=='degree' else 'Target age (years)')
                if feature=='degree':
                    ax.lines[-1].set_xdata(q.center.to_numpy()+1)
                    # Rebuild the band at the shifted degree coordinates.
                    for coll in list(ax.collections):coll.remove()
                    ax.fill_between(q.center+1,q.low,q.high,color=color,alpha=.14,linewidth=0)
                    ax.xaxis.set_major_locator(FixedLocator([1,3,10,30,100,300]))
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axhline(1,color='.5',ls=':',lw=.8);ax.set_ylabel('Observed / uniform opportunity')
    return export(fig,dest,'Main_Fig03_Attachment',
      'Left column: neural; right: laser. (a,b) Degree enrichment, (c,d) age enrichment, '
      '(e,f) semantic-distance enrichment within and across frozen communities. '
      'All panels use the final configured time stratum (default 2023-2025). Bands are source-patent bootstrap intervals conditional on sampled candidates. Low-support bins remain masked.',cfg)


def growth(cfg,domains,dest):
    fig,a=plate(3);names=list(MODELS)
    for col,(spec,out,_,_,_) in enumerate(domains):
        q=pd.read_csv(out/'tables/growth_simulations.csv');obs=q.loc[q.model.eq('Observed')].iloc[0]
        for j,model in enumerate(names):
            color,marker=MODELS[model];g=q.loc[q.model.eq(model)]
            a[0,col].scatter(g.mean_edge_distance,g.cross_edge_share,color=color,marker=marker,s=11,alpha=.28)
            a[0,col].scatter(g.mean_edge_distance.mean(),g.cross_edge_share.mean(),color=color,marker=marker,s=46,label=MODEL_LABELS[model],edgecolor='white',linewidth=.5)
            for row,metric in [(1,'target_collision'),(2,'triads_per_new_edge')]:
                values=g[metric].dropna();lo,hi=values.quantile([.025,.975]);mean=values.mean()
                a[row,col].plot([j,j],[lo,hi],color=color,lw=2)
                a[row,col].scatter(j,mean,color=color,marker=marker,s=35,zorder=3)
        a[0,col].scatter(obs.mean_edge_distance,obs.cross_edge_share,marker='*',color='black',s=85,label='Observed',zorder=5)
        a[0,col].set(xlabel='Mean new-edge distance',ylabel='Cross-community fraction')
        if col==0:a[0,col].legend(fontsize=8.4,loc='best')
        for row,metric,label in [(1,'target_collision','Target-collision probability'),(2,'triads_per_new_edge','New transitive triples / new edge')]:
            a[row,col].axhline(obs[metric],color='black',ls='--',lw=1.)
            a[row,col].set(xticks=range(3),xticklabels=['Degree/age','+ Distance','+ Community'],ylabel=label)
            a[row,col].tick_params(axis='x',labelrotation=23)
            if metric=='target_collision':a[row,col].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    return export(fig,dest,'Main_Fig04_Growth',
      'Left: neural; right: laser. (a,b) Each realization and mechanism mean in distance-crossing coordinates; stars are observed citations. '
      '(c,d) Probability that two distinct new citations share a target. (e,f) Transitive triples containing at least one modeled new edge, per modeled edge. '
      'Vertical intervals span the central 95% of simulation realizations; dashed lines are observed. '
      'Three nested kernels use common fixed costs and preserve source reference counts, chronology and the nonmodeled graph. Default window: 2023-2025.',cfg)


def connectivity(cfg,domains,dest):
    fig,a=plate(2)
    styles={'Random':('#687C89','-'),'Degree':('#B56F43','--'),'Similar first':('#B56F43','--'),'Distant first':('#43836B',':')}
    for col,(_,out,_,_,_) in enumerate(domains):
        q=pd.read_csv(out/'tables/percolation_summary.csv')
        for row,family in enumerate(['site','bond']):
            for strategy,g in q.loc[q.family.eq(family)].groupby('strategy',sort=False):
                color,ls=styles[strategy];_band(a[row,col],g,'fraction','relative_mean','relative_low','relative_high',color,strategy,ls)
            a[row,col].set(xlabel='Removed node fraction' if row==0 else 'Inserted edge fraction',ylabel='Largest component / baseline',ylim=(0,1.03))
            if col==0:a[row,col].legend(fontsize=8.5)
    return export(fig,dest,'Main_Fig05_Connectivity',
      'Left: neural; right: laser. (a,b) Static random and initial-degree node removal. '
      '(c,d) Random, short-distance-first and long-distance-first edge insertion. '
      'Largest components are divided by their own intact-graph value, while fractions on the x-axis use all selected nodes or the common valid-distance edge set. '
      'Bands describe order/tie-breaking variation. Full-population fractions, relative AUC and half-size fractions are retained in tables.',cfg)


def supplements(cfg,domains,dest):
    files=[]
    # One common prefix of the convergence-supported diffusion grids.
    profiles=[pd.read_csv(out/'tables/diffusion_profile.csv') for _,out,_,_,_ in domains]
    limits=[q.loc[q.supported,'tau'].max() for q in profiles];common=float(np.min(limits))
    if not np.isfinite(common):raise RuntimeError('No common supported diffusion scale; inspect diffusion_profile.csv.')
    fig,a=plate(2)
    for col,(q,domain) in enumerate(zip(profiles,domains)):
        color=domain[-1];q=q.loc[q.tau<=common]
        _band(a[0,col],q,'tau','return_probability','return_low','return_high',color)
        a[0,col].set(xscale='log',yscale='log',xlabel=r'Diffusion scale $\tau$',ylabel='Heat return per vertex')
        se=1.96*q.entropy_normalized_se.to_numpy();a[1,col].plot(q.tau,q.entropy_normalized,color=color)
        a[1,col].fill_between(q.tau,np.maximum(0,q.entropy_normalized-se),np.minimum(1,q.entropy_normalized+se),color=color,alpha=.14,linewidth=0)
        a[1,col].set(xscale='log',xlabel=r'Diffusion scale $\tau$',ylabel=r'Entropy $S/\ln n$',ylim=(-.02,1.02))
    files.append(export(fig,dest,'SI_Fig01_Diffusion',
      f'Left: neural; right: laser. (a,b) Heat-return trace per vertex, (c,d) normalized spectral entropy on the largest connected components. Common displayed upper scale {common:.5g}; all requested scales and numerical changes remain in diffusion_profile.csv. Bands show probe variability. Small components use dense spectra.',cfg))
    fig,a=plate(2)
    for spec,out,nodes,edges,color in domains:
        q=pd.read_csv(out/'tables/annual_snapshots.csv')
        a[0,0].plot(q.year,q.mean_degree,color=color,label=spec['id']);a[0,1].plot(q.year,q.giant_fraction,color=color,label=spec['id'])
    for ax,label in zip(a[0],['Mean undirected degree','Largest component / all patents']):ax.set(xlabel='Grant year',ylabel=label);ax.legend()
    for col,(_,out,_,_,_) in enumerate(domains):
        q=pd.read_csv(out/'tables/degree_age_enrichment.csv');q=q.loc[q.split.eq('test')]
        z=q.pivot(index='age_bin',columns='degree_bin',values='enrichment').sort_index().to_numpy()
        masked=np.ma.masked_invalid(np.log2(z));im=a[1,col].imshow(masked,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
        a[1,col].set(xticks=range(7),xticklabels=['0','1','2-3','4-7','8-15','16-31','32+'],
                     yticks=range(6),yticklabels=['0-1','1-2','2-5','5-10','10-20','20-30'],xlabel='In-degree bin',ylabel='Age bin (years)')
        a[1,col].tick_params(axis='x',labelrotation=35);cb=fig.colorbar(im,ax=a[1,col],fraction=.045,pad=.035);cb.set_label(r'$\log_2$ enrichment',fontsize=9)
    files.append(export(fig,dest,'SI_Fig02_Growth_Context',
      '(a,b) Evolution of mean degree and largest-component fraction. (c,d) Final-period degree-age enrichment for neural and laser; white cells are low-support bins.',cfg))
    fig,a=plate(2)
    for col,(_,out,nodes,e,color) in enumerate(domains):
        for b,ls,label in [(0,'-','Within'),(1,'--','Cross')]:
            x,p=cdf(e.loc[e.cross_community.eq(b),'distance']);a[0,col].plot(x,p,color=color,ls=ls,label=label)
        a[0,col].set(xlabel='Semantic distance',ylabel='Empirical CDF');a[0,col].legend()
        _,p,*_=mixing(e[['s','t']].to_numpy(),nodes.community.to_numpy());im=a[1,col].imshow(p,origin='lower',cmap='Blues',aspect='equal')
        a[1,col].set(xlabel='Cited community',ylabel='Citing community');a[1,col].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5));a[1,col].yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        cb=fig.colorbar(im,ax=a[1,col],fraction=.045,pad=.035);cb.set_label('Joint edge share',fontsize=9)
    files.append(export(fig,dest,'SI_Fig03_Semantic_Geometry',
      'Left: neural; right: laser. (a,b) Within/cross distance distributions on the common mapped valid-vector subset. (c,d) Joint source-target community edge shares.',cfg))
    fig,a=plate(2)
    for col,(_,out,_,_,_) in enumerate(domains):
        table=pd.read_csv(out/'tables/growth_edge_curves.csv')
        for row,metric in enumerate(['positive_new_indegree','distance']):
            for model,g in table.loc[table.metric.eq(metric)].groupby('model',sort=False):
                if model=='Observed':q=g.sort_values('x');color='black';ls='-';label='Observed'
                else:
                    q=g.groupby('x',as_index=False).probability.mean();color=MODELS[model][0];ls=LINE_STYLES[list(MODELS).index(model)+1];label=MODEL_LABELS[model]
                if row==0:q=q.loc[q.probability.gt(0)]
                a[row,col].plot(q.x,q.probability,color=color,ls=ls,label=label)
            a[row,col].set(xlabel='Positive new in-degree' if row==0 else 'New-edge distance',ylabel='Conditional CCDF' if row==0 else 'Empirical CDF')
            if row==0:
                a[row,col].set(xscale='log',yscale='log')
                a[row,col].xaxis.set_major_locator(FixedLocator([1,3,10,30,100,300,1000]))
                a[row,col].xaxis.set_major_formatter(ScalarFormatter())
                a[row,col].xaxis.set_minor_formatter(NullFormatter())
        if col==0:a[1,col].legend(fontsize=8.5)
    files.append(export(fig,dest,'SI_Fig04_Growth_Distributions',
      'Left: neural; right: laser. (a,b) Positive new-in-degree CCDFs. (c,d) New-edge semantic-distance CDFs. Simulated curves average realizations; observed curves are black.',cfg))
    fig,a=plate(2)
    for col,(_,out,nodes,_,color) in enumerate(domains):
        im=a[0,col].hexbin(np.log1p(nodes.degree),nodes.clustering,gridsize=20,mincnt=1,cmap='Blues',linewidths=0,rasterized=True)
        a[0,col].set(xlabel=r'$\ln(1+\mathrm{degree})$',ylabel='Local clustering');cb=fig.colorbar(im,ax=a[0,col],fraction=.045,pad=.035);cb.set_label('Vertices',fontsize=9)
        x,count=np.unique(nodes.coreness,return_counts=True);a[1,col].bar(x,count/len(nodes),color=color)
        a[1,col].set(xlabel='Undirected coreness',ylabel='Fraction of all vertices');a[1,col].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
    files.append(export(fig,dest,'SI_Fig05_Hierarchy',
      'Left: neural; right: laser. (a,b) Clustering versus total degree. (c,d) Core-shell populations. All selected vertices, including isolates, enter these descriptive distributions.',cfg))
    return files


def make_figures(cfg):
    style();domains=_domains(cfg);dest=shared_dir(cfg)/'figures'
    files=[fun(cfg,domains,dest) for fun in [atlas,topology_null,attachment,growth,connectivity]]
    files.extend(supplements(cfg,domains,dest))
    records=[{'figure':p.stem,'file':str(p.relative_to(shared_dir(cfg))),
              'caption':str((dest/'captions'/f'{p.stem}.txt').relative_to(shared_dir(cfg)))} for p in files]
    pd.DataFrame(records).to_csv(dest/'figure_index.csv',index=False)
    return files
