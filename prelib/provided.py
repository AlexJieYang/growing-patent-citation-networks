"""Replot only the uploaded empirical aggregate tables, without inventing runs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .publication import style,plate,export,FIELDS


def make_provided_figures(project=None):
    root=Path(project or Path(__file__).resolve().parents[1]);data=root/'provided_results/data';dest=root/'provided_results/figures'
    top=pd.read_csv(data/'topology_summary.csv').set_index('domain');null=pd.read_csv(data/'topology_null_summary.csv')
    cfg={'figure_formats':['pdf','svg','png'],'figure_dpi':600,'synthetic_data':False};style();files=[]
    domains=['neural','laser'];fig,a=plate(3)
    annual=pd.read_csv(data/'cohort_statistics.csv')
    for i,(domain,color) in enumerate(zip(domains,FIELDS)):
        q=annual.loc[annual.domain.eq(domain)].sort_values('year')
        a[0,0].plot(q.year,q.patents,color=color,label=domain,ls=['-','--'][i])
        stat=top.loc[domain]
        for ax,key,label in [(a[0,1],'mean_degree','Mean undirected degree'),
            (a[1,0],'indegree_gini','Incoming-citation Gini'),(a[1,1],'giant_fraction','Largest component / all patents'),
            (a[2,0],'mean_clustering','Mean clustering'),(a[2,1],'max_core','Maximum coreness')]:
            ax.bar(i,stat[key],color=color,width=.58);ax.set(xticks=[0,1],xticklabels=domains,ylabel=label)
    a[0,0].set(xlabel='Grant year',ylabel='Annual focal patents');a[0,0].legend()
    files.append(export(fig,dest,'Provided01_Network_Summary',
      'Actual uploaded aggregate results through 2025-12-30. (a) Focal-patent arrivals in 2000-2025. (b-f) Complete-network descriptive statistics. This plate redraws the supplied tables; it is not a new patent-level run.',cfg))
    fig,a=plate(2)
    for ax,(metric,label) in zip(a.flat,[('transitive_closure','Transitive closure'),('path_degeneracy','Two-step route pairs'),
                                      ('cross_edge_share','Cross-community fraction'),('mean_edge_distance','Mean semantic distance')]):
        for i,(domain,color) in enumerate(zip(domains,FIELDS)):
            q=null.loc[null.domain.eq(domain)&null.statistic.eq(metric)].iloc[0]
            ax.plot([i-.1,i+.1],[q.null_mean,q.observed],color=color,lw=1)
            ax.scatter(i-.1,q.null_mean,marker='s',facecolor='white',edgecolor=color,s=32,label='Null mean' if i==0 else None)
            ax.scatter(i+.1,q.observed,marker='D',color=color,s=31,label='Observed' if i==0 else None)
        ax.set(xticks=[0,1],xticklabels=domains,ylabel=label,xlim=(-.4,1.4))
        if metric=='path_degeneracy':ax.ticklabel_format(axis='y',style='sci',scilimits=(3,3))
    a[0,0].legend()
    files.append(export(fig,dest,'Provided02_Ensemble_Comparison',
      'Actual observations and temporal-null means from the uploaded tables. Intervals are omitted because replicate numeric values are not available in these aggregate tables. The newly executed workflow exports the complete draw distribution.',cfg))
    fig,a=plate(2);heat=pd.read_csv(data/'heat_excerpt.csv')
    for col,(domain,color) in enumerate(zip(domains,FIELDS)):
        q=heat.loc[heat.domain.eq(domain)&heat.tau.le(100)].sort_values('tau')
        a[0,col].plot(q.tau,q.return_probability,'o-',color=color,markersize=3)
        a[0,col].fill_between(q.tau,q.return_low,q.return_high,color=color,alpha=.14,linewidth=0)
        n=round(top.loc[domain,'nodes']*top.loc[domain,'giant_fraction']);se=1.96*q.entropy_se/np.log(n)
        a[1,col].plot(q.tau,q.entropy_normalized,'o-',color=color,markersize=3)
        a[1,col].fill_between(q.tau,np.maximum(0,q.entropy_normalized-se),np.minimum(1,q.entropy_normalized+se),color=color,alpha=.14,linewidth=0)
        a[0,col].set(xscale='log',yscale='log',xlabel=r'Diffusion scale $\tau$',ylabel='Heat return per vertex')
        a[1,col].set(xscale='log',xlabel=r'Diffusion scale $\tau$',ylabel=r'Entropy $S/\ln n$',ylim=(-.02,1.02))
    files.append(export(fig,dest,'Provided03_Diffusion_Summary',
      'Left: neural; right: laser. Actual tabulated heat-return and entropy values at the supplied scales up to 100, joined as a visual guide. Bands use the supplied probe uncertainties. These older estimates have not undergone the new convergence gate; rerun notebook 03 for the revised spectral analysis.',cfg))
    return files
