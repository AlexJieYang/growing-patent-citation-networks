"""Independent closed forms and graph-library checks of scientific algorithms."""
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import pytest
from scipy import sparse
from prelib.metrics import (dag_subgraphs,local_diversity,partition_nmi,
    network_statistics,mixing,adjacency)
from prelib.percolation import site_curve,bond_curve
from prelib.spectral import heat_slq,lanczos_measure
from prelib.kinetics import opportunity_estimates,kaplan_meier
from prelib.swaps import TemporalSwapChain,stratified_permutation,empirical_two_sided


@pytest.mark.parametrize('seed',range(8))
def test_dag_triads_against_induced_census(seed):
    rng=np.random.default_rng(seed);n=12
    pairs=np.array([(i,j) for i in range(n) for j in range(i) if rng.random()<.28],int).reshape(-1,2)
    graph=nx.DiGraph();graph.add_nodes_from(range(n));graph.add_edges_from(pairs)
    exact=nx.triadic_census(graph);ours=dag_subgraphs(pairs,n,batch=3)
    for key in ['030T','021D','021U','021C']:assert ours[key]==exact[key]
    two=list(nx.all_simple_paths(graph,11,0,cutoff=2))
    degeneracy=0
    for a in range(n):
        for b in range(n):
            count=sum(1 for v in graph.successors(a) if graph.has_edge(v,b))
            degeneracy+=count*(count-1)//2
    assert ours['path_degeneracy']==degeneracy


def test_diamond_and_reference_pair_identity():
    edges=np.array([[3,1],[3,2],[1,0],[2,0]])
    assert dag_subgraphs(edges,4)['path_degeneracy']==1
    vec=np.array([[1,0],[0,1],[-1,0],[0,-1]],float)
    c=np.array([0,0,1,1]);B,S,bc,sc=local_diversity(edges,vec,c)
    assert S[3]==pytest.approx(1-vec[1]@vec[2])
    assert B[3]==pytest.approx(.5)
    vec[2]=np.nan;c[2]=-1
    B,S,bc,sc=local_diversity(edges,vec,c)
    assert np.isnan(B[3]) and np.isnan(S[3])
    assert bc[3]==.5 and sc[3]==.5


def test_distinct_semantic_and_community_thresholds():
    e=np.array([[3,0],[3,1],[3,2]])
    v=np.array([[1,0],[0,1],[np.nan,np.nan],[1,0]])
    B,S,bc,sc=local_diversity(e,v,np.array([0,1,1,0]),.9,.6)
    assert np.isfinite(B[3]) and S[3]==pytest.approx(1)
    assert bc[3]==1 and sc[3]==pytest.approx(2/3)


def test_partition_information_and_modularity():
    a=np.array([0,0,1,1]);assert partition_nmi(a,a+10)==pytest.approx(1)
    assert partition_nmi(a,[0,1,0,1])==pytest.approx(0,abs=1e-14)
    e=np.array([[1,0],[3,2],[3,1]])
    v=np.eye(4);years=np.arange(4)+2000
    ours=network_statistics(e,v,a,years)['frozen_modularity']
    G=nx.Graph();G.add_nodes_from(range(4));G.add_edges_from(e)
    assert ours==pytest.approx(nx.community.modularity(G,[{0,1},{2,3}]))
    c=a.copy();c[0]=-1
    assert mixing(e,c)[-1].sum()==2


def components_brute(G):
    sizes=sorted([len(p) for p in nx.connected_components(G)],reverse=True)
    L=sizes[0] if sizes else 0;S=sizes[1] if len(sizes)>1 else 0
    rest=sizes[1:];chi=sum(x*x for x in rest)/sum(rest) if sum(rest) else 0
    return L,S,chi


@pytest.mark.parametrize('seed',range(5))
def test_percolation_matches_full_recomputation(seed):
    G=nx.gnp_random_graph(17,.16,seed=seed);n=len(G);e=np.array(G.edges(),int).reshape(-1,2)
    rng=np.random.default_rng(seed);order=rng.permutation(n);grid=np.linspace(0,1,n+1)
    out=site_curve(e,n,order,grid)
    for row in out.itertuples():
        H=G.copy();H.remove_nodes_from(order[:int(round(row.fraction*n))])
        L,S,chi=components_brute(H)
        assert (row.largest,row.second)==(L,S)
        assert row.susceptibility==pytest.approx(chi)
        assert row.giant_fraction==pytest.approx(L/n)
    edge_order=rng.permutation(len(e));out=bond_curve(e,n,edge_order,np.linspace(0,1,len(e)+1))
    for row in out.itertuples():
        H=nx.Graph();H.add_nodes_from(range(n));H.add_edges_from(e[edge_order[:row.retained_edges]])
        L,S,chi=components_brute(H)
        assert (row.largest,row.second)==(L,S)
        assert row.susceptibility==pytest.approx(chi)


def test_empty_percolation_and_disconnected_ties():
    e=np.empty((0,2),int)
    assert bond_curve(e,3,[],[0,.5,1]).largest.eq(1).all()
    q=site_curve([[0,1],[2,3]],4,[0,1,2,3],[0,.5,1])
    assert q.iloc[0].largest==2 and q.iloc[0].second==2
    assert q.iloc[0].susceptibility==2
    assert q.iloc[-1].largest==0


def test_heat_complete_graph_closed_form():
    n=9;B=np.ones((n,n))-np.eye(n);tau=np.r_[0,np.geomspace(.01,50,20)]
    result,lam,weight=heat_slq(B,tau,probes=4,steps=n,seed=8)
    eigen=n/(n-1);Z=1+(n-1)*np.exp(-tau*eigen)
    E=(n-1)*eigen*np.exp(-tau*eigen)/Z
    C=tau**2*((n-1)*eigen**2*np.exp(-tau*eigen)/Z-E**2)
    np.testing.assert_allclose(result.return_probability,Z/n,atol=2e-14)
    np.testing.assert_allclose(result.entropy,np.log(Z)+tau*E,atol=2e-13)
    np.testing.assert_allclose(result.heat_capacity,C,atol=2e-13)
    assert result.entropy.iloc[0]==pytest.approx(np.log(n))
    assert sum(weight)==pytest.approx(1)


def test_lanczos_trace_against_dense_spectrum():
    B=nx.to_numpy_array(nx.path_graph(8));d=B.sum(1);z=np.sqrt(d);z/=np.linalg.norm(z)
    L=np.eye(8)-(B/np.sqrt(d[:,None]*d[None,:]));taus=np.array([0,.1,1,10])
    trace=np.ones(len(taus))
    for e in np.eye(8):
        x,w,norm=lanczos_measure(sparse.csr_matrix(L),e,8,z)
        trace+=np.exp(-taus[:,None]*x)@(w*norm)
    exact=np.exp(-taus[:,None]*np.linalg.eigvalsh(L)).sum(1)
    np.testing.assert_allclose(trace,exact,atol=5e-13)
    with pytest.raises(ValueError):heat_slq(np.eye(4),[1])
    with pytest.raises(ValueError):heat_slq([[0,1],[0,0]],[1])


def test_opportunity_ht_matches_enumerated_uniform_risk():
    # Each case has four targets with strata [0,0,1,1], K=1.
    # Repeating all three possible controls makes the HT expectation exact.
    rows=[];risk=[];sources=[]
    labels=np.array([0,0,1,1]);r=0
    for chosen in range(4):
        for control in range(4):
            if control==chosen:continue
            rows.append([labels[chosen],labels[control]]);risk.append(4);sources.append(r);r+=1
    q=opportunity_estimates(np.array(rows),risk,sources,2,19)
    np.testing.assert_allclose(q.expected_uniform,[6,6])
    np.testing.assert_allclose(q.enrichment,[1,1])
    assert q.expected_uniform.sum()==pytest.approx(len(rows))
    q=opportunity_estimates(np.array([[0,1]]),[4],[1],2,0)
    assert q.enrichment.isna().all()
    assert opportunity_estimates(np.empty((0,3),int),[],[],3).observed.sum()==0


def test_kaplan_meier_ties_and_censoring():
    q=kaplan_meier([1,2,2,3],[1,1,0,0])
    assert q.loc[q.days.eq(1),'survival'].iloc[0]==pytest.approx(.75)
    assert q.loc[q.days.eq(2),'survival'].iloc[0]==pytest.approx(.5)
    assert q.loc[q.days.eq(2),'at_risk'].iloc[0]==3
    assert (q.low<=q.survival).all() and (q.high>=q.survival).all()
    assert kaplan_meier([1],[1]).survival.iloc[-1]==0
    assert kaplan_meier([1,2],[0,0]).survival.eq(1).all()


def test_temporal_swap_constraints_and_fixed_proposals():
    days=np.arange(16)*40;years=np.repeat([2000,2001],8)
    e=np.array([(s,t) for s in range(8,16) for t in range(8) if (s+t)%3==0])
    chain=TemporalSwapChain(e,days,years,21,1000,proposal_batch=31)
    original=chain.edges.copy();result=chain.step(1001)
    assert result['fixed_proposals']==1001==result['attempts']
    assert result['accepted_swaps']>0
    assert not np.array_equal(chain.edges,original)
    chain.assert_invariants()


def test_swap_stationarity_on_enumerated_state_space():
    # Enumerate the whole 3x3 bipartite degree-regular space: six matchings.
    # Fixed proposals, including self proposals, should sample it uniformly.
    e=np.array([[3,0],[4,1],[5,2]]);days=np.r_[np.zeros(3),np.ones(3)]
    years=np.r_[np.repeat(2000,3),np.repeat(2001,3)]
    chain=TemporalSwapChain(e,days,years,913,proposal_batch=64)
    counts={p:0 for p in itertools.permutations(range(3))}
    for _ in range(6000):
        chain.step(4);counts[tuple(chain.edges[:,1])]+=1
    assert max(abs(x-1000) for x in counts.values())<180


def test_semantic_permutation_keeps_baseline_invalids_and_strata():
    nodes=pd.DataFrame({'grant_year':[1999]*3+[2001]*6,
                        'tech_code':['A']*5+['B']*4,
                        'vector_valid':[True,True,True,True,False,True,True,True,False]})
    p=stratified_permutation(nodes,np.random.default_rng(1),2000)
    np.testing.assert_array_equal(p[:3],np.arange(3))
    np.testing.assert_array_equal(p[[4,8]],[4,8])
    np.testing.assert_array_equal(nodes.tech_code.to_numpy()[p],nodes.tech_code)
    np.testing.assert_array_equal(nodes.grant_year.to_numpy()[p],nodes.grant_year)
    assert np.isnan(empirical_two_sided(1,[1,1,1]))
    assert empirical_two_sided(100,np.arange(99))==pytest.approx(.02)
