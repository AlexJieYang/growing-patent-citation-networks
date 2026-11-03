import numpy as np
import networkx as nx
import pytest
from prelib.metrics import transitive_triangles
from prelib.growth_core import target_collision
from prelib.diffusion import diffusion_profile


def test_target_collision_equals_enumerated_distinct_pairs():
    targets=np.array([0,0,0,1,1,2]);k=np.bincount(targets)
    pairs=[targets[i]==targets[j] for i in range(len(targets)) for j in range(i)]
    assert target_collision(k)==pytest.approx(np.mean(pairs))
    assert target_collision([1,1,1])==0
    assert np.isnan(target_collision([1]))


def test_incremental_transitive_triples_match_induced_census():
    fixed=np.array([[2,0],[3,2],[4,1]]);new=np.array([[5,3],[5,2],[3,0]])
    def census(e):
        g=nx.DiGraph();g.add_nodes_from(range(6));g.add_edges_from(e)
        return nx.triadic_census(g)['030T']
    full=np.vstack([fixed,new])
    assert transitive_triangles(full,6)-transitive_triangles(fixed,6)==census(full)-census(fixed)


def test_diffusion_dense_and_slq_follow_complete_graph_closed_form():
    n=12;B=np.ones((n,n))-np.eye(n);tau=np.geomspace(.01,100,18)
    cfg={'spectral_exact_limit':20,'slq_probes':8,'slq_steps':12,'spectral_coarse_probes':4,'spectral_coarse_steps':8,'seed':2}
    exact=diffusion_profile(B,tau,cfg);approx=diffusion_profile(B,tau,{**cfg,'spectral_exact_limit':0})
    Z=1+(n-1)*np.exp(-tau*n/(n-1))
    np.testing.assert_allclose(exact.return_probability,Z/n,atol=1e-13)
    np.testing.assert_allclose(approx.return_probability,exact.return_probability,atol=1e-13)
    np.testing.assert_allclose(approx.entropy_normalized,exact.entropy_normalized,atol=1e-12)
    assert approx.supported.all() and exact.supported.all()


def test_diffusion_gate_is_a_prefix_with_shared_probes():
    B=nx.to_numpy_array(nx.path_graph(35));tau=np.geomspace(.01,100,30)
    cfg={'spectral_exact_limit':0,'slq_probes':8,'slq_steps':25,'spectral_coarse_probes':4,'spectral_coarse_steps':8,
         'spectral_entropy_tolerance':1e-6,'spectral_return_tolerance':1e-6,'seed':21}
    q=diffusion_profile(B,tau,cfg)
    assert not q.supported.all()
    flags=q.supported.to_numpy();assert not np.any(flags[1:]&~flags[:-1])
