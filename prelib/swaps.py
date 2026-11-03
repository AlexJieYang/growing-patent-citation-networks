import numpy as np
import pandas as pd

class TemporalSwapChain:
    def __init__(self, edges, days, years, seed, max_age_days=None, proposal_batch=32768):
        self.edges = np.asarray(edges, dtype="int32").copy()
        self.days = np.asarray(days)
        self.years = np.asarray(years)
        self.max_age_days = max_age_days
        self.proposal_batch = int(proposal_batch)
        self.rng = np.random.default_rng(seed)
        self.edge_set = set(map(tuple, self.edges.tolist()))
        self.buckets = {}
        self.keys = []
        for i, (s, t) in enumerate(self.edges):
            key = (int(years[s]), int(years[t]))
            self.buckets.setdefault(key, []).append(i)
            self.keys.append(key)
        self.buckets = {k: np.asarray(v, dtype=int) for k, v in self.buckets.items()}
        self.bucket_id = np.empty(len(self.edges), dtype="int32")
        bucket_arrays = list(self.buckets.values())
        for i, idx in enumerate(bucket_arrays): self.bucket_id[idx] = i
        self.bucket_sizes = np.array([len(x) for x in bucket_arrays], dtype=int)
        self.bucket_ptr = np.r_[0, np.cumsum(self.bucket_sizes)]
        self.bucket_indices = np.concatenate(bucket_arrays)
        self.base_out = np.bincount(self.edges[:, 0], minlength=len(days))
        self.base_in = np.bincount(self.edges[:, 1], minlength=len(days))
        self.original_sources = self.edges[:, 0].copy()
        self.target_year_per_slot = self.years[self.edges[:, 1]].copy()
        self.year_span = int(self.years.max()-self.years.min()+1)
        self.incoming_year_profile = np.bincount(self.edges[:, 1].astype("int64")*self.year_span
                                            + self.years[self.edges[:, 0]]-self.years.min(), minlength=len(days)*self.year_span)
        self.base_year_pair = self.year_pair_counts()
        self.assert_invariants()

    def year_pair_counts(self):
        sy = self.years[self.edges[:, 0]]-self.years.min()
        ty = self.years[self.edges[:, 1]]-self.years.min()
        return np.bincount(sy*self.year_span+ty, minlength=self.year_span**2)

    def assert_invariants(self):
        e = self.edges
        assert len(self.edge_set) == len(e), "duplicate edge in null graph"
        assert np.all(e[:, 0] != e[:, 1]), "self edge in null graph"
        assert np.all(self.days[e[:, 1]] < self.days[e[:, 0]]), "future/same-day citation in null graph"
        if self.max_age_days is not None:
            assert np.all(self.days[e[:, 0]]-self.days[e[:, 1]] <= self.max_age_days)
        assert np.array_equal(np.bincount(e[:, 0], minlength=len(self.days)), self.base_out)
        assert np.array_equal(np.bincount(e[:, 1], minlength=len(self.days)), self.base_in)
        assert np.array_equal(e[:, 0], self.original_sources)
        assert np.array_equal(self.years[e[:, 1]], self.target_year_per_slot)
        incoming = np.bincount(e[:, 1].astype("int64")*self.year_span+self.years[e[:, 0]]-self.years.min(),
                               minlength=len(self.days)*self.year_span)
        assert np.array_equal(incoming, self.incoming_year_profile)
        assert np.array_equal(self.year_pair_counts(), self.base_year_pair)

    def step(self, proposals):
        # Record after a FIXED number of proposals, including rejections.
        # Sampling after a fixed number of accepted moves instead would sample
        # the jump chain and overweight states with higher acceptance rates.
        limit = int(proposals)
        success = 0
        E = len(self.edges)
        for start in range(0, limit, self.proposal_batch):
            count = min(self.proposal_batch, limit-start)
            a_batch = self.rng.integers(E, size=count)
            bucket = self.bucket_id[a_batch]
            offset = self.rng.integers(self.bucket_sizes[bucket])
            b_batch = self.bucket_indices[self.bucket_ptr[bucket]+offset]
            # Bucket membership is invariant, so batching proposals changes no transition law.
            for a,b in zip(a_batch,b_batch):
                if a == b: continue
                s,t = map(int,self.edges[a]); u,v = map(int,self.edges[b])
                if s == u or t == v or s == v or u == t: continue
                first,second = (s,v),(u,t)
                if first in self.edge_set or second in self.edge_set: continue
                if not (self.days[v] < self.days[s] and self.days[t] < self.days[u]): continue
                if self.max_age_days is not None and (self.days[s]-self.days[v] > self.max_age_days
                                                      or self.days[u]-self.days[t] > self.max_age_days): continue
                self.edge_set.remove((s,t)); self.edge_set.remove((u,v))
                self.edge_set.update([first,second])
                self.edges[a,1],self.edges[b,1] = v,t
                success += 1
        self.assert_invariants()
        return {"fixed_proposals": limit, "accepted_swaps": success, "attempts": limit,
                "accepted_swaps_per_edge": success / E,
                "proposal_acceptance": success / limit if limit else 0.}


def permutation_strata(nodes, minimum_year=None):
    return [idx[np.asarray(nodes.vector_valid.iloc[idx],bool)] if 'vector_valid' in nodes else idx for idx in nodes.groupby(["grant_year","tech_code"],sort=False).indices.values()
            if minimum_year is None or nodes.iloc[idx[0]].grant_year >= minimum_year]


def stratified_permutation(nodes, rng, minimum_year=None, strata=None):
    permutation = np.arange(len(nodes))
    for idx in (strata if strata is not None else permutation_strata(nodes,minimum_year)):
        permutation[idx] = rng.permutation(idx)
    return permutation


def empirical_two_sided(observed, draws):
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if not np.isfinite(observed) or not len(d) or np.std(d) < 1e-14:
        return np.nan
    lower = (1 + np.sum(d <= observed)) / (len(d) + 1)
    upper = (1 + np.sum(d >= observed)) / (len(d) + 1)
    return min(1., 2 * min(lower, upper))


