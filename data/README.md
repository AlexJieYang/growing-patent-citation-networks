# Input data

The workflow reads processed USPTO/PatentsView utility-patent records and precomputed embeddings. It does not download or redistribute the patent tables. Set `source_root` in `config.local.json` or use `--source-root`.

| Path relative to source_root | Required columns or shape |
|---|---|
| `02_core/patent_dates.parquet` | `patent_id`, `patent_date`, `filing_date` |
| `02_core/patent_primary_field.parquet` | `patent_id`, `cpc_subclass`, `cpc_group` |
| `02_core/citation.parquet` | `patent_id` (source), `citation_patent_id` (target) |
| `06_embedding_metrics/patentsberta_v1/02_cache/index.parquet` | `patent_id`, `embedding_row` |
| `06_embedding_metrics/patentsberta_v1/02_cache/vectors.npy` | Two-dimensional numeric matrix addressed by `embedding_row` |

Embedding paths and the row-column name are configurable. Relative `source_root` is resolved against the repository directory; relative embedding paths are resolved under `source_root`. Absolute paths are also supported.

## Identifiers and alignment

- Patent identifiers must be numeric utility-grant identifiers stored as integers or digit strings; floating-point identifiers are rejected.
- Date and primary-field dimensions require unique patent identifiers.
- The embedding index must map identifiers to actual physical matrix rows. Sorting identifiers does not imply that same vector order.
- Internal edges require target grant day strictly earlier than source grant day. Self-links, same-day/future citations, and duplicate pairs are excluded.
- Finite, nonzero vectors are normalized. Unknown communities and invalid vectors remain distinct from observed zeros, and their vertices remain in the full census.

The paper uses 768-dimensional PatentSBERTa embeddings of concatenated titles and abstracts. The code consumes cached vectors and does not encode text or select a checkpoint. Preserve encoder settings and the original patent-to-row mapping when preparing the cache. The generated example instead uses artificial 32-dimensional vectors and is marked synthetic.

## Optional inputs

`edge_source = "applicant"` or `"examiner"` selects `citation_applicant.parquet` or `citation_examiner.parquet`, using the same endpoint columns.

`cpc_match_mode = "any"` also requires `02_core/patent_cpc.parquet` with `patent_id`, `cpc_subclass`, and `cpc_group`.

`prepared_data_root` may point to an existing run directory containing `data/nodes.parquet`, `reference_edges.parquet`, `forward_edges.parquet`, `internal_edges.parquet`, `vectors.npy`, and `observation_window.json`. Reuse also checks its citation-source/follow-up declaration (`prepared_scope.json`, or the supported legacy overview), CPC/date census, observation end, and requested scope. It avoids citation rescanning but still requires CPC/date metadata.

## Coverage and storage

The effective observation end is bounded by the requested cutoff, source coverage, and metadata maximum grant date. It is saved in `outputs/<run_name>/data/observation_window.json`.

Keep real inputs in `data/raw/` or an external directory. Parquet/NPY/NPZ matrices and `data/raw/` are ignored by Git. Source records are distributed through the [USPTO PatentsView program](https://www.uspto.gov/ip-policy/economic-research/patentsview).
