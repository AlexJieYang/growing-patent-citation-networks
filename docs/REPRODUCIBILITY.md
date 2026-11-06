# Reproducing the analysis

The public configuration preserves the research parameters with `data/raw` as its portable input location. Notebook 00 and the command-line entry point save effective settings in `outputs/active_run_config.json`, plus run-specific copies in `outputs/<run_name>/models/config_used.json`. Later notebooks use the active run unless `PRE_CONFIG` explicitly overrides it.

A fresh run resets its own named result directory. Choose a new `run_name` to retain earlier results. Input and prepared-data directories must remain outside the directory being replaced. File presence is checked before a reset; detailed schema and scope checks follow during preparation.

## Randomness and numerical settings

The base seed is `20261101`. Stochastic stages and mechanism/realization streams derive their seeds from the configuration. Each run records Python, platform, and package versions in `models/environment.json`. Reproduction also depends on that numerical environment.

| Quantity | Empirical default |
|---|---:|
| Network grant years | 1980–2025 |
| Requested observation end | 2025-12-31 |
| Community baseline; resolution | 1980–1999; 1.3 |
| Maximum target age | 30 years |
| Opportunity periods | 2000–2018; 2019–2022; 2023–2025 |
| Event cap per period/universe | 4,000 |
| Nonchosen controls per event | 32 |
| Source bootstrap replicates | 199 |
| Growth window | 2023 through the effective 2025 end |
| Growth realizations per mechanism/domain | 30 |
| Alpha; age scale | 1; 10 years |
| Active semantic/community costs | 4; 1 |
| Recorded null states | 199 |
| Null burn-in / spacing | 20E / 10E proposals |
| Site/random-insertion realizations | 20 |
| Connectivity grid points | 41 |
| Diffusion scales | 65, logarithmic from 0.01 to 100 |
| Fine / coarse SLQ | 32 probes × 96 steps / 16 × 48 |
| Figure DPI | 600 |

The synthetic example has deterministic generated inputs, reduced budgets, and 300-dpi PNGs. All ten plates carry a synthetic-data watermark.

## Populations

Full graphs retain isolates and unknown attributes. Opportunity correction uses the full or known-attribute historical universe according to the statistic. Growth uses the complete common known-attribute new-edge population; the 4,000-event cap only applies to opportunity sampling.

Growth preserves source reference counts and strict chronology. Sources on the same day share the pre-date degree state. The fixed backbone contains historical and contemporaneous unmodeled edges; observations and simulations subtract that same backbone for incremental transitivity.

The constrained chain preserves directed degrees and local citation-year profiles, counting rejections in its clock. Ensemble variation, source bootstrap variation, growth-realization variation, ordering variation, and probe variability are distinct uncertainty summaries.

## Output locations

| Location under `outputs/<run_name>/` | Content |
|---|---|
| `data/` | Shared prepared records, vectors, observation coverage |
| `models/` | Effective configuration and software environment |
| `tables/` | Shared population and topology summaries |
| `domains/<domain>/data/` | Domain graph, vectors, opportunities, layout, communities |
| `domains/<domain>/tables/` | Structural, opportunity, diffusion, connectivity, growth, null results |
| `domains/<domain>/models/` | Algorithm summaries and simulated examples |
| `figures/` | Five main and five supplemental plates in PDF/SVG/PNG |
| `figures/captions/` | Standalone captions |
| `figures/figure_index.csv` | Figure file index |

Configure empirical inputs through `source_root` in the selected configuration. The synthetic workflow uses its own generated inputs and result directory.
