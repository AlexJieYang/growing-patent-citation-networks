# Patent network growth

[中文说明](README.zh-CN.md)

Research code for **Transitive closure and target reuse in growing patent citation networks**. Six Python notebooks and a command-line entry point cover data preparation, network structure, historical attachment opportunities, diffusion, connectivity, chronological growth, constrained null ensembles, and publication figures.

The implementation uses pandas, NumPy, SciPy, NetworkX, and Matplotlib. The workflow supports empirical patent networks and a generated two-domain example.

## Installation

Use Python 3.11 or later. Create a virtual environment in this directory:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -r requirements.txt
```

## Synthetic example

```bash
python run_all.py --demo
python -m pytest -q
```

Results appear in `outputs/demo_physics/`. The example produces all ten figure plates in PDF, SVG, and PNG, marked **SYNTHETIC DATA**. It uses smaller simulation budgets and 300-dpi PNGs; the empirical configuration retains 30 growth realizations, 199 null states, and 600 dpi.

## Empirical analysis

The workflow uses processed USPTO/PatentsView tables and an indexed embedding matrix. Data types and alignment conventions are described in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

Provide the data directory directly, without editing the tracked configuration:

```bash
python run_all.py --source-root /path/to/PatentsView
```

On Windows, for example:

```powershell
python run_all.py --source-root "C:\data\PatentsView"
```

Alternatively, copy the public configuration to an ignored local file:

```bash
python -c "from shutil import copyfile; copyfile('config.json', 'config.local.json')"
```

Edit `source_root` in `config.local.json`, then run:

```bash
python run_all.py --config config.local.json
```

To retain several analyses, use a different result name:

```bash
python run_all.py --config config.local.json --run-name comparison_2025
```

Each fresh run rebuilds its named directory under `outputs/`. Keep inputs outside that result directory. Effective settings are saved under `outputs/` and do not overwrite `config.json`.

## Notebooks

```bash
jupyter lab
```

Run **00 → 01 → 02 → 03 → 04 → 05** from this directory. Notebook 00 uses `config.local.json` when present, otherwise `config.json`; optional `SOURCE_ROOT` and `RUN_NAME` variables override those settings. Later notebooks read the active configuration saved by 00. Restart at 00 when switching synthetic and empirical inputs. An explicitly set `PRE_CONFIG` environment variable takes priority in later notebooks.

| Notebook | Contents |
|---|---|
| `00_Data_Preparation.ipynb` | CPC selection, chunked citation ingestion, vector alignment, observation coverage |
| `01_Network_Structure.ipynb` | Degree, arrivals, components, clustering, coreness, transitive closure |
| `02_Attachment_Opportunities.ipynb` | Historical candidate sets, opportunity enrichment, source bootstrap |
| `03_Diffusion_and_Connectivity.ipynb` | Heat return, spectral entropy, node removal, edge insertion |
| `04_Growth_and_Null_Models.ipynb` | Three growth mechanisms and the temporal degree-preserving ensemble |
| `05_Publication_Figures.ipynb` | Five main-text and five supplemental figure plates |

## Research design

The default selects neural-network CPC families and H01S3 laser patents granted in 1980–2025. The requested observation end is 2025-12-31; source coverage and the available metadata bound the actual end, recorded in `observation_window.json`.

- Frozen communities use references originating before 2000.
- Opportunity periods are 2000–2018, 2019–2022, and 2023–2025, with up to 4,000 events per period/universe and 32 nonchosen controls per event.
- Growth uses 2023–2025, 30 realizations per mechanism, and fixed modeled reference counts per source.
- The mechanisms successively include degree/age, semantic cost, and community cost. Parameters are `alpha = 1`, `age_tau = 10 years`, and active costs `beta = 4`, `kappa = 1`.
- The structural ensemble records 199 states, preserving directed degrees and local citation-year margins. Rejections count in the proposal clock.

See [METHODS.md](METHODS.md) for estimator definitions and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for numerical settings and output locations.

The paper uses 768-dimensional PatentSBERTa embeddings of concatenated titles and abstracts. The workflow reads precomputed vectors and aligns them by patent identifier; it does not train the encoder. See [Bekamiri, Hain, and Jurowetzki (2024)](https://doi.org/10.1016/j.techfore.2024.123536). Patent records originate from the [USPTO through PatentsView](https://www.uspto.gov/ip-policy/economic-research/patentsview).

## Outputs and budgets

Shared summaries and figure plates appear under `outputs/<run_name>/`. Domain data, tables, and models appear under `outputs/<run_name>/domains/neural/` and `domains/laser/`. Captions are separate files in `figures/captions/`; plots retain panel letters, axes, and legends.

Default budgets are 30,000 selected nodes per domain, 600,000 nodes including citation boundaries, 2,000,000 retained edges, 3 GiB of selected vectors, and 100,000,000 cached growth candidate pairs. Citation tables are read in 200,000-row batches; vectors use memory mapping and numerical threads default to four. These component budgets do not bound total process memory.

## Source checks and tests

`python -m pytest -q` runs unit tests immediately. Data-dependent tests activate after `python run_all.py --demo`, checking chronology, vector alignment, opportunity histories, growth backbones, ensemble invariants, and figure formats. These research tests are run explicitly in a configured local environment. GitHub Actions runs only static source checks: Python syntax, JSON, notebook structure and clean execution state.

Run the same static checks locally with `python tools/check_sources.py`.

Clean notebook execution artifacts before committing:

```bash
python tools/clean_notebooks.py
python tools/clean_notebooks.py --check
```

`.gitignore` excludes raw inputs, local configuration, generated outputs, and caches. Distributed notebooks have no saved outputs or execution counts. Author and software citation metadata is in [CITATION.cff](CITATION.cff).

## License

The research code is available under the [MIT License](LICENSE). External patent records, embeddings and other third-party materials retain their original licenses and access conditions.
