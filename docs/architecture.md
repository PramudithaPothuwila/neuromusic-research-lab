# Architecture

```
configs/*.yaml ──> neuromusic.config (pydantic models, config_hash)
                        │
neuromusic.workspace ───┤ paths, configs, DatasetRegistry, unit labels
                        │
ingestion ──> preprocessing ──> features ──> analysis / clustering / modeling
   │              │                 │                    │
registry.json   data/interim     data/processed     experiments/<id>/
manifest.csv    (per prep hash)  (prep+feat hash)   record.json + tables
                                                         │
                                       reporting ──> reports/<id>/ (md, html, csv, png, json)

services.py        data-pipeline orchestration (scan/audit, preprocess, features)
experiments/runner ExperimentRunner: statistics, classification, regression, clustering, ISC
cli.py             argparse front end
dashboard/         Streamlit presentation layer (calls services/runner only)
```

Principles:

* Scientific code has no Streamlit dependency; the dashboard and CLI call the same services.
* Every derived artifact is keyed by configuration hashes; changing a config creates a new set
  rather than overwriting.
* Raw data is read-only. Exclusions and metadata corrections live in the registry with reasons
  and sources (`provenance_audit` vs `manual`).
* Experiments are created once (`ExperimentStore.create` refuses existing ids) and record the
  dataset fingerprint, configs, seeds, CV strategy, environment and metrics.
* Fold isolation is structural: every model is a scikit-learn `Pipeline` fitted inside
  `LeaveOneGroupOut`, and tuning uses `GroupKFold` over training participants only.

Module map: `domain/` entities and enums; `ingestion/` CSV reader, validator, filename parser,
registry, provenance audit, ratings, song attributes, stimulus catalogue; `preprocessing/`
filtering, referencing, artifacts, segmentation, quality, pipeline; `features/` spectral,
temporal, complexity (Hjorth), asymmetry/regions, extractor, aggregation, store; `analysis/`
descriptive, comparisons, mixed effects, correlations, synchronization; `clustering/`;
`modeling/` datasets, baselines, classifiers, regressors, validation, metrics, importance;
`experiments/`; `visualization/` (Plotly); `reporting/` (Matplotlib + Jinja2).
