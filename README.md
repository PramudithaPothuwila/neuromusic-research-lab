# NeuroMusic Research Lab

**A Data-Driven Investigation of Music-Evoked EEG Patterns Using Spectral Analysis,
Unsupervised Learning, and Cross-Participant Generalization.**

NeuroMusic Research Lab is a local, reproducible research application for analysing
electroencephalography (EEG) recorded while participants listen to songs. It takes raw headset
CSV exports and turns them into validated recordings, cleaned signals, documented EEG
features, statistical comparisons between songs, exploratory clusters, subject-independent
machine-learning results and exportable research reports, all through either a command line
or a Streamlit dashboard.

The software puts scientific validity ahead of impressive-looking results. It never modifies
raw data, never invents missing metadata such as units, reference, timing or emotion labels,
and reports when an analysis cannot be performed.

---

## 1. Research questions

| ID | Question | How the platform addresses it |
|---|---|---|
| RQ1 | Do EEG spectral characteristics differ across songs? | Participant-level tests (within- or between-participant), mixed-effects models, FDR correction |
| RQ2 | Which channels, regions and bands differ most? | Per-channel, per-region and asymmetry features; effect sizes; primary vs. exploratory feature families |
| RQ3 | Does unsupervised learning reveal structure? | PCA, K-Means and hierarchical clustering with silhouette, stability, and song-vs-participant agreement |
| RQ4 | Can a model identify the song for an unseen participant? | Leave-One-Subject-Out classification against a dummy baseline, with a permutation test |
| RQ5 | Do participants show synchronized EEG during the same song? | Inter-subject correlation from song onset (recordings start at the beginning of the song) |
| RQ6 (optional) | Do EEG features predict self-reported emotion? | Rating importer and LOSO regression, enabled only when participant ratings exist |

EEG band power is not treated as a direct measure of happiness, sadness or any other emotion.

---

## 2. Data

### EEG recordings

Each CSV holds one participant listening to one song:

```
sample_index,time_s,EEG.AF3,EEG.F7,EEG.F3,EEG.FC5,EEG.T7,EEG.P7,EEG.O1,EEG.O2,EEG.P8,EEG.T8,EEG.FC6,EEG.F4,EEG.F8,EEG.AF4
```

* 14 channels at 256 Hz (the sampling rate is detected from the timestamps, not assumed).
* Values carry a large DC offset. Units, scaling, the electrode reference and the device are
  **unconfirmed**, so amplitudes are reported in arbitrary units (a.u.).
* Filenames such as `SRC006_SOURCE_P004_Nadagam_Geeya.csv` are parsed into source,
  participant and song. Song names are normalised, so `Hadisida_Manda` and `Hadisida Manda`
  are recognised as the same song.

### Provenance audit and current dataset status

Filenames do not guarantee independent recordings. Every file is fingerprinted, and files
whose signals are near-identical are grouped into "signal families". The audit of the 87
supplied files found:

| Status | Files | Meaning |
|---|---|---|
| original | 31 | independent recordings (one is only 3.9 s long and fails validation) |
| derived copy | 28 | the same signal as an earlier file, relabelled as another participant |
| label conflict | 11 + 2 byte-identical | one signal carrying different song labels; the true song is unknown |
| synthetic | 15 | `SYNTH*` files derived from real recordings |

**30 independent analysable recordings from 22 participants across 16 songs remain.** Most
participants heard only one song. `data/metadata/recording_manifest.csv` lists every file
with its identified song, provenance and exclusion reason. Excluded copies are not
independent participants, and their origin should be confirmed with whoever collected the
data.

### Song metadata and audio

* `data/metadata/song_attributes.csv`: per-song valence, arousal, danceability and tempo
  proxies, plus survey means for 20 songs. These describe songs, not individual participants,
  and are used only for exploratory song-level comparisons.
* `data/metadata/song_sources.json`: reference video for each song. The owner confirmed that
  the supplied audio files are the exact files played during recording.
* Every recording starts at the beginning of its song (owner statement), but recordings are
  shorter than the songs: about 2 minutes, sometimes 21–94 s, of 2.6–7.3 minute songs.
* `data/stimuli/incoming/`: drop licensed song audio here, named by song title, then run
  `neuromusic stimuli scan`. Imported audio is stored with a provenance record covering
  origin, licence statement and checksum.

---

## 3. How the analysis works

```
raw CSV ─► validation ─► provenance audit ─► preprocessing ─► windows ─► features
                                                                            │
                     report ◄── experiments (statistics, clustering, LOSO ML, ISC) ◄─┘
```

1. **Validation**: required columns, numeric values, duplicated or non-monotonic samples,
   timestamp gaps, sampling rate, flat channels and extreme excursions. Problems are
   flagged, never silently fixed.
2. **Preprocessing** (MNE-Python / SciPy): DC removal, optional notch, zero-phase 1–40 Hz
   band-pass, configurable re-referencing (the original reference is kept by default) and
   optional ICA.
3. **Artifact detection**: flatlines, sudden steps, amplitude bursts, possible movement,
   muscle-like high-frequency power, drift, outlying channels and line noise. All thresholds
   are relative (robust z-scores) because units are unknown. Every flag records where it is
   and why.
4. **Segmentation**: 4 s windows with 2 s overlap, each labelled good, flagged or rejected.
5. **Features**: Welch PSD band power (theta, alpha, beta) in absolute, log and relative form;
   alpha/beta and theta/beta ratios; spectral entropy; amplitude statistics; Hjorth
   parameters; regional power; and frontal asymmetry. Window features are aggregated to one
   row per participant-song trial, because hundreds of windows from one trial are not
   independent observations.
6. **Statistics**: participants are the unit of analysis. The platform uses within-participant
   tests when participants heard several songs and otherwise falls back to a clearly labelled
   between-participant design. Results include mixed-effects models, effect sizes, confidence
   intervals, permutation p-values and multiple-comparison correction.
7. **Unsupervised learning** (exploratory): PCA and clustering, with diagnostics showing
   whether structure follows songs or individual participants.
8. **Machine learning**: dummy, logistic regression, SVM, random forest, plus optional MLP
   (backpropagation) and XGBoost. All models are evaluated with Leave-One-Subject-Out
   validation, and scaling and tuning are fitted on training participants only. Outputs are
   held-out metrics, confusion matrices, per-participant accuracy, permutation importance and
   a within-participant label-permutation test.
9. **Reporting**: Markdown and HTML reports with CSV tables, PNG figures, JSON results, a
   limitations section and full reproducibility details.

Every run is stored as an experiment, and none is ever overwritten. Each experiment records
the dataset fingerprint, configuration, seed, cross-validation scheme, software versions and
metrics.

---

## 4. Installation

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically).

```bash
uv sync                      # core dependencies from uv.lock
uv sync --extra xgboost      # optional XGBoost model
uv sync --extra stimuli      # optional audio acquisition tool
```

Everything runs locally; nothing is uploaded.

## 5. Usage

### Dashboard

```bash
uv run neuromusic dashboard          # http://localhost:8501
```

The dashboard has twelve pages: Overview, Dataset Manager, EEG Explorer, Signal Quality,
Preprocessing, Spectral Analysis, Song Comparison, Clustering, Model Training, Model
Evaluation, Neural Synchronization and Reports.

### Command line

```bash
uv run neuromusic scan            # register data/raw/eeg, run provenance audit, write manifest
uv run neuromusic features        # preprocess and extract features
uv run neuromusic stats           # descriptive and inferential statistics
uv run neuromusic cluster         # exploratory PCA / clustering
uv run neuromusic classify --permutations 200   # LOSO song classification
uv run neuromusic report          # reports/<id>/report.html (+ .md, tables, figures)
uv run neuromusic stimuli scan    # import song audio from data/stimuli/incoming
uv run neuromusic experiments     # list past experiments
uv run neuromusic demo --root demo_workspace    # synthetic demo, labelled as such
```

Add new recordings by copying them into `data/raw/eeg/` and running `neuromusic scan`. No code
changes are needed.

### Configuration

All analysis settings live in `configs/` (`dataset.yaml`, `preprocessing.yaml`,
`features.yaml`, `statistics.yaml`, `experiments.yaml`). Changing a setting creates a new,
separately versioned set of outputs instead of overwriting old ones. Once units, reference
or device are confirmed, record them there. Playback onsets are recorded per recording with
`neuromusic align --onset <s> --evidence "<source>"`.

---

## 6. Repository layout

```
configs/              YAML configuration
data/raw/eeg/         original EEG files (read-only, not committed)
data/metadata/        registry, manifest, song attributes, song sources
data/stimuli/         song audio (incoming/ → audio/), not committed
data/interim/         cleaned signals per preprocessing configuration
data/processed/       feature tables (Parquet)
experiments/          one directory per experiment run
reports/              generated reports
src/neuromusic/       scientific package (ingestion, preprocessing, features, analysis,
                      clustering, modeling, experiments, visualization, reporting, cli)
dashboard/            Streamlit presentation layer
tests/                unit and integration tests
docs/                 architecture, dataset, methodology, validation, limitations
```

## 7. Quality and testing

```bash
uv run pytest                          # unit and integration tests
uv run ruff check src tests dashboard
uv run mypy
```

The tests use synthetic signals with known properties, for example a 10 Hz sine must
dominate the alpha band and a filter must remove 60 Hz. They also check that no model is ever
trained on the participant it is tested on.

## 8. Limitations

* Only 22 independent participants, and most heard a single song, so song and participant
  effects are largely confounded.
* With 16 songs and 1–5 recordings each, song classification has very little data per class.
* Units, reference and device are unverified; recordings cover only the opening of each song.
* Song attributes are song-level aggregates, not participants' own emotional ratings.
* Successful classification of these songs would not show generalisation to new songs.

See `docs/research_limitations.md` for the full list, and `docs/methodology.md` for exact
definitions.

## 9. Privacy

Participant identifiers are pseudonymous (`P001`…). Raw EEG, audio, registries and results are
excluded from Git. ZIP imports are checked for path traversal and unsupported files, and no
dataset content is ever executed.
