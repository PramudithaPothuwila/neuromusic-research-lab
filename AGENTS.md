# Agent instructions

* Python 3.12, `uv`. Run `uv run pytest`, `uv run ruff check src tests dashboard`, `uv run mypy`
  after changes.
* Scientific code lives in `src/neuromusic`; `dashboard/` is presentation only and must call
  `neuromusic.services` / `ExperimentRunner` rather than computing.
* Never modify files under `data/raw/`. Record exclusions and metadata corrections in the
  registry (`DatasetRegistry.update_recording`), with reasons.
* Do not assume signal units, reference, device, playback alignment or emotion labels. Keep
  them configurable and documented.
* Any learned transform must be fitted inside the LOSO training fold (use sklearn `Pipeline`).
* Windows are not independent observations; inference uses participants or trials.
* Synthetic data must stay labelled synthetic in every output.
* Do not overwrite experiments or reports; new configs produce new hashed output sets.
* Never commit participant data, audio, or credentials.
