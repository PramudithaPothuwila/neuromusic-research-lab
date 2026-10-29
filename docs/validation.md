# Validation

## Automated tests (`uv run pytest`)

| Area | Tests |
|---|---|
| Import / validation | valid CSV, missing channel, duplicate sample indices, timestamp gaps, wrong sampling rate, non-numeric values, missing required column, flatline, unparseable filename (manual correction), re-import detection, unsupported type, invalid ZIP, ZIP path traversal, ZIP import |
| Provenance | near-duplicate relabelled copy, byte-identical copy with conflicting label, manual inclusion survives re-audit |
| Signal processing | band power of known sinusoids, no double counting at band edges, overlapping bands rejected, band-pass attenuation, window length/overlap, artifact flags for synthetic flatline/step/movement, clean noise unflagged, window quality, finite features, Hjorth of a sine |
| Statistics | paired effect + CI, Friedman + post-hoc, null feature, insufficient participants, between design, mixed model |
| Modelling | LOSO never fits on the held-out participant (instrumented transformer), separable data beats dummy, within-participant permutation preserves structure, participant minimum, window-level majority vote |
| Metadata | song attributes, stimulus aliases, rights confirmation, audio provenance sidecar, rating validation |
| Integration | synthetic workspace: import → audit → features → statistics → classification → clustering → report; synthetic labelling; no experiment overwrite; synchronization unavailable without alignment and computed with it |

## Static checks

`uv run ruff check src tests dashboard` and `uv run mypy`.

## Scientific checks to perform on real data

* Inspect `recording_manifest.csv` with the data owner before any analysis is reported.
* Review per-channel quality and flagged windows on the Signal Quality page.
* Compare learned models against the dummy baseline and the permutation null; report the
  participant-level accuracy distribution, not only pooled accuracy.
