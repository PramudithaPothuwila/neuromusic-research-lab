# Methodology

## Preprocessing (`configs/preprocessing.yaml`)

1. Read raw CSV (read-only); select configured channels in configured order.
2. Sampling rate: `auto` uses the timestamp-derived rate; a configured rate that conflicts with
   timestamps is an error (no silent resampling).
3. Analysis span: `trim_start_seconds`, `trim_end_seconds`, and the recording's
   `playback_onset_s` if set.
4. Median DC removal → optional notch (50 Hz + harmonics) → zero-phase FIR band-pass
   (1–40 Hz default, MNE `filter_data`) → re-reference (`preserve_original` default;
   `average` or `channels`) → optional ICA (only with explicitly listed components).
5. Artifact detection on 1 s segments, all relative: flatline (segment SD < 1 % of channel
   median SD), steps (sample-difference robust z > 15), excessive amplitude (peak-to-peak
   robust z > 8), high-frequency contamination (45–100 Hz power fraction robust z > 5), possible
   movement (amplitude flags on ≥ half the channels), recording-level drift (> 90 % of raw power
   < 0.5 Hz), outlying channel variance (robust z across channels > 5), possible poor contact
   (> 30 % of power at line frequency). Each flag stores channel, start/end, value, threshold
   and reason.
6. Windows (4 s, 2 s overlap). Status: `good`, `flagged`, or `rejected` (only when automatic
   rejection is enabled and ≥ 3 channels are flagged or a global flag overlaps).
7. Outputs: `data/interim/cleaned/<preprocessing_hash>/` (signals `.npz`, windows/flags/quality
   Parquet, `ProcessingRun` JSON). The hash covers the preprocessing and acquisition configs.

## Features (`configs/features.yaml`)

Welch PSD (2 s Hann segments, 50 % overlap, median averaging). Band power = rectangle-rule
integral over half-open bands (theta [4, 8), alpha [8, 13), beta [13, 30) Hz); overlapping band
definitions are rejected. Per channel: absolute, log10, relative (to 1–40 Hz) power;
alpha/beta and theta/beta ratios; normalised spectral entropy; mean, SD, variance, RMS,
peak-to-peak, MAD; Hjorth activity, mobility, complexity. Regions (configurable, recorded
channels only): mean band power, log and relative power; asymmetry ln(right) − ln(left) for
F4/F3, AF4/AF3, F8/F7, FC6/FC5.

Trial features: median over valid stimulus windows, plus IQR for region features. Trials with
< 5 valid windows are marked invalid. Stored as Parquet under
`data/processed/features/<prep_hash>_<feature_hash>/` with a manifest.

## Statistics (`configs/statistics.yaml`)

Unit of analysis: participant (trials averaged within participant-song).

* `within` design (participants heard several songs): paired Wilcoxon + paired t, rank-biserial
  r, Cohen's d_z, bootstrap CI, sign-flip permutation (2 songs); Friedman + Kendall's W,
  within-participant permutation, Holm-corrected pairwise Wilcoxon (≥ 3 songs). Complete cases.
* `between` design (fallback under `auto` when no participant heard every song): Kruskal–Wallis
  across songs with ≥ 2 participants, epsilon-squared, label-permutation p-value. Participant and
  song are confounded in this design.
* Mixed models: `feature ~ song + (1 | participant)` by ML, likelihood-ratio test vs. intercept
  model, convergence and singular-fit reporting.
* Multiple comparisons: Benjamini–Hochberg (default), separately for the pre-specified primary
  family (`region.log_abs_power.`, `region.rel_power.`, `asymmetry.`) and exploratory features.
* Song-attribute associations: Spearman correlation of song-mean features with song attributes;
  n = number of songs; exploratory only.

## Unsupervised learning

Median imputation + robust scaling → PCA; K-Means over a k range (not tied to the number of
songs) with silhouette, cluster sizes, bootstrap ARI stability, ARI vs. song and vs. participant;
optional Ward clustering. Labelled exploratory; clusters are not emotional states.

## Supervised learning

Observations: trials (default) or valid windows. Models: dummy (most frequent), multinomial
logistic regression, SVM, random forest; optional MLP (backpropagation) and XGBoost.
Leave-One-Subject-Out: per fold, imputation, scaling and grid search (inner GroupKFold over
training participants) are fitted on training participants only. Window-level runs also report
trial-level majority votes. Metrics: accuracy, balanced accuracy (participant-bootstrap CI),
macro-F1, per-class precision/recall/support, confusion matrix, per-participant accuracy, mean
training accuracy (reference only). Permutation importance is computed on held-out participants.
Optional permutation test: labels permuted among each participant's trials (windows of a trial
move together), full pipeline re-run per permutation.

Reinforcement learning is not used: there is no sequential decision process or reward signal
in this design. The backpropagation-trained MLP is available but should not be expected to beat
linear baselines at this sample size.

## Synchronization

Enabled only when ≥ 2 participants' recordings of a song have a verified `playback_onset_s`.
Per-channel mean pairwise Pearson correlation of cleaned signals over the common aligned
duration; sliding-window ISC; circular-shift surrogates (min shift 5 s) for p-values with FDR
across channels.
