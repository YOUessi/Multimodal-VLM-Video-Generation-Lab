# Batch 59: prospective class-disjoint natural-video replication and scene-group risk screen

## Research bottleneck

The historical DEV-only cross-group audits showed that a tracker satisfying
mean occlusion safety can fail on held-out scene groups. Batch 56 added 36
natural-video DEV scenes, yet scene-level risk support remains weak, and a
low mean false-accept rate does **not** imply safety across unseen domains.

This protocol is **prospectively specified before the Batch 59 media or
private query annotations are opened**. Prior TEST datasets are already
opened and must never be used to choose or revise this protocol.

## Source, population and independence firewall

- GitHub main is the source of truth. Each run pins one clean commit SHA.
- 42 new Kinetics-700/TAP-Vid-Kinetics video IDs **and distinct action
  classes** with 36 DEV and six sealed TEST videos, four official point
  tracking queries per video.
- Exclude the six historical 12-video cohorts (Batch 42, 45E, 46E, 48B,
  53B, 55) and the 42-video Batch 56 cohort: **114 previous IDs/classes**.
- Exclusions may read only historical public video manifests; never load
  historical query oracles, predictions or TEST result files.
- Freeze data selection, image sequences, oracle hashes, implementation and
  method selection rules before TEST.
- Nine unchanged CPU tracking methods, identical original primary DEV
  selection (coverage >=30% and scene-macro occlusion false acceptance <=10%,
  maximize Jaccard among admitted candidates). Fallback remains explicitly
  ineligible when no candidate meets both.
- SmolVLM2-2.2B three-frame baseline and its existing DEV-only admission
  thresholds remain identical. Do not run TEST VLM inference if blocked.
- CPU latency excludes video decode and common affine preprocessing; report
  both primary screen-motion and background-relative *proxy* diagnostics.
  An affine proxy is not physical camera velocity.

## New secondary DEV-only diagnostic (R07B)

Each of the 36 DEV scenes belongs to exactly one of six fixed consecutive
six-scene groups: indices 0–5, 6–11, 12–17, 18–23, 24–29, 30–35.
These boundaries are fixed before Batch 59 DEV observations.

For each of the nine candidates:

1. Calculate scene-macro occluded false-accept mean in each group, ignoring
   scenes with zero occlusion support. A group with zero support is unknown,
   not evidence of a successful safety check.
2. Record group mean exceedance of the original 10% error limit and whole
   DEV visible coverage against the unchanged 30% minimum.
3. An exploratory counterfactual group-screen candidate requires all six
   groups to have occlusion support, no group mean above 10% and global
   coverage >=30%. Choose maximal DEV Jaccard if more than one qualifies.
4. Separately record the exact one-sided 95% familywise Clopper–Pearson
   upper bound on the fraction of future **scenes** exceeding a 10% error
   budget; the method family size is nine. This is not the frame-level
   false-accept probability.

**R07B cannot change the frozen method selected by R07, cannot change
the VLM gate, cannot read TEST, and cannot authorize deployment.**
The report must recompute the R07B result from pinned DEV files before it
can pass. A counterfactual method is never promoted using already opened
TEST scores.

The six groups are arbitrary scene-index blocks, not randomized domains.
DEV training and this audit share observations; scene exchangeability is
unproved. Even favorable results are exploratory, not a safety certificate.

## Execution and reproducibility contract

The 24 stages execute one frozen plan with exactly one project cron owner,
fail-closed one-attempt stages, GPU lock, pinned TEST prediction before
independent TEST scoring, post-evaluation regression, public/private data
hash audit and immutable final report. All prior workspaces and failed logs
are retained. No video, private oracle, generated frames, model weights,
or secrets are permitted in GitHub commits.

Disk guard: refuse new shard downloads if less than 10 GiB free at R00.
A full download may need additional capacity; do not delete unrelated
files or run an additional GPU experiment.

## Hypotheses and negative outcomes

- H1: the secondary group screen may reveal safety risk missed by the
  frozen global DEV mean. It is **not** expected to improve the tracker
  without a distinct independent prospective intervention.
- H2: independent TEST can still have a severe worst-scene occlusion
  error, even if DEV global and group means pass.
- Negative outcome: no safe group candidate, weak coverage, unsupported
  occlusion scenes, blocked VLM DEV gate, insufficient disk or download
  failure. Preserve these outcomes rather than lowering thresholds.
- Do not claim success based on aggregate averages when some video scenes
  violate the 10% budget.

The final report records DEV and independent TEST coverage, precision,
PCK8, Jaccard, scene-macro invisible false-accept and CPU p50/p95,
plus the new group diagnostic and provenance pins. No deployment
authorization follows automatically.
