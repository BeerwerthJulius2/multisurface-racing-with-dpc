# Minimal paper changes from the exploratory results

Target: ECC, with the hardware experiments. Keep the contribution as it is; adjust the
claims so they match what section 1-6 of `RESULTS.md` actually shows. Line numbers refer
to `paper/root.tex` as of commit 9282449.

## Must change (the current wording overclaims)

1. **Abstract, "implied friction that follows the spatially varying ground truth".**
   At constant friction the implied friction still swings by about +-0.25 along the lap,
   so part of that pattern is geometry. Replace with something like: *the prediction is
   shaped predominantly by friction-consistent data where the maneuver loads the tires,
   and mixes freely where it does not.*

2. **Contribution 1 (l. 48).** "predominantly assigns weight to trajectories consistent
   with the current friction regime" -> state it as conditional: predominantly *in
   maneuvers that approach the friction limit*, and note that the low-friction regime is
   the identifiable one.

3. **Section IV-C, l. 358** ("The implied friction follows the spatial variation of the
   track friction closely"). Soften to the measured version: about half the friction
   difference is recovered in corners, essentially nothing on straights.

4. **Section III, "Two regimes", l. 258** ("if the subsets are locally indistinguishable
   in the past window, their future predictions are similar as well"). False in general:
   a straight-line window is compatible with both regimes even when the next 2 s differ.
   Condition it on the upcoming maneuver staying in the friction-insensitive regime.

## Should add (cheap, each ~2-4 sentences)

5. **Conditions paragraph (l. 267ff): the observability criterion.** Friction is only
   identifiable near the limit. The dataset reaches ~90% of the mu=0.5 limit but only
   ~41% of the mu=1.1 limit, which is why the low-friction regime is recognized and the
   high-friction one is not. This explains the asymmetry instead of apologizing for it.

6. **Conditions paragraph: paired coverage.** Discrimination needs comparable trajectories
   under both regimes at the current operating point (+22% with them, +4% without).

7. **Section III or IV: the lambda_sigma / T_ini trends.** Both behave as the argument
   predicts (T_ini 3/5/10 -> 11/22/45%; lambda_sigma 1/10/101 -> 9/22/61%). This turns
   the Conditions paragraph from intuition into a measured statement. Add that the
   sharpening does not transfer to closed loop (lambda_sigma=101 diverges).

8. **Related Work: the LPV line.** Position against the LPV fundamental lemma and
   LPV-DeePC (Verhoek, Toth, Haesaert et al. - verify the exact references): they assume
   a measured scheduling signal, we have none. Also kernel/Koopman DeePC (Lian & Jones)
   and the regularization-as-robustness reading (Coulson/Dorfler/Markovsky). Currently
   Section III cites none of this and an ECC reviewer will notice.

9. **One paragraph of negative results** (pre-empts three obvious reviewer suggestions):
   group sparsity 48%, persistence prior 22%, explicit hard selection 0% and diverging,
   all against the 51% baseline; extra output channels (yaw rate, v*yaw_rate) reduce
   distinguishability. The implicit weighting already extracts what the data supports.

## Space

ECC is 6 pages and the hardware results have priority.

- Compress the uniform-friction experiment to one paragraph plus a two-line table
  (straights vs corners). The hardware runs are themselves uniform-friction tests.
- The Nuerburgring subsection (l. 361-383, two figures) is the main compression
  candidate: it is a scope-and-limitation result, and section 6 of `RESULTS.md` shows
  that track cannot test the mechanism anyway.

## Figures

- Keep Fig. 5 (implied friction on the multi-friction track) but caption it with the
  caveat that part of the spatial variation is geometry.
- Add, if space allows: implied friction vs lateral acceleration for the two uniform
  runs - one panel, and the single most convincing piece of evidence.

## Not for this paper

- lambda_g scaling with dataset size (the Select-DPC retuning question).
- Distance-weighted regularization over a larger candidate pool.
- The offline/closed-loop asymmetry discrepancy (section 5 vs 1 of `RESULTS.md`) is
  unresolved; keep it out of the claims rather than explaining it away.
