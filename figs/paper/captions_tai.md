# IEEE TAI figures (scripts/05_paper_figures.py)

**fig1_data.** (a) Fish oil sample 1___SNA1_Liver_Male_A (sep2015), 10–30 min, m/z 40–400: TIC (top), log intensity (bottom), spectrum at the tallest apex (14.40 min; largest ions m/z 55, 74, 87). (b) First-sample TICs of batches 0, 3, 6, 9 of test world 6 (K = 10, world seed 140006); triangles: one compound (latent 21.83 min) in each batch; red: a corrupted anchor of batch pair 3–6 in the K = 10, complete, budget 30, ε = 30% cell, moved -19.8 bins. (c) True drifts span -4.42 to +3.91 min.

**fig2_graphs.** The six topologies for K = 10 (test world 6; node colours as in fig. 1c). Edges: complete 45, density 0.4 18, ring 10, chain 9, star 9, weak-star 9. Star hub: batch 0 (5 samples); weak-star hub: batch 5, the batch with the fewest samples (3 samples). Density 0.4 as drawn in the K = 10, budget 30, ε = 15% cell.

**fig3_synthetic_results.** (a) World-level median map error over the 20 test worlds (boxes: quartiles, whiskers: range). Stars are scored on the pairs they reach. (b) Paired per-world difference; large markers: medians; lines: least-squares fit on log2 K with 95 % bands from resampling test worlds within each K (script 02). Density 0.4 slope -0.0084 min per doubling [-0.0107, -0.0063].

**fig4_breakdown_diagnosis.** (a) Test world 3 of K = 5, chain, budget 30, ε = 30% (the broken world closest to the setting median 1.525 min; 16/20 worlds broken), refitted on script 02's anchors. Edge 1–2 is the chain edge with the largest joint error. World median map error: Joint (CV) 1.538, Joint (λ = 10) 0.116, Pairwise 0.073 min. (b) Broken test worlds (> 0.3 min) over script 03's corrupted settings (4 topologies × K ∈ {5, 10} × 3 budgets × 2 ε): Cross-validated λ 175/960; Fixed λ = 10 160/960; Per-world oracle λ 80/960. The oracle uses the truth: a bound, not a rule.

**fig5_fish_results.** Fish oil data (K = 5, complete graph): held-out spectral-anchor error, median and interquartile range over 20 thinning/corruption draws per condition.
