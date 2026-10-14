# Paper figures (scripts/20_paper_figures.py)

Numbers are computed from the world or from the results files named in each caption.

**fig1_data.** One draft-run world (10 batches, world seed 5000). (a) Normalised TIC of the first sample of batches 0, 3, 6, 9, each on its own retention-time axis; triangles mark three compounds (latent 12.50, 21.28, 32.26 min). (b) True drift of every batch, from -3.60 to 3.24 min.

**fig2_graphs.** Anchor-graph structures of the same world, exactly as script 11 drew them; edges per structure: star 9, chain 9, chain+2 11, ring 10, density 1.0 45, density 0.4 18, density 0.2 9.

**fig_regime.** Paired median over worlds of (joint − competitor) median map error (min), with the worlds the joint model wins; symmetric-log colour, linear within ±0.01 min. Largest advantage over composed pairwise: -1.113 min (chain, ε 30%, 8/10 worlds).

**fig_breakdown.** Per-world median map error under ε = 30 %, same anchors refitted; shaded: joint above 0.3 min. ring: 6/10 worlds broken, oracle at most 0.076 min; chain: 2/10 worlds broken, oracle at most 0.097 min.

**fig_connectivity.** Under ε 30 %, from all pairs to 0.2 linked: joint 0.033 → 0.117 min, composed pairwise 0.085 → 0.773 min. Indirect pairs under ε 30 %: joint 0.385 min on the chain and 1.140 min on the ring; composed pairwise 1.938 and 1.459 min.
