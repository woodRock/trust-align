# Draft figures (scripts/11_draft_connectivity.py)

DRAFT: fixed λ = 10, K = 10, 20 anchors per kept pair, drift family (i), 10 seeds. Superseded by the pre-registered Experiment 1 grid. Numbers are computed.

**regime_map.png.** Median map error of the joint model minus that of composed pairwise alignment (left) and of a star fit through the best-connected batch, on the pairs the star reaches (right), for each anchor-graph structure and noise level. Blue: joint lower. The joint model's largest advantage over composed pairwise is -1.336 min (chain, ε 30%).

**loop_closure.png.** Median map error on batch pairs without direct anchors as loops are added to a chain of 10 batches. Under ε 30%: joint 0.385 min on the chain and 1.140 min on the ring; composed pairwise 1.938 and 1.459 min.

**edge_removal.png.** Median map error over all batch pairs as direct anchor connections are removed from the complete graph (nested deletions, always connected). Under ε 30%, from 1 to 0.2 of pairs kept: joint 0.033 → 0.117 min, composed pairwise 0.085 → 0.773 min.

