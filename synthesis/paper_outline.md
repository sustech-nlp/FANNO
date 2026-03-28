# FANNO-Dev: Paper Structure & Figure/Table Mapping

## Total Deliverables: 58 Figures (PNG+PDF), 24 LaTeX Tables, 40 JSON Reports

---

## §1 Introduction
- Motivation: instruction data diversity matters
- Problem: no quantitative diversity measurement for synthesized data
- Contribution summary (5 points from abstract)
- **Fig 28**: Architecture overview diagram (8 pipelines → cleaning → output)
- **Fig 1**: Source distribution overview
- **Fig 34**: Dataset composition overview (6-panel)

## §2 Related Work
- Self-Instruct, WizardLM, Evol-Instruct, Alpaca, LIMA
- DataFlow and data mixing optimization
- Vendi Score and diversity measurement
- **Tab 10**: Baseline comparison table (FANNO-Dev vs 7 methods)

## §3 FANNO-Dev Framework
### §3.1 Synthesis Pipelines
- 8 pipeline descriptions with prompt examples
- **Tab 1**: Dataset statistics (153,351 cleaned from 211K raw)
- **Tab 12**: Comprehensive pipeline stats (domains, types, Vendi, efficiency)
- **Tab 18**: Master per-pipeline table (volume, diversity, embedding, efficiency, quality)
- **Fig 14**: Cross-source distance heatmap (0.961 avg)

### §3.2 Self-Inversion (Trajectory Reversal)
- Novel contribution: answer→question synthesis
- **Fig 32**: Angle shift quality analysis (mean dist=0.458, 82.6% > 0.3)
- 0% same angle, 5000 unique new angles

### §3.3 Data Cleaning
- Three-stage pipeline: quality → exact dedup → near dedup
- **Fig 10**: Per-source rejection rates (0.5% Self-Inversion to 48.5% Code QA)
- **Tab 7**: Pipeline efficiency ranking

### §3.4 Multi-Turn Dialog
- 8 patterns, 15 scenarios, 2-8 turns
- **Fig 29**: Multi-turn detailed analysis (4-panel)
- **Tab 11**: Multi-turn statistics
- **Fig 19**: Depth quality analysis (TTR stable at 0.84-0.85)
- **Fig 35**: Coherence analysis (adjacency sim=0.575, decay curve, Q-A pattern)

### §3.5 Output Formats
- Alpaca (125,280 samples) and ShareGPT (141,901 samples) format descriptions

## §4 Diversity Analysis
### §4.1 Vendi Score Measurement
- Definition and computation
- **Tab 2**: Diversity metrics
- **Fig 5**: Per-source Vendi Score
- **Fig Dashboard**: 6-panel comprehensive overview

### §4.2 Scaling Law
- Exponential saturation model: V(N) = 55.4(1-exp(-N/745)) + 109.0, R²=0.996
- **Fig 30**: Scaling curve with 12 data points + fit
- **Fig 2**: Earlier scaling analysis
- **Tab 4**: Scaling analysis (ceiling at 164.5 for single-turn)

### §4.3 Template Space Analysis
- Domain × Type utilization (4.1% of possible combinations used)
- **Fig 6**: Domain-Type heatmap
- **Fig 12**: Template utilization panels
- **Fig 16**: Domain Zipf distribution (long-tail, 2,297 unique domains)
- **Fig 18**: Type-difficulty heatmap

### §4.4 Cross-Source Complementarity
- Leave-one-out analysis
- Self-Inversion as tag space discoverer
- **Tab 5**: Complementarity table
- **Fig 9**: Marginal contribution + coverage matrix

## §5 Data Selection Strategies
- 5 strategies compared across 4 subset sizes
- **Fig 27**: Selection strategy comparison (3-panel: Vendi vs N, relative improvement, quality-speed)
- **Fig 3**: Earlier selection plot
- **Tab 3**: Selection strategy table
- **Tab 14**: Detailed Vendi scores per strategy per size
- Key finding: K-Center-Greedy +33% at N=500, random near-optimal at scale

## §6 Deep Analysis
### §6.1 Embedding Space Analysis
- t-SNE, PCA, cluster analysis
- **Fig 21**: t-SNE visualization (7 pipelines, 3500 points)
- **Fig 22**: PCA analysis (50+ components for 90% variance)
- **Fig 23**: Embedding space bubble chart (within-diversity vs centroid distance)
- **Tab 13**: Embedding analysis per-pipeline table

### §6.2 Cluster Analysis
- 20 K-Means clusters: 7 pure, 6 mixed, 7 diverse
- **Fig 25**: Cluster composition heatmap + size/purity bars
- Self-Inversion in all 20 clusters

### §6.3 Coverage Analysis
- Density mapping, sparse/dense region identification
- **Fig 33**: Semantic space coverage with convex hulls
- Code/Math dominate sparse regions; Creative in dense regions

### §6.4 Source Overlap & Isolation
- KNN overlap (K=20) in embedding space
- **Fig 42**: Overlap heatmap, isolation scores, top overlapping pairs
- **Tab 20**: Source isolation summary
- Creative (0.953) and Code (0.917) most isolated; Self-Inversion bridges all (0.224)

### §6.5 Pipeline Efficiency
- Rejection vs Vendi correlation (r=-0.776)
- **Fig 11**: Scatter plot with trend line
- **Fig 17**: Pareto efficiency frontier
- **Tab 7**: Efficiency ranking (Self-Inversion #1, Document QA #2)

### §6.6 Cross-Metric Correlation
- TTR→Vendi (r=0.830), Efficiency→Diversity (r=0.776)
- **Fig 31**: 6×6 correlation matrix
- **Fig 24**: Multi-dimensional radar comparison (5-axis per pipeline)

### §6.7 Linguistic Diversity
- TTR, hapax ratio, vocabulary size, n-gram counts
- **Fig 13**: Multi-panel comparison
- **Fig 15**: Answer quality (length vs TTR)
- **Fig 45**: N-gram diversity (unique bigrams, vocabulary overlap, hapax ratio)
- **Fig 38**: Keyword analysis (TF-IDF distinctive terms per pipeline)
- Document QA: #1 composite diversity (0.921), 32.7% exclusive vocabulary

### §6.8 Question Complexity & Q-A Alignment
- Q/A length ratio, bigrams per sample
- **Fig 26**: 4-panel complexity analysis
- **Tab 15**: Question complexity table
- **Fig 46**: Q-A alignment (Document QA Hit@1=97%, Creative 32%)

### §6.9 Quality Scoring
- Heuristic quality scores (length, TTR, structure)
- **Fig 36**: Quality score distribution per pipeline (all >72, Code QA 85.1)
- **Fig 37**: Difficulty-level diversity (inverted-U: easy < expert < hard < medium)
- **Tab 17**: Difficulty diversity table
- **Fig 44**: Response structure distribution (code blocks, numbered steps, paragraphs)

### §6.10 Information-Theoretic Analysis
- Shannon entropy, JSD, cluster coverage per source
- **Fig 50**: Entropy, JSD heatmap, information decomposition
- Self-Inversion: highest entropy (3.636 nats), 96% cluster coverage
- Overall uniformity: 96.4%

### §6.11 Dedup-Diversity Tradeoff
- **Fig 8**: Dedup aggressiveness curve
- **Tab 6**: Dedup tradeoff table

## §7 Discussion
- Cost analysis: $2,245 total, $15.58/1K (**Tab 9**)
- Optimal source ordering: Self-Inv + Doc QA = 96% of max diversity (**Fig 49**)
- Leave-one-out: Doc QA most valuable (+6.2%), Creative hurts (-3.2%) (**Fig 48**)
- Mixing ratio optimization: oversample Self-Inversion 3.2%→17.4% (**Fig 39**)
- Per-type diversity: 60 types analyzed, Vendi range [2.6, 64.5] (**Fig 41**, **Tab 19**)
- Per-domain diversity: 20 Complex QA domains, Sci&Tech highest (**Fig 47**)
- Topic modeling: 20 NMF topics, pipeline-specific clusters (**Fig 40**)
- Limitations: tag space utilization ceiling (4.1%), Vendi saturation
- Future work: expanding template space, cross-type hybridization
- **Tab 8**: Claims and evidence summary
- **Tab 16**: Key findings with cross-references (17 findings)
- **Tab 21**: Comprehensive evidence summary (22 claims)

### §6.12 UMAP Visualization
- Alternative to t-SNE: UMAP with 3 parameter configs
- **Fig 51**: 3-panel UMAP visualization (n=15/30/50)
- Self-Inversion most spread (6.158), Creative most compact (1.279)

### §6.13 Self-Inversion Deep Dive
- Trajectory reversal quality and domain explosion
- **Fig 55**: 4-panel deep dive (domain distribution, top-20, Q-A lengths, exclusivity)
- 2,274 exclusive domains (vs 16 for Complex QA, 15 for Code)

### §6.14 Per-Pipeline Composition
- **Fig 52**: Code QA cross-language analysis (8 languages, 16 topics)
- **Fig 53**: Math & Reasoning detailed composition (15 math topics, 12 reasoning types)
- **Fig 54**: Creative Writing & Complex QA composition (12 creative types, 20 domains, 12 QA types)

## §7 Discussion
- Cost analysis: $2,245 total, $15.58/1K (**Tab 9**)
- Optimal source ordering: Self-Inv + Doc QA = 96% of max diversity (**Fig 49**)
- Leave-one-out: Doc QA most valuable (+6.2%), Creative hurts (-3.2%) (**Fig 48**)
- Mixing ratio optimization: oversample Self-Inversion 3.2%→17.4% (**Fig 39**)
- Per-type diversity: 60 types analyzed, Vendi range [2.6, 64.5] (**Fig 41**, **Tab 19**)
- Per-domain diversity: 20 Complex QA domains, Sci&Tech highest (**Fig 47**)
- Topic modeling: 20 NMF topics, pipeline-specific clusters (**Fig 40**)
- Limitations: tag space utilization ceiling (4.1%), Vendi saturation
- Future work: expanding template space, cross-type hybridization
- **Tab 8**: Claims and evidence summary
- **Tab 16**: Key findings with cross-references (17 findings)
- **Tab 21**: Comprehensive evidence summary (22 claims)
- **Tab 23**: Master per-pipeline comparison (10 dimensions)
- **Tab 24**: Cross-reference index (20 claims with evidence pointers)

## §8 Conclusion

## Appendix
- **Fig 4**: Length distribution
- **Fig 7**: Earlier t-SNE plot
- **Fig 20**: Code language/topic distribution
- **Fig 43**: 8-panel poster summary
- **Fig 56**: Grand summary figure (8-panel overview)
- **Fig 57**: Comprehensive 8-axis radar chart (per-pipeline)
- **Tab 22**: Curated sample showcase (2 examples per pipeline)
- Contamination check results (0 real matches, 416 false positives)
- Format validation results (100% pass)
- **Tab 9**: Cost breakdown per pipeline
- LLaMA-Factory training recipe (see TRAINING_RECIPE.md)
