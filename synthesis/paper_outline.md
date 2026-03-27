# FANNO-Dev: Paper Structure & Figure/Table Mapping

## Total Deliverables: 34 Figures (PNG+PDF), 16 LaTeX Tables

---

## §1 Introduction
- Motivation: instruction data diversity matters
- Problem: no quantitative diversity measurement for synthesized data
- Contribution summary (5 points from abstract)
- **Fig 28**: Architecture overview diagram
- **Fig 1**: Source distribution overview

## §2 Related Work
- Self-Instruct, WizardLM, Evol-Instruct, Alpaca, LIMA
- DataFlow and data mixing optimization
- Vendi Score and diversity measurement
- **Tab 10**: Baseline comparison table (FANNO-Dev vs 7 methods)

## §3 FANNO-Dev Framework
### §3.1 Synthesis Pipelines
- 8 pipeline descriptions with prompt examples
- **Tab 1**: Dataset statistics
- **Tab 12**: Comprehensive pipeline stats (domains, types, Vendi, efficiency)
- **Fig 14**: Cross-source distance heatmap (0.961 avg)

### §3.2 Self-Inversion (Trajectory Reversal)
- Novel contribution: answer→question synthesis
- **Fig 32**: Angle shift quality analysis (mean dist=0.458, 82.6% > 0.3)
- 0% same angle, 5000 unique new angles

### §3.3 Data Cleaning
- Three-stage pipeline: quality → exact dedup → near dedup
- **Fig 10**: Per-source rejection rates
- **Tab 7**: Pipeline efficiency ranking

### §3.4 Multi-Turn Dialog
- 8 patterns, 15 scenarios, 2-8 turns
- **Fig 29**: Multi-turn detailed analysis
- **Tab 11**: Multi-turn statistics
- **Fig 19**: Depth quality analysis (TTR stable at 0.84-0.85)

### §3.5 Output Formats
- Alpaca and ShareGPT format descriptions
- **Fig 34**: Dataset composition overview

## §4 Diversity Analysis
### §4.1 Vendi Score Measurement
- Definition and computation
- **Tab 2**: Diversity metrics
- **Fig 5**: Per-source Vendi Score
- **Fig Dashboard**: 6-panel comprehensive overview

### §4.2 Scaling Law
- Exponential saturation model
- **Fig 30**: Scaling curve with fit (R²=0.996)
- **Fig 2**: Earlier scaling analysis
- **Tab 4**: Scaling analysis

### §4.3 Template Space Analysis
- Domain × Type utilization (4.1% used)
- **Fig 6**: Domain-Type heatmap
- **Fig 12**: Template utilization panels
- **Fig 16**: Domain Zipf distribution (long-tail)
- **Fig 18**: Type-difficulty heatmap

### §4.4 Cross-Source Complementarity
- Leave-one-out analysis
- Self-Inversion as tag space discoverer
- **Tab 5**: Complementarity table
- **Fig 9**: Marginal contribution + coverage matrix

## §5 Data Selection Strategies
- 5 strategies compared across 4 subset sizes
- **Fig 27**: Selection strategy comparison (3-panel)
- **Fig 3**: Earlier selection plot
- **Tab 3**: Selection strategy table
- **Tab 14**: Detailed Vendi scores per strategy per size
- Key finding: K-Center-Greedy +33% at N=500, random near-optimal at scale

## §6 Deep Analysis
### §6.1 Embedding Space Analysis
- t-SNE, PCA, cluster analysis
- **Fig 21**: t-SNE visualization (7 pipelines, 3500 points)
- **Fig 22**: PCA analysis (50+ components for 90% variance)
- **Fig 23**: Embedding space bubble chart
- **Tab 13**: Embedding analysis per-pipeline table

### §6.2 Cluster Analysis
- 20 K-Means clusters: 7 pure, 6 mixed, 7 diverse
- **Fig 25**: Cluster composition heatmap
- Self-Inversion in all 20 clusters

### §6.3 Coverage Analysis
- Density mapping, sparse/dense region identification
- **Fig 33**: Semantic space coverage with convex hulls
- Code/Math dominate sparse regions; Creative in dense regions

### §6.4 Pipeline Efficiency
- Rejection vs Vendi correlation (r=-0.776)
- **Fig 11**: Scatter plot with trend line
- **Fig 17**: Pareto efficiency frontier
- **Tab 7**: Efficiency ranking (Self-Inversion #1, Document QA #2)

### §6.5 Cross-Metric Correlation
- TTR→Vendi (r=0.830), Efficiency→Diversity (r=0.776)
- **Fig 31**: 6×6 correlation matrix
- **Fig 24**: Multi-dimensional radar comparison

### §6.6 Linguistic Diversity
- TTR, hapax ratio, vocabulary size
- **Fig 13**: Multi-panel comparison
- **Fig 15**: Answer quality (length vs TTR)
- Document QA: #1 composite diversity (0.921)

### §6.7 Question Complexity
- Q/A length ratio, bigrams per sample
- **Fig 26**: 4-panel complexity analysis
- **Tab 15**: Question complexity table

### §6.8 Dedup-Diversity Tradeoff
- **Fig 8**: Dedup aggressiveness curve
- **Tab 6**: Dedup tradeoff table

## §7 Discussion
- Cost analysis: $2,245 total, $15.58/1K (**Tab 9**)
- Limitations: tag space utilization ceiling (4.1%)
- Future work: expanding template space, cross-type hybridization
- **Tab 8**: Claims and evidence summary (15 claims)
- **Tab 16**: Key findings with cross-references

## §8 Conclusion

## Appendix
- **Fig 4**: Length distribution
- **Fig 7**: Earlier t-SNE plot
- **Fig 20**: Code language/topic distribution
- Contamination check results (0 matches)
- Format validation results (100% pass)
- **Tab 9**: Cost breakdown per pipeline
- LLaMA-Factory training recipe (see TRAINING_RECIPE.md)
