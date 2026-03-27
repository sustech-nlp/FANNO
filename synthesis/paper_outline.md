# FANNO-Dev: Paper Structure & Figure/Table Mapping

## §1 Introduction
- Motivation: instruction data diversity matters
- Problem: no quantitative diversity measurement for synthesized data
- Contribution summary (5 points from abstract)
- **Fig 1**: Source distribution overview

## §2 Related Work
- Self-Instruct, WizardLM, Evol-Instruct, Alpaca, LIMA
- DataFlow and data mixing optimization
- Vendi Score and diversity measurement
- **Tab 10**: Baseline comparison table

## §3 FANNO-Dev Framework
### §3.1 Synthesis Pipelines
- 8 pipeline descriptions with prompt examples
- **Tab 1**: Dataset statistics
- **Fig 14**: Cross-source distance heatmap

### §3.2 Data Cleaning
- Three-stage pipeline
- **Fig 10**: Per-source rejection rates
- **Tab 7**: Pipeline efficiency

### §3.3 Output Formats
- Alpaca and ShareGPT format descriptions

## §4 Diversity Analysis
### §4.1 Vendi Score Measurement
- Definition and computation
- **Tab 2**: Diversity metrics
- **Fig 5**: Per-source Vendi Score

### §4.2 Scaling Law
- Exponential saturation model
- **Fig 2**: Scaling curve with fit
- **Tab 4**: Scaling analysis

### §4.3 Template Space Analysis
- Domain × Type utilization
- **Fig 6**: Domain-Type heatmap
- **Fig 12**: Template utilization panels
- **Fig 16**: Domain Zipf distribution

### §4.4 Cross-Source Complementarity
- Leave-one-out analysis
- Self-Inversion as tag space discoverer
- **Tab 5**: Complementarity table
- **Fig 9**: Marginal contribution + coverage matrix

## §5 Data Selection Strategies
- 5 strategies compared
- **Fig 3**: Selection strategy comparison
- **Tab 3**: Selection strategy table
- Key finding: random near-optimal at scale

## §6 Deep Analysis
### §6.1 Dedup-Diversity Tradeoff
- **Fig 8**: Dedup aggressiveness curve
- **Tab 6**: Dedup tradeoff table

### §6.2 Pipeline Efficiency
- Rejection vs Vendi correlation (r=-0.776)
- **Fig 11**: Scatter plot
- **Tab 7**: Efficiency ranking

### §6.3 Linguistic Diversity
- **Fig 13**: Multi-panel comparison
- **Fig 15**: Answer quality (length vs TTR)

### §6.4 Embedding Space Visualization
- **Fig 7**: t-SNE plot

## §7 Discussion
- Cost analysis (Tab 9)
- Limitations: tag space utilization ceiling
- Future work: expanding template space
- **Tab 8**: Claims and evidence summary

## §8 Conclusion

## Appendix
- **Fig 4**: Length distribution
- Multi-turn dialog analysis
- Contamination check results
- Format validation results
