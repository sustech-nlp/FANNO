# FANNO-Dev Dataset Card

## Dataset Description

**FANNO-Dev** is a large-scale, diverse instruction-following dataset synthesized using GPT-4o through 8 specialized pipelines. It is designed to maximize semantic diversity (measured by Vendi Score) while maintaining high quality.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total samples | 151,031+ |
| Single-turn | 132,323 |
| Multi-turn | 18,708 |
| Unique domains | 2,297 |
| Question types | 25 |
| Language | English |
| Avg question length | 51 words |
| Avg answer length | 338 words |
| Vendi Score | 182.75 |
| Avg pairwise cosine distance | 0.9514 |

### Data Sources

| Source | Count | % | Description |
|--------|------:|---:|-------------|
| Complex QA | ~50K | 37% | Multi-hop, counterfactual, comparative, analogy questions |
| Reasoning QA | ~18K | 13% | Deductive, inductive, causal, spatial reasoning |
| Code QA | ~15K | 12% | Coding across 8 languages, 16 topics |
| Multi-Turn Dialog | ~17K | 12% | 8 conversation patterns, 15 scenarios |
| Math QA | ~11K | 8% | Elementary to competition-level mathematics |
| Document-Grounded QA | ~16K+ | 11% | FANNO-style document→question→answer pipeline |
| Creative Writing | ~9K | 6% | 12 creative writing tasks |
| Self-Inversion | ~5K | 3% | Trajectory inversion (question generation from answers) |

### Format

Available in two formats:
- **Alpaca** (`merged_alpaca.jsonl`): `{instruction, input, output, source, domain, difficulty, type}`
- **ShareGPT** (`merged_sharegpt.jsonl`): `{conversations: [{from, value}], source, domain}`

### Quality Assurance

Three-stage data cleaning pipeline:
1. **Quality filter** (99.9% pass rate): Refusal detection, length validation, character ratio
2. **Exact dedup** (MD5 hash): Removes identical samples
3. **Near dedup** (80-char prefix): Removes near-duplicate questions
4. Overall rejection rate: 27.6%

### Diversity Analysis

- **Vendi Score** (effective dimensionality): 182.75 on 15K embedding sample
- **Scaling law**: Vendi(N) = 141.1 × (1 - e^(-N/362)) + 38.4, R²=0.99
- **Saturation ceiling**: ~180 Vendi Score (limited by 4.1% tag space utilization)
- **Cross-source distance**: 0.96 average cosine distance (nearly orthogonal pipelines)
- **Optimal selection**: K-Center-Greedy yields +33% diversity for small subsets
- **Pipeline efficiency**: Self-Inversion most API-efficient (Eff=18.8, 0.5% rejection)
- **Linguistic diversity**: Document QA ranks #1 in composite (TTR + hapax + vocab)

### Embedding Space Analysis

- **Intrinsic dimensionality**: PCA requires 50+ components for 90% variance (high-dimensional distribution)
- **Self-Inversion**: Highest within-source diversity (0.943), appears in all 20 K-Means clusters
- **Code QA**: Most semantically unique (0.677 centroid distance from global center)
- **7 pure clusters, 6 mixed, 7 diverse** in 20-cluster analysis
- **Angle shift quality**: Mean cosine distance 0.458 between original and new angles (82.6% > 0.3)
- **Cross-metric correlation**: TTR predicts Vendi (r=0.830), Efficiency predicts Diversity (r=0.776)

### Efficiency

- **Estimated cost**: ~$2,245 total ($15.58 per 1K clean samples)
- **Pareto-optimal pipelines**: Self-Inversion and Document QA
- **Rejection-diversity correlation**: r = -0.776 (efficient pipelines are diverse)

### Generation

- **Model**: GPT-4o (Azure OpenAI, 17 endpoints)
- **Framework**: Extended FANNO (ACL 2025 Findings) with 7 additional synthesis pipelines
- **Parallel workers**: 30-50 concurrent API calls per pipeline
- **Templates**: 400+ prompt templates (domains × types × difficulties × styles)

### Intended Use

- Instruction-following fine-tuning of language models
- Diversity-aware data selection experiments
- Benchmarking data synthesis methodologies

### Citation

```bibtex
@inproceedings{fanno-dev-2026,
  title={FANNO-Dev: Diversity-Verified Large-Scale Instruction Data Synthesis},
  year={2026},
}
```

### License

Apache 2.0 (code); CC-BY-4.0 (data)
