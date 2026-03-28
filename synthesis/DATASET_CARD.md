# FANNO-Dev Dataset Card

## Dataset Description

**FANNO-Dev** is a large-scale, diverse instruction-following dataset synthesized using GPT-4o through 8 specialized pipelines. It is designed to maximize semantic diversity (measured by Vendi Score) while maintaining high quality.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total samples | 153,351 |
| Single-turn | 134,643 |
| Multi-turn | 18,708 |
| Unique domains | 2,297 |
| Question types | 25 |
| Language | English |
| Avg question length | 51 words |
| Avg answer length | 338 words |
| Vendi Score | 182.75 |
| Avg pairwise cosine distance | 0.951 |

### Data Sources

| Source | Count | % | Description |
|--------|------:|---:|-------------|
| Complex QA | 54,223 | 35.4% | Multi-hop, counterfactual, comparative, analogy questions across 20 domains, 12 types |
| Document QA | 21,432 | 14.0% | FANNO-style document→question→answer pipeline |
| Multi-Turn Dialog | 18,708 | 12.2% | 8 conversation patterns, 15 scenarios, avg 3.3 turns |
| Reasoning QA | 17,789 | 11.6% | 12 reasoning types: deductive, inductive, causal, spatial, etc. |
| Code QA | 15,411 | 10.0% | 8 programming languages, 16 topics |
| Math QA | 11,450 | 7.5% | 15 math topics, 5 levels (elementary to competition) |
| Creative Writing | 9,363 | 6.1% | 12 creative writing task types |
| Self-Inversion | 4,975 | 3.2% | Trajectory inversion: answer→question synthesis with 2,288 unique domains |

### Format

Available in two formats:
- **Alpaca** (`merged_alpaca.jsonl`, 125,280 samples): `{instruction, input, output, source, domain, difficulty, type}`
- **ShareGPT** (`merged_sharegpt.jsonl`, 141,901 samples): `{conversations: [{from, value}], source, domain}`

### Quality Assurance

Three-stage data cleaning pipeline:
1. **Quality filter** (99.9% pass rate): Refusal detection, length validation, character ratio
2. **Exact dedup** (MD5 hash): Removes identical samples
3. **Near dedup** (80-char prefix): Removes near-duplicate questions
4. Overall rejection rate: 27.4%

**Formal validation results:**
- 100% JSON valid in both formats
- 100% required fields present
- 100% valid conversation structure (ShareGPT)
- 0.01% refusal rate (14/125,280)
- 0 encoding issues
- 0 benchmark contamination (MMLU, GSM8K, HumanEval, ARC, HellaSwag)

### Diversity Analysis

- **Vendi Score** (effective dimensionality): 182.75 on 15K embedding sample
- **Scaling law**: Vendi(N) = 55.4 × (1 - e^(-N/745)) + 109.0, R²=0.996
- **Saturation ceiling**: ~164.5 Vendi Score (limited by 4.1% tag space utilization)
- **Cross-source distance**: 0.961 average cosine distance (nearly orthogonal pipelines)
- **Optimal selection**: K-Center-Greedy yields +33% diversity for small subsets (N=500)
- **Pipeline efficiency**: Self-Inversion most API-efficient (Eff=18.8, 0.5% rejection)
- **Diversity density**: Self-Inversion 32.4 Vendi/1K (12× more than Complex QA)
- **Linguistic diversity**: Document QA ranks #1 in composite (TTR + hapax + vocab)

### Embedding Space Analysis

- **Intrinsic dimensionality**: PCA requires 50+ components for 90% variance
- **Self-Inversion**: Highest within-source diversity (0.943), appears in all 20 K-Means clusters, lowest isolation (0.224)
- **Code QA**: Most semantically unique (0.677 centroid distance), highest isolation (0.917)
- **Creative Writing**: Most isolated (0.953), 91% multi-paragraph response structure
- **7 pure clusters, 6 mixed, 7 diverse** in 20-cluster analysis
- **20 latent topics** discovered via NMF: Code→5 topics, Math→5, Creative→1 pure
- **Angle shift quality**: Mean cosine distance 0.458 between original and new angles (82.6% > 0.3)
- **Q-A alignment**: Document QA best (Hit@1=97%), Creative lowest (Hit@1=32% by design)
- **Cross-metric correlation**: TTR→Vendi (r=0.830), Efficiency→Diversity (r=0.776)

### Efficiency

- **Estimated cost**: ~$2,245 total ($15.58 per 1K clean samples)
- **Pareto-optimal pipelines**: Self-Inversion and Document QA
- **Rejection-diversity correlation**: r = -0.776 (efficient pipelines are diverse)
- **Optimal mixing**: Upweight Self-Inversion (3.2%→17.4%) and Creative Writing (6.1%→12.5%)

### Generation

- **Model**: GPT-4o (Azure OpenAI, 17 endpoints)
- **Framework**: Extended FANNO (ACL 2025 Findings) with 7 additional synthesis pipelines
- **Parallel workers**: 30-50 concurrent API calls per pipeline
- **Templates**: 400+ prompt templates (domains × types × difficulties × styles)

### Self-Inversion as Tag-Space Discoverer

- **2,288 unique domains** generated via trajectory reversal (vs 20 for Complex QA)
- **2,274 exclusive domains** not found in any other pipeline
- Domain explosion factor: 143× more exclusive domains than Complex QA
- Bridges all 20 K-Means clusters (only pipeline with 100% coverage)
- Highest entropy (3.636 nats) and diversity density (25.1 Vendi/1K)

### Deliverables

- 111 publication-quality figures (PNG+PDF)
- 25 LaTeX tables
- 57 JSON analysis reports
- Comprehensive training recipe (LLaMA-Factory compatible)
- Curated sample showcase (16 examples from 8 pipelines)
- 117 git commits on dev branch

### Intended Use

- Instruction-following fine-tuning of language models
- Diversity-aware data selection experiments
- Benchmarking data synthesis methodologies
- Source mixing ratio optimization research

### Citation

```bibtex
@inproceedings{fanno-dev-2026,
  title={FANNO-Dev: Diversity-Verified Large-Scale Instruction Data Synthesis},
  year={2026},
}
```

### License

Apache 2.0 (code); CC-BY-4.0 (data)
