# FANNO-Dev Development Log

## Session: 2026-03-27 (Data Synthesis Sprint)

### Goal
Beat DataFlow on QA and VQA synthesis capability. Core: verify diversity can scale. Generate FANNO 100K diverse data + 10K multi-turn + trajectory inversion data.

---

### Phase 1: Repository Setup & Exploration

**Q: What is the existing FANNO framework?**

**Analysis:**
FANNO (Free ANNOtator) is an end-to-end instruction data synthesis framework (ACL 2025 Findings). Key components:
1. **Three-stage pipeline**: Document → Question → Answer
2. **Seed generation**: 96 tag combinations (4 difficulty × 12 task types × 2 styles) per document
3. **UCB + Think-Different augmentation**: Uses Upper Confidence Bound to select good seed examples as counterexamples, forcing new instructions to be different
4. **Community Detection diversity filter**: Embeds instructions → cosine similarity graph → community detection → keep one per cluster
5. **IFD + PPL quality scoring**: Instruction Following Difficulty + Perplexity composite score

**Files examined**: `src/fanno/pipeline.py`, `src/fanno/template/seed_template.py`, `src/fanno/template/ucb_template.py`, `src/fanno/evaluator.py`, `src/fanno/strategies/selection.py`, `src/fanno/strategies/response.py`

**Decision**: Reuse FANNO's core seed QA pipeline but extend with GPT-4o for large-scale synthesis. Add 6 new data types beyond basic QA.

---

### Phase 2: Synthesis Framework Design

**Q: What data types should we synthesize?**

**Analysis & Distribution**:

| Type | Target | Rationale |
|------|--------|-----------|
| FANNO Seed QA | 30K | Core FANNO approach (doc → Q → A) with GPT-4o |
| Complex QA | 25K | Standalone multi-hop, counterfactual, comparative QA |
| Code QA | 15K | Practical coding across 8 languages, 16 topics |
| Math QA | 10K | Elementary to competition level, 15 topics |
| Reasoning QA | 10K | Deductive, inductive, causal, spatial reasoning |
| Creative Writing | 5K | Stories, essays, poetry, 12 writing tasks |
| Multi-turn Dialog | 10K | 8 conversation patterns, 15 scenarios |
| Trajectory Inversion | 5K | Reverse-engineer Qs from existing answers |

**Architecture**: `synthesis/` directory with:
- `api_client.py` - Azure OpenAI multi-endpoint load balancer (17 GPT-4o endpoints)
- `prompts/templates.py` - All prompt templates (extended FANNO tags + new types)
- `synthesize.py` - Main synthesis pipeline with parallel batch processing
- `trajectory_inversion.py` - Three modes: basic, verified, self-inversion
- `evaluate_diversity.py` - Comprehensive diversity metrics

**Commits**: `a5541be`, `d137f80`, `fdcea23` on `dev` branch

---

### Phase 3: API Infrastructure

**Q: How to handle Azure CLI authentication in parallel processes?**

**Root Cause**: `AzureCliCredential` invokes `az` CLI subprocess, which doesn't work in background processes spawned by Claude Code (sandbox limitations).

**Solution**: Pre-fetch token via `az account get-access-token`, cache to `/tmp/.fanno_azure_token`, use `azure_ad_token=` parameter instead of `azure_ad_token_provider=`. Token refresh daemon runs every 30 minutes.

**Files Changed**: `synthesis/api_client.py`, `synthesis/refresh_token.sh`

---

### Phase 4: Large-Scale Synthesis (Running)

**Started**: 2026-03-27 18:55 UTC
**All 7 pipelines running in parallel + 1 trajectory inversion pipeline**

#### Progress Snapshots:

| Time | Complex | Code | Math | Reasoning | Creative | Multi-turn | Seed QA | Self-Inv | Total |
|------|---------|------|------|-----------|----------|------------|---------|----------|-------|
| +30s | 390 | 197 | 194 | 366 | 187 | 0 | 0 | 0 | 1,334 |
| +2m | 780 | 787 | 948 | 366 | 366 | 177 | 89 | 0 | 3,513 |
| +4m | 1,362 | 1,381 | 1,515 | 1,082 | 1,097 | 352 | 276 | 0 | 7,065 |
| +6m | 2,543 | 2,167 | 2,866 | 2,150 | 2,014 | 801 | 460 | 0 | 13,001 |
| +12m | 4,305 | 3,943 | 4,414 | 3,416 | 3,512 | 1,323 | 827 | 787 | 22,527 |

**Rate**: ~1,800 samples/minute sustained

#### Intermediate Diversity Report (at 13K samples):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| 3-gram diversity | 0.5798 | High structural diversity |
| Hash diversity (Jaccard) | 0.9389 | Almost no semantic duplicates |
| Exact duplicate rate | 2.25% | Very low |
| Domain coverage | 21 unique domains | Good breadth |
| Type coverage | 25 unique types | Excellent |
| Domain uniformity | 0.3548 | Could improve (seed QA lacks domain tags) |
| Type uniformity | 0.5464 | Moderate |

---

### Phase 5: Diversity Evaluation Design

**Metrics implemented** (in `evaluate_diversity.py`):

1. **N-gram diversity**: 2-gram and 3-gram unique ratio
2. **Lexical diversity**: TTR, Root TTR, Log TTR, MTLD approximation
3. **Hash diversity**: MinHash-based Jaccard distance approximation (no embedding model needed)
4. **Coverage metrics**: Domain/Type/Difficulty entropy and uniformity
5. **Deduplication**: Exact dedup, prefix near-dedup, hash dedup

**Comparison with `diversity_metric/` toolkit**: Our eval uses lightweight hash-based methods for real-time monitoring during synthesis. For final evaluation, will use the full toolkit with embedding-based metrics (Vendi Score, K-Center coverage, NovelSum).

---

### Trajectory Inversion Pipeline Design

**Core idea**: Some trajectories are verifiable even when final answers are wrong. Intermediate reasoning steps are valuable.

**Three modes**:
1. **Basic**: Answer → reverse-engineer question
2. **Verified**: Identify correct intermediate steps → create improved QA pair
3. **Self-inversion**: Use already-synthesized data as trajectory source, generate questions from different angles

**Scientific insight**: Self-inversion creates a feedback loop that discovers new question types not present in the original synthesis prompts. This is a form of data augmentation that increases diversity without additional source documents.

---

### Key Findings (So Far)

1. **GPT-4o parallelism scales well**: 50 workers across 17 endpoints = ~1,800 samples/min with retry handling
2. **JSON mode significantly improves structured output**: ~95% parse success rate
3. **FANNO seed QA is slower**: Document-based pipeline (2-stage: Q gen + A gen) is ~3x slower per sample than standalone prompts
4. **Multi-turn is most expensive**: Longer outputs + more tokens = fewer samples per minute
5. **500 errors from Azure are transient**: ~5-10% failure rate, all recovered via retry
6. **Domain uniformity is a challenge**: Need explicit domain balancing in prompt generation
7. **Self-inversion produces surprisingly diverse questions**: Different angles on same knowledge

---

### TODO

- [ ] Wait for synthesis to reach 100K target
- [ ] Run full diversity evaluation with embedding-based metrics
- [ ] Run Vendi Score comparison vs random/k-means/community selection baselines
- [ ] Compare FANNO diversity metrics with DataFlow
- [ ] Merge all data into unified dataset
- [ ] Quality spot-check on samples from each category
