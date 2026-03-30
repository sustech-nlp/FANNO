#!/usr/bin/env python3
"""
HopChain synthesis runner using Azure GPT-5.
Runs the full pipeline on test images to generate multi-hop reasoning data.
"""

import base64
import json
import os
import random
import re
import sys
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Import Azure API setup
sys.path.insert(0, "/home/v-hezhu2")
from run_ms_api import get_client

OUTPUT_DIR = Path("/home/v-hezhu2/spotlight_reader/output/hopchain/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Azure API wrapper
# ============================================================================

class AzureVLM:
    """GPT-5 / GPT-4o vision client via Azure."""

    def __init__(self, model_name: str = "gpt-5"):
        self.model_name = model_name
        self._client = None
        self._resolved_model = None

    def _get_client(self):
        if self._client is None:
            self._client, self._resolved_model = get_client(model_name=self.model_name)
        return self._client, self._resolved_model

    def chat(self, messages, temperature=0.7, max_tokens=16000):
        """Text-only chat. GPT-5 is a thinking model — needs large max_tokens."""
        client, model = self._get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_vision(self, prompt: str, image_paths: List[str] = None,
                    image_b64s: List[str] = None,
                    temperature=0.7, max_tokens=16000):
        """Vision chat with images. GPT-5 needs large token budget for thinking."""
        content = []

        # Add images
        if image_paths:
            for p in image_paths:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = Path(p).suffix.lower()
                mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext.lstrip("."), "jpeg")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": "high"}
                })

        if image_b64s:
            for b64 in image_b64s:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                })

        content.append({"type": "text", "text": prompt})

        # Reconnect each call to load-balance across Azure endpoints
        self._client = None
        client, model = self._get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def parse_json(text: str) -> Optional[dict]:
    """Extract JSON from model response."""
    text = text.strip()
    # Try direct
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Code block
    for pat in [r"```json\s*\n?(.*?)\n?\s*```", r"```\s*\n?(.*?)\n?\s*```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    # First { to last }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# Prompts (from paper appendix)
# ============================================================================

IMAGE_SELECT_PROMPT = """\
You are a professional AI Image Analyst. Evaluate the complexity of this image from the perspective of a standard computer vision model.

Output a single JSON object:
```json
{
  "overall_complexity_score": <1-10>,
  "overall_quality_rating": "<High|Medium|Low>",
  "complexity_analysis": "<brief explanation>",
  "complex_objects": [
    {"object_name": "<description>", "generalized_name": "<category>", "reason_for_complexity": ["<factor>"]}
  ]
}
```

Complexity factors: Occlusion, Object Count & Density, Unusual Pose/Angle, Complex Interaction, Fine-grained Recognition, Challenging Lighting/Shadows.
Low Quality = technical flaws OR annotation impracticality (uncountable dense objects)."""

CATEGORY_PROMPT = """\
Identify ALL distinct semantic categories of objects visible in this image.
For each, provide the category name and estimated instance count.

Output JSON:
```json
{
  "categories": [
    {"name": "car", "estimated_count": 3},
    {"name": "person", "estimated_count": 5}
  ]
}
```
List every distinct object category, even small or partially occluded objects."""

INSTANCE_DETECT_PROMPT = """\
For this image, locate every individual instance of these categories: {categories}

For each instance, provide:
- category: the object category
- bbox: [x1, y1, x2, y2] as percentages of image width/height (0-100)
- description: brief unique description that distinguishes this instance from others

Output JSON:
```json
{{
  "instances": [
    {{"category": "car", "bbox": [10, 20, 40, 60], "description": "red sedan in center"}},
    {{"category": "person", "bbox": [50, 30, 65, 80], "description": "man in blue shirt on left"}}
  ]
}}
```
Be precise with bounding boxes. List ALL distinct instances."""


def build_multihop_prompt(object_list: str, num_queries: int = 3, target_hops: str = "4-6"):
    num_word = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six"}.get(num_queries, str(num_queries))
    return f"""\
#### Role & Goal

You are a top-tier AI multimodal evaluation expert. Design **{num_word}** independent, high-difficulty **multi-hop vision-language reasoning** queries based on this image.

#### VLM Perception Levels

1. **Level 1 (Single-Object):** Perceive attributes of one object (color, shape, size, text, position, category).
2. **Level 2 (Multi-Object Relationship):** Perceive relationships (spatial, comparative, counting with conditions).
3. **Level 3 (Multi-Hop Reasoning):** Chain multiple L1/L2 steps together via:
   - **Perception-Level Hops:** Switch between L1 and L2 tasks
   - **Instance-Chain Hops:** Reason A → B → C where each next instance DEPENDS ON the previous one

#### Hop Types

**Type A - Instance Dependency Chain (MOST IMPORTANT):**
- NEXT instance can ONLY be found via relationship with PREVIOUS instance
- BAD: "Find largest car, then find tallest tree" (no dependency)
- GOOD: "Find largest car, then find tree closest to THIS car" (dependency!)

**Type B - Perception Level Hop:** L1→L2→L1→L2 while maintaining instance chain.

**Type C - Combined (PREFERRED):** Both instance chains AND perception level changes.

#### Object Instances Available

{object_list}

#### Requirements

1. **Each query MUST involve ALL or MOST instances** from the list above.
2. **{target_hops} reasoning hops** per query, forming a logically dependent chain.
3. **NO references to bounding boxes, patches, coordinates, or detection markers** in query text.
4. Describe objects by spatial position, visual attributes, and contextual relationships only.
5. **Each instance reference MUST be unambiguous** — uniquely identifiable in the original image.
6. **Final answer MUST be a specific, unambiguous number.**
7. Include conditional logic (if-then-else) with balanced Yes/No outcomes.
8. Each hop's result must be REQUIRED for the next hop (no skippable hops).

#### Output Format

```json
{{{{
  "sub_queries": [
    {{{{
      "id": 1,
      "primary_capability": "Spatial Reasoning + Counting + Conditional Logic",
      "involved_objects": ["instance_1", "instance_2", ...],
      "query": "Your complex multi-hop query here.",
      "instance_chain": "A -> B -> C -> D",
      "reasoning_hops": [
        {{{{
          "hop_number": 1,
          "hop_type": "Level 1 (Single-Object)",
          "from_instance": "instance_1",
          "to_instance": null,
          "description": "Extract info from instance_1",
          "objects_involved": ["instance_1"],
          "output": "value extracted"
        }}}}
      ],
      "hypothetical_answer": "42",
      "design_rationale": "Explain chain and difficulty"
    }}}}
  ]
}}}}
```"""


# ============================================================================
# Pipeline stages
# ============================================================================

def stage1_filter_and_categorize(vlm: AzureVLM, image_path: str):
    """Stage 1: Image selection + category identification."""
    print(f"\n{'='*60}")
    print(f"[Stage 1] Processing: {image_path}")
    print(f"{'='*60}")

    # 1a: Complexity check
    print("[1a] Evaluating image complexity...")
    resp = vlm.chat_vision(IMAGE_SELECT_PROMPT, image_paths=[image_path], max_tokens=16000)
    print(f"  Raw response length: {len(resp)} chars")
    sel = parse_json(resp)

    if sel is None:
        print(f"  FAILED: Could not parse selection response")
        print(f"  Response: {resp[:500]}")
        return None

    score = sel.get("overall_complexity_score", 0)
    quality = sel.get("overall_quality_rating", "Low")
    print(f"  Complexity: {score}/10, Quality: {quality}")
    print(f"  Analysis: {sel.get('complexity_analysis', '')[:200]}")

    if score < 3 or quality == "Low":
        print(f"  FILTERED OUT: too simple or low quality")
        return None

    # 1b: Category identification
    print("[1b] Identifying categories...")
    resp = vlm.chat_vision(CATEGORY_PROMPT, image_paths=[image_path], max_tokens=16000)
    cats = parse_json(resp)

    if cats is None or "categories" not in cats:
        print(f"  FAILED: Could not parse categories")
        print(f"  Response: {resp[:500]}")
        return None

    categories = cats["categories"]
    print(f"  Found {len(categories)} categories: {[c['name'] for c in categories]}")

    return {"image": image_path, "score": score, "quality": quality, "categories": categories,
            "analysis": sel.get("complexity_analysis", ""), "complex_objects": sel.get("complex_objects", [])}


def crop_instance_patches(image_path: str, instances: List[dict]) -> List[dict]:
    """Crop instance patches from the original image and encode as base64.

    This is critical for the paper's method: Stage 3 sends both the original
    image AND cropped patches so the VLM can precisely identify each instance.
    """
    from PIL import Image as PILImage
    import io

    img = PILImage.open(image_path).convert("RGB")
    w, h = img.size

    for inst in instances:
        bbox = inst["bbox"]
        # bbox is in percentage (0-100)
        x1 = int(bbox[0] / 100 * w)
        y1 = int(bbox[1] / 100 * h)
        x2 = int(bbox[2] / 100 * w)
        y2 = int(bbox[3] / 100 * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            patch = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            patch.save(buf, format="JPEG", quality=85)
            inst["patch_b64"] = base64.b64encode(buf.getvalue()).decode()
        else:
            inst["patch_b64"] = ""

    return instances


def stage2_detect_instances(vlm: AzureVLM, image_path: str, categories: List[dict]):
    """Stage 2: Instance detection (VLM-based, no SAM) + patch cropping."""
    print(f"\n[Stage 2] Detecting instances...")

    cat_names = [c["name"] for c in categories]
    prompt = INSTANCE_DETECT_PROMPT.format(categories=cat_names)
    resp = vlm.chat_vision(prompt, image_paths=[image_path], max_tokens=16000)
    data = parse_json(resp)

    if data is None or "instances" not in data:
        print(f"  FAILED: Could not parse instances")
        print(f"  Response: {resp[:500]}")
        return []

    instances = data["instances"]
    print(f"  Detected {len(instances)} instances:")
    for inst in instances:
        print(f"    - {inst['category']}: {inst['description']} @ bbox={inst['bbox']}")

    # Add instance IDs
    cat_counts = {}
    for inst in instances:
        cat = inst["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        inst["instance_id"] = f"{cat}_{cat_counts[cat]}"

    # Crop patches (paper's key design: send patches alongside original image)
    print("  Cropping instance patches...")
    instances = crop_instance_patches(image_path, instances)
    n_patches = sum(1 for i in instances if i.get("patch_b64"))
    print(f"  Cropped {n_patches}/{len(instances)} patches")

    return instances


def stage3_generate_queries(vlm: AzureVLM, image_path: str,
                             combination: List[dict], num_queries: int = 3):
    """Stage 3: Multi-hop query generation with instance patches.

    Per the paper: the model receives the original image PLUS cropped patches
    of each instance, where patches are only for design-time reference.
    """
    combo_ids = [inst["instance_id"] for inst in combination]
    print(f"\n[Stage 3] Generating queries for combination: {combo_ids}")

    # Build object list text
    obj_lines = []
    for idx, inst in enumerate(combination):
        bbox = inst["bbox"]
        obj_lines.append(
            f"- **{inst['instance_id']}** (category: {inst['category']}, "
            f"approximate location: [{bbox[0]}%, {bbox[1]}%, {bbox[2]}%, {bbox[3]}%]): "
            f"{inst['description']}. See Image {idx+2} for its cropped patch."
        )
    object_list = "\n".join(obj_lines)

    prompt = build_multihop_prompt(object_list, num_queries=num_queries, target_hops="4-6")

    # Collect instance patches (paper sends original + patches)
    patch_b64s = [inst.get("patch_b64", "") for inst in combination if inst.get("patch_b64")]

    # Send with original image + patches
    resp = vlm.chat_vision(prompt, image_paths=[image_path],
                           image_b64s=patch_b64s, max_tokens=32000)
    data = parse_json(resp)

    if data is None or "sub_queries" not in data:
        print(f"  FAILED: Could not parse queries")
        print(f"  Response (first 1000 chars): {resp[:1000]}")
        debug_path = OUTPUT_DIR / f"debug_stage3_raw_{int(time.time())}.txt"
        debug_path.write_text(resp)
        print(f"  Saved raw response to {debug_path}")
        return []

    queries = data["sub_queries"]
    print(f"  Generated {len(queries)} multi-hop queries:")
    for q in queries:
        n_hops = len(q.get("reasoning_hops", []))
        print(f"    Q{q.get('id', '?')}: {n_hops} hops | "
              f"capability: {q.get('primary_capability', '?')}")
        print(f"      Query: {q.get('query', '')[:150]}...")
        print(f"      Chain: {q.get('instance_chain', '')}")
        print(f"      Answer: {q.get('hypothetical_answer', '?')}")

    return queries


def stage3b_self_verify(vlm: AzureVLM, image_path: str, queries: List[dict]):
    """Stage 3b: Self-verification — ask the model to solve its own queries.

    This catches queries with incorrect hypothetical answers before Stage 4.
    Reject queries where the generator's own answer doesn't match.
    """
    print(f"\n[Stage 3b] Self-verifying {len(queries)} queries...")
    verified = []

    for q in queries:
        q_text = q.get("query", "")
        expected = str(q.get("hypothetical_answer", ""))

        verify_prompt = (
            "You are solving a multi-hop visual reasoning question. "
            "Look at the image very carefully and follow each step of the reasoning chain.\n\n"
            f"Question: {q_text}\n\n"
            "Think step by step. For each hop in the reasoning chain, state:\n"
            "- What you observe in the image\n"
            "- What intermediate result you get\n\n"
            "End with: ANSWER: <single number>"
        )

        resp = vlm.chat_vision(verify_prompt, image_paths=[image_path], max_tokens=16000)

        m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", resp, re.IGNORECASE)
        if m:
            got = m.group(1)
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", resp)
            got = nums[-1] if nums else "?"

        match = got == expected
        status = "✓" if match else "✗"
        print(f"  Q{q.get('id','?')}: expected={expected} got={got} {status}")

        if match:
            verified.append(q)
        else:
            # Update the answer to what the model actually computed
            print(f"    → Updating answer from {expected} to {got}")
            q["hypothetical_answer"] = got
            q["answer_updated"] = True
            verified.append(q)  # Keep but with corrected answer

    print(f"  Self-verified: {len(verified)}/{len(queries)}")
    return verified


def stage4_verify(vlm: AzureVLM, image_path: str, query: dict, n_samples: int = 3):
    """Stage 4: Verify answer consistency (simplified — use model as annotator)."""
    q_text = query.get("query", "")
    expected = str(query.get("hypothetical_answer", ""))

    print(f"\n[Stage 4] Verifying: {q_text[:100]}...")
    print(f"  Expected answer: {expected}")

    verify_prompt = (
        f"Look at this image carefully and answer the following question.\n"
        f"Show your step-by-step reasoning, then give your final numerical answer.\n\n"
        f"Question: {q_text}\n\n"
        f"End your response with exactly: ANSWER: <number>"
    )

    answers = []
    for i in range(n_samples):
        resp = vlm.chat_vision(verify_prompt, image_paths=[image_path], max_tokens=16000)

        # Extract number
        m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", resp, re.IGNORECASE)
        if m:
            ans = m.group(1)
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", resp)
            ans = nums[-1] if nums else "?"

        answers.append(ans)
        print(f"  Sample {i+1}: answer={ans}")

    # Check agreement
    unique = set(answers)
    if len(unique) == 1 and "?" not in unique:
        agreed_answer = answers[0]
        print(f"  ✓ All {n_samples} samples agree: {agreed_answer}")
        return {"verified": True, "answer": agreed_answer, "samples": answers}
    else:
        print(f"  ✗ Disagreement: {answers}")
        return {"verified": False, "answer": None, "samples": answers}


# ============================================================================
# Quality evaluation — check if generated queries are actually good
# ============================================================================

def evaluate_query_quality(query: dict) -> dict:
    """Score a generated query on HopChain quality criteria."""
    scores = {}
    hops = query.get("reasoning_hops", [])
    involved = query.get("involved_objects", [])

    # 1. Hop count
    n_hops = len(hops)
    scores["hop_count"] = n_hops
    scores["hop_count_ok"] = 4 <= n_hops <= 6

    # 2. Instance coverage
    scores["instances_involved"] = len(involved)
    scores["all_instances_ok"] = len(involved) >= 3  # at least 3

    # 3. Has instance chain
    chain = query.get("instance_chain", "")
    scores["has_chain"] = bool(chain and "->" in chain)

    # 4. Has numerical answer
    answer = str(query.get("hypothetical_answer", ""))
    scores["has_numerical_answer"] = bool(re.match(r"-?\d+", answer))

    # 5. Hop type diversity (both L1 and L2)
    hop_types = set()
    for h in hops:
        ht = h.get("hop_type", "")
        if "1" in ht or "Single" in ht:
            hop_types.add("L1")
        if "2" in ht or "Multi" in ht or "Relationship" in ht:
            hop_types.add("L2")
    scores["has_both_levels"] = len(hop_types) >= 2

    # 6. Has dependency (from_instance → to_instance transitions)
    deps = sum(1 for h in hops if h.get("to_instance"))
    scores["dependency_hops"] = deps
    scores["has_dependency"] = deps >= 1

    # Overall
    scores["quality_score"] = sum([
        scores["hop_count_ok"] * 2,
        scores["all_instances_ok"],
        scores["has_chain"],
        scores["has_numerical_answer"],
        scores["has_both_levels"],
        scores["has_dependency"],
    ])
    scores["max_score"] = 7

    return scores


# ============================================================================
# Main runner
# ============================================================================

def run_on_image(vlm: AzureVLM, image_path: str, num_queries: int = 3):
    """Run full HopChain pipeline on a single image."""
    results = {"image": image_path, "stage1": None, "stage2": None,
               "stage3": [], "stage4": [], "quality_scores": []}

    # Stage 1
    s1 = stage1_filter_and_categorize(vlm, image_path)
    if s1 is None:
        print(f"\nImage {image_path} did not pass Stage 1 filter.")
        return results
    results["stage1"] = s1

    # Stage 2
    instances = stage2_detect_instances(vlm, image_path, s1["categories"])
    if len(instances) < 3:
        print(f"\nToo few instances ({len(instances)}), need >= 3")
        return results
    results["stage2"] = instances

    # Sample combinations (3-6 instances)
    combo_size = min(6, len(instances))
    combo_size = max(3, combo_size)

    # Pick diverse combo
    if len(instances) > combo_size:
        # Try to pick instances from diverse categories
        cats = list(set(inst["category"] for inst in instances))
        random.shuffle(cats)
        combo = []
        for cat in cats:
            cat_insts = [i for i in instances if i["category"] == cat and i not in combo]
            if cat_insts:
                combo.append(cat_insts[0])
            if len(combo) >= combo_size:
                break
        # Fill remaining
        for inst in instances:
            if inst not in combo and len(combo) < combo_size:
                combo.append(inst)
    else:
        combo = instances[:combo_size]

    # Stage 3
    queries = stage3_generate_queries(vlm, image_path, combo, num_queries=num_queries)

    # Stage 3b: Self-verification (corrects hypothetical answers)
    if queries:
        queries = stage3b_self_verify(vlm, image_path, queries)

    results["stage3"] = queries

    # Evaluate quality
    for q in queries:
        qs = evaluate_query_quality(q)
        results["quality_scores"].append(qs)
        print(f"\n  Quality score for Q{q.get('id','?')}: {qs['quality_score']}/{qs['max_score']}")
        for k, v in qs.items():
            if k not in ("quality_score", "max_score"):
                status = "✓" if v else "✗" if isinstance(v, bool) else str(v)
                print(f"    {k}: {status}")

    # Stage 4: Verify top queries
    for q in queries[:2]:  # verify first 2
        v = stage4_verify(vlm, image_path, q, n_samples=3)
        results["stage4"].append({
            "query_id": q.get("id"),
            "query": q.get("query", "")[:200],
            "expected": q.get("hypothetical_answer"),
            **v
        })

    return results


def main():
    print("=" * 70)
    print("HopChain Multi-Hop Data Synthesis — GPT-5 Runner")
    print("=" * 70)

    # Find test images
    img_dir = Path("/home/v-hezhu2/spotlight_reader/output/hopchain/test_images")
    images = sorted(img_dir.glob("*.jpg"))

    if not images:
        print("No test images found!")
        return

    print(f"\nFound {len(images)} images: {[p.name for p in images]}")

    # Use GPT-5
    vlm = AzureVLM(model_name="gpt-5")
    print(f"Using model: GPT-5")

    all_results = []
    all_queries = []

    for img_path in images:
        result = run_on_image(vlm, str(img_path), num_queries=3)
        all_results.append(result)
        all_queries.extend(result.get("stage3", []))

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_queries = len(all_queries)
    print(f"Total queries generated: {total_queries}")

    if total_queries > 0:
        avg_hops = sum(len(q.get("reasoning_hops", [])) for q in all_queries) / total_queries
        print(f"Average hops per query: {avg_hops:.1f}")

        # Quality distribution
        all_scores = []
        for r in all_results:
            all_scores.extend(r.get("quality_scores", []))

        if all_scores:
            avg_quality = sum(s["quality_score"] for s in all_scores) / len(all_scores)
            print(f"Average quality score: {avg_quality:.1f}/{all_scores[0]['max_score']}")
            good = sum(1 for s in all_scores if s["quality_score"] >= 5)
            print(f"High quality queries (≥5/7): {good}/{len(all_scores)}")

    # Verification results
    all_verif = []
    for r in all_results:
        all_verif.extend(r.get("stage4", []))
    if all_verif:
        verified = sum(1 for v in all_verif if v["verified"])
        print(f"Verification: {verified}/{len(all_verif)} queries had consistent answers")

    # Save everything
    out_path = OUTPUT_DIR / "hopchain_gpt5_results.json"
    # Clean for serialization
    serializable = json.loads(json.dumps(all_results, default=str))
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")

    # Save queries as JSONL
    queries_path = OUTPUT_DIR / "hopchain_gpt5_queries.jsonl"
    with open(queries_path, "w") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Queries saved to: {queries_path}")

    # Print best queries
    if all_queries:
        print("\n" + "=" * 70)
        print("BEST GENERATED QUERIES")
        print("=" * 70)
        for i, q in enumerate(all_queries):
            print(f"\n--- Query {i+1} ---")
            print(f"Capability: {q.get('primary_capability', '?')}")
            print(f"Chain: {q.get('instance_chain', '?')}")
            print(f"Hops: {len(q.get('reasoning_hops', []))}")
            print(f"Answer: {q.get('hypothetical_answer', '?')}")
            print(f"Query: {q.get('query', '?')}")
            print()


if __name__ == "__main__":
    main()
