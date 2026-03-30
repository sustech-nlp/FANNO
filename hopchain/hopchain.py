#!/usr/bin/env python3
"""
HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning
==============================================================================

Faithful reproduction of the HopChain pipeline from:
  "HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning"
  Shenzhi Wang, Shixuan Liu, Jing Zhou, Chang Gao et al. (Qwen Team + Tsinghua LeapLab)

Pipeline overview (4 stages):
  Stage 1: Category Identification — VLM extracts semantic categories from image
  Stage 2: Instance Segmentation  — SAM3 segments each category into instances
  Stage 3: Multi-Hop Query Gen    — VLM generates multi-hop reasoning queries
  Stage 4: Annotation & Calibration — Human annotation + difficulty filtering

This script implements Stages 1-3 fully automated, with Stage 4 partially
automated (using model-based verification instead of human annotators).

Dependencies:
  pip install openai Pillow segment-anything-2 transformers torch
  # For SAM3, follow: https://github.com/facebookresearch/sam2
  # For VLM, uses OpenAI-compatible API (Azure OpenAI / vLLM / local)
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HopChainConfig:
    """Full pipeline configuration."""

    # ----- API -----
    api_base: str = ""           # OpenAI-compatible base URL
    api_key: str = ""            # API key
    vlm_model: str = "gpt-4o"   # VLM model name (ideally Qwen3-VL-235B)
    weak_model: str = "gpt-4o-mini"  # Weaker model for difficulty calibration
    max_workers: int = 8

    # ----- SAM -----
    sam_checkpoint: str = ""     # Path to SAM2 checkpoint
    sam_model_type: str = "vit_h"  # SAM model type

    # ----- Pipeline -----
    min_complexity_score: int = 5     # Minimum image complexity score (1-10)
    min_quality: str = "Medium"       # Minimum quality rating
    combo_size_min: int = 3           # Min instances per combination
    combo_size_max: int = 6           # Max instances per combination
    max_combos_per_image: int = 5     # Max combinations to sample per image
    queries_per_combo: int = 4        # Queries to generate per combination
    target_hops: str = "4-6"          # Target hop count range
    annotation_agreement: int = 4     # Required annotator agreement (out of 4)
    weak_model_samples: int = 8       # Samples for difficulty calibration
    weak_model_threshold: float = 1.0 # Remove if weak model accuracy >= this

    # ----- Output -----
    output_dir: str = "./hopchain_output"

    @classmethod
    def from_file(cls, path: str) -> "HopChainConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Prompt Templates (verbatim from paper's Appendix)
# ============================================================================

IMAGE_SELECTION_PROMPT = """\
# Role and Goal

You are a professional AI Image Analyst. Your primary goal is to evaluate the complexity of a given image from the perspective of a standard computer vision model. This analysis will identify images that pose a challenge to AI perception, especially in tasks like object detection, counting, and recognition.

# Core Task

For the image I provide, you must:

1.  Evaluate its overall perceptual complexity on a scale of **1 to 10**.
2.  Assess the image quality (High / Medium / Low).
3.  Identify the specific objects or areas that contribute to the complexity.
4.  Output the results as a **single JSON object** as required, with no explanatory text outside of the JSON.

# Key Definitions

## 1. Image Complexity Factors (What makes an image complex for AI?)

*   **Occlusion**: Objects are partially hidden by other objects. The more a key object is occluded, the higher the complexity.
*   **Object Count & Density**: The image contains a large number of distinct objects, especially of the same category (e.g., a crowd of people, a fleet of cars), making counting and individual identification difficult.
*   **Unusual Pose/Angle**: Objects are shown from unconventional viewpoints (e.g., a person doing a handstand, a car seen directly from above).
*   **Complex Interaction**: Objects interact in a way that blurs their individual boundaries (e.g., people hugging, tangled wires, a pile of clothes).
*   **Fine-grained Recognition**: Requires distinguishing between very similar sub-categories (e.g., telling two different bird species apart, or different models of the same car brand).
*   **Challenging Lighting/Shadows**: Harsh shadows, reflections, or poor lighting obscure the shape and details of objects.

## 2. Image Quality Guidelines (How to handle low-quality or unusable images)

Your analysis must focus on clear, perceptible objects.

*   **High/Medium Quality**: The image is generally clear, and its main subjects are recognizable, even if the scene is complex.
*   **Low Quality**: An image should be rated "Low Quality" when it is unusable for precise perceptual analysis due to any of the following reasons:
    *   **Technical Flaws:** The image is too blurry, noisy, severely overexposed, or underexposed, making objects unrecognizable.
    *   **Annotation Impracticality:** **Even if the image itself is clear, the extreme number and density of objects make meaningful, verifiable annotation impossible (e.g., it's impossible to draw a bounding box for or accurately count every single object). Such images lack a "Ground Truth" for model evaluation.**

*   **Crucial**:
    *   **Case 1 (Technical Flaws):** If a scene is chaotic due to blur or lighting issues, making it impossible to distinguish individual objects, its quality should be rated **"Low"** with a **low complexity score (1-3)**.
    *   **Case 2 (Annotation Impracticality):** **If a scene contains a vast, uncountable collection of objects (e.g., a massive but clear crowd where individuals cannot be boxed or counted; a bookshelf or rack packed with countless small items), its quality should also be rated "Low," even if the image is technically clear.**
    *   **Conclusion:** **For both of the "Low Quality" cases described above, the perceptual complexity score should be low (1-3)**, as they are not good test cases for perception but rather issues of data usability or quality itself.

# Output Format Requirements

The output must be a structured **JSON object** (with no additional text). The structure is as follows:

```json
{
  "overall_complexity_score": <Integer, 1-10>,
  "overall_quality_rating": "<High | Medium | Low>",
  "complexity_analysis": "<String, a brief explanation for the score, referencing the complexity factors above>",
  "complex_objects": [
    {
      "object_name": "<Description of the specific object>",
      "generalized_name": "<General category like Person, Animal, Vehicle, Plant, etc.>",
      "reason_for_complexity": [
        "<Select from the list of complexity factors, e.g., 'Occlusion', 'Fine-grained Recognition'>"
      ]
    }
  ]
}
```

# Final Instruction

After I provide an image, strictly follow all the rules above, perform the analysis based on the actual image content, and output a JSON object that strictly adheres to the specified format."""

CATEGORY_IDENTIFICATION_PROMPT = """\
You are a precise visual object detector. Given an image, identify ALL distinct semantic categories of objects visible in the image.

For each category, provide:
1. The category name (e.g., "car", "person", "traffic_sign", "tree")
2. An estimated count of instances of that category

Output a JSON object with this format:
```json
{
  "categories": [
    {"name": "car", "estimated_count": 3},
    {"name": "person", "estimated_count": 5},
    {"name": "traffic_sign", "estimated_count": 2}
  ]
}
```

Be thorough — list every distinct object category you can identify, even small or partially occluded objects. Only list categories where you can clearly identify at least one instance."""

MULTIHOP_QUERY_PROMPT = """\
#### **Role & Goal**

You are a top-tier AI multimodal capability evaluation expert. Your task is to design a set of **{num_queries_word}** independent, high-difficulty **multi-hop vision-language reasoning** sub-queries based on a **complex image**. These sub-queries will serve as "capability modules" to evaluate a model's comprehensive visual intelligence.

#### **Understanding VLM Perception Capabilities**

VLM (Vision-Language Model) perception capabilities can be categorized into three levels:

1. **Single-Object Perception (Level 1):** Perceiving attributes of a single object (e.g., color, shape, size, text content, position, category).
2. **Multi-Object & Relationship Perception (Level 2):** Perceiving multiple objects and their relationships (e.g., spatial relationships like "A is to the left of B", comparative relationships like "A is larger than B", counting objects that satisfy certain conditions).
3. **Multi-Hop Reasoning (Level 3):** Chaining multiple perception steps together. **Multi-hop can occur in TWO dimensions:**
   - **Perception-Level Hops:** Moving between Level 1 and Level 2 perception tasks
   - **Instance-to-Instance Hops:** Reasoning from instance_1 -> instance_2 -> instance_3 -> ... -> instance_n, where **finding the next instance DEPENDS ON the previous instance**. The key is that instance_N+1 can ONLY be located by using information from or relationships with instance_N.

**YOUR TASK: Design queries that are strictly Level 3 (Multi-Hop Reasoning).** Each query must chain together multiple perception steps across BOTH dimensions - hopping between different perception levels AND hopping across different instances with DEPENDENCY relationships.

#### **Types of Multi-Hop Reasoning (Design queries using ALL types)**

**Type A - Instance Dependency Chain (MOST IMPORTANT):**
- The NEXT instance can ONLY be found by using its relationship with the PREVIOUS instance
- Each hop must establish a dependency: "the X that is [relationship] to the previous instance"
- **BAD (no dependency):** "Find the largest car, then find the tallest tree" (tree doesn't depend on car)
- **GOOD (with dependency):** "Find the largest car, then find the tree closest to THIS car" (tree depends on which car was found)

**Type B - Perception Level Hop:**
- Level 1 (single object) -> Level 2 (multi-object relationship) -> Level 1 -> Level 2 -> ...
- Each level transition should maintain the instance dependency chain

**Type C - Combined with Strong Dependencies (PREFERRED):**
- Combine instance dependency chains with perception level changes for maximum complexity

**You are provided with MULTIPLE images:**
1. **Image 1 (Original):** The original image containing all visual information. **This is the ONLY image that will be available when answering the queries.**
2. **Instance Patch Images (Image 2, 3, 4, ...):** Cropped patches of each individual instance from the original image.

**CRITICAL: You are designing queries for a SPECIFIC COMBINATION of objects. The following object instances are the ONLY instances you should consider for this task:**

{object_list}

**[WARNING] MANDATORY REQUIREMENT - INVOLVE ALL INSTANCES:**
- Each sub-query MUST involve AS MANY instances as possible from the combination list above.
- If the combination has N instances, each query should involve at least (N-1) instances, preferably all N.
- DO NOT design queries that only involve 2-3 objects when 5-6 objects are available.

**CRITICAL CONSTRAINT - NO PATCH/BOX REFERENCES & UNAMBIGUOUS INSTANCE REFERENCES:**
- NEVER mention detection borders, bounding boxes, patches, or coordinates in the query text.
- Describe objects using ONLY spatial positions, contextual relationships, functional descriptions, and visual attributes.
- Each instance reference MUST uniquely identify exactly ONE instance in the original image.

#### **Core Principles**

1. **Independence:** Each query must be completely self-contained.
2. **Multi-Hop Structure:** Each query MUST have {target_hop_count_info} hops with logical dependency chains.
3. **Maximum Instance Coverage:** Each query MUST involve ALL instances from the provided combination.
4. **High Perception & Logic Difficulty:** Non-trivial perception at each hop, with conditional logic, set operations, or numerical computations.
5. **Deterministic Numerical Answer:** The final answer must be a specific, unambiguous number.
6. **Visual Grounding:** All information must be obtainable only from the image.
7. **Unambiguous Phrasing:** Two people with good vision should arrive at the EXACT SAME answer.
8. **Balanced Conditional Outcomes:** For if-then-else conditions, ensure roughly half evaluate to "No".

#### **Output Format**

Output a single JSON object:

```json
{{
  "sub_queries": [
    {{
      "id": 1,
      "primary_capability": "Spatial Reasoning + Counting + Conditional Logic",
      "involved_objects": ["instance_1", "instance_2", ...],
      "query": "Your complex multi-hop query here.",
      "instance_chain": "A -> B -> C -> D -> E",
      "reasoning_hops": [
        {{
          "hop_number": 1,
          "hop_type": "Level 1 (Single-Object)",
          "from_instance": "instance_1",
          "to_instance": null,
          "description": "Extract information from instance_1",
          "objects_involved": ["instance_1"],
          "output": "value extracted"
        }}
      ],
      "hypothetical_answer": "42",
      "design_rationale": "Explain the chain and difficulty"
    }}
  ]
}}
```"""


# ============================================================================
# Utility Functions
# ============================================================================

def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image_base64(pil_image) -> str:
    """Encode a PIL Image to base64 string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from a model response, handling markdown code blocks."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from code block
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    # Last resort: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def number_to_word(n: int) -> str:
    """Convert integer to English word for prompt template."""
    words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    }
    return words.get(n, str(n))


# ============================================================================
# VLM Client (OpenAI-compatible)
# ============================================================================

class VLMClient:
    """Wrapper for OpenAI-compatible VLM API calls."""

    def __init__(self, config: HopChainConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")

            self._client = OpenAI(
                base_url=self.config.api_base or None,
                api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY", ""),
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the text response."""
        model = model or self.config.vlm_model
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def chat_with_image(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        image_b64s: Optional[List[str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a vision chat request with one or more images."""
        content: List[Dict[str, Any]] = []

        # Add images
        sources = []
        if image_paths:
            sources.extend(
                ("data:image/png;base64," + encode_image_base64(p))
                if not p.startswith("data:")
                else p
                for p in image_paths
            )
        if image_b64s:
            sources.extend(f"data:image/png;base64,{b}" for b in image_b64s)

        for url in sources:
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": "high"},
            })

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)


# ============================================================================
# Stage 1: Image Selection & Category Identification
# ============================================================================

@dataclass
class ImageAnalysis:
    """Result of image selection and category identification."""
    image_path: str
    complexity_score: int = 0
    quality_rating: str = "Low"
    complexity_analysis: str = ""
    complex_objects: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[Dict[str, Any]] = field(default_factory=list)
    passed_filter: bool = False


def stage1_select_and_identify(
    vlm: VLMClient,
    image_path: str,
    config: HopChainConfig,
) -> ImageAnalysis:
    """Stage 1: Evaluate image complexity and identify semantic categories.

    Combines the image selection (filtering) and category identification steps.
    In the paper, these are done by Qwen3-VL-235B-A22B-Thinking.
    """
    result = ImageAnalysis(image_path=image_path)

    # --- Step 1a: Image complexity evaluation ---
    logger.info(f"[Stage 1a] Evaluating complexity: {image_path}")
    selection_response = vlm.chat_with_image(
        prompt=IMAGE_SELECTION_PROMPT,
        image_paths=[image_path],
        temperature=0.3,
        max_tokens=2048,
    )

    selection_data = parse_json_response(selection_response)
    if selection_data is None:
        logger.warning(f"Failed to parse image selection response for {image_path}")
        return result

    result.complexity_score = selection_data.get("overall_complexity_score", 0)
    result.quality_rating = selection_data.get("overall_quality_rating", "Low")
    result.complexity_analysis = selection_data.get("complexity_analysis", "")
    result.complex_objects = selection_data.get("complex_objects", [])

    # Filter check
    quality_rank = {"High": 3, "Medium": 2, "Low": 1}
    min_rank = quality_rank.get(config.min_quality, 2)
    actual_rank = quality_rank.get(result.quality_rating, 0)

    if result.complexity_score < config.min_complexity_score or actual_rank < min_rank:
        logger.info(
            f"  Image filtered out: score={result.complexity_score}, "
            f"quality={result.quality_rating}"
        )
        return result

    result.passed_filter = True

    # --- Step 1b: Category identification ---
    logger.info(f"[Stage 1b] Identifying categories: {image_path}")
    cat_response = vlm.chat_with_image(
        prompt=CATEGORY_IDENTIFICATION_PROMPT,
        image_paths=[image_path],
        temperature=0.3,
        max_tokens=2048,
    )

    cat_data = parse_json_response(cat_response)
    if cat_data and "categories" in cat_data:
        result.categories = cat_data["categories"]
        logger.info(
            f"  Found {len(result.categories)} categories: "
            f"{[c['name'] for c in result.categories]}"
        )
    else:
        logger.warning(f"Failed to parse categories for {image_path}")
        result.passed_filter = False

    return result


# ============================================================================
# Stage 2: Instance Segmentation (SAM3 or fallback)
# ============================================================================

@dataclass
class Instance:
    """A detected instance with spatial localization."""
    instance_id: str          # e.g., "car_1"
    category: str             # e.g., "car"
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    bbox_normalized: Tuple[float, float, float, float]  # in 0-1000 range
    mask: Optional[Any] = None  # binary mask (numpy array)
    patch_b64: str = ""       # base64 encoded cropped patch

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "category": self.category,
            "bbox": list(self.bbox),
            "bbox_normalized": list(self.bbox_normalized),
        }


def _try_sam_segmentation(
    image_path: str,
    categories: List[Dict[str, Any]],
    config: HopChainConfig,
) -> Optional[List[Instance]]:
    """Try using SAM2/SAM3 for instance segmentation."""
    try:
        import torch
        import numpy as np
        from PIL import Image as PILImage

        # Try SAM2 import
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2
        except ImportError:
            try:
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            except ImportError:
                return None

        if not config.sam_checkpoint or not Path(config.sam_checkpoint).exists():
            return None

        logger.info("[Stage 2] Using SAM for instance segmentation")
        img = PILImage.open(image_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img)

        # Build SAM model and generate masks
        sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(img_np)

        instances = []
        cat_counters: Dict[str, int] = {}

        for mask_data in masks:
            bbox = mask_data["bbox"]  # [x, y, w_box, h_box]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = x1 + bbox[2], y1 + bbox[3]

            # Assign to nearest category (simplified — in practice use
            # a classifier or the VLM to label each mask)
            cat_name = "object"
            for cat in categories:
                cat_name = cat["name"]
                break

            cat_counters[cat_name] = cat_counters.get(cat_name, 0) + 1
            inst_id = f"{cat_name}_{cat_counters[cat_name]}"

            # Crop patch
            patch = img.crop((x1, y1, x2, y2))
            patch_b64 = encode_pil_image_base64(patch)

            instances.append(Instance(
                instance_id=inst_id,
                category=cat_name,
                bbox=(x1, y1, x2, y2),
                bbox_normalized=(
                    round(x1 / w * 1000),
                    round(y1 / h * 1000),
                    round(x2 / w * 1000),
                    round(y2 / h * 1000),
                ),
                mask=mask_data["segmentation"],
                patch_b64=patch_b64,
            ))

        return instances if instances else None

    except Exception as e:
        logger.warning(f"SAM segmentation failed: {e}")
        return None


def _vlm_fallback_segmentation(
    vlm: VLMClient,
    image_path: str,
    categories: List[Dict[str, Any]],
) -> List[Instance]:
    """Fallback: use VLM to estimate bounding boxes for instances.

    When SAM is not available, we ask the VLM to provide approximate
    bounding box coordinates for each instance. Less precise but functional.
    """
    from PIL import Image as PILImage

    logger.info("[Stage 2] Using VLM fallback for instance localization")

    cat_names = [c["name"] for c in categories]
    prompt = f"""\
For the image provided, locate each individual instance of these categories: {cat_names}

For each instance, provide:
- category: the object category
- bbox: [x1, y1, x2, y2] as percentages of image width/height (0-100)
- description: brief distinguishing description

Output JSON:
```json
{{
  "instances": [
    {{"category": "car", "bbox": [10, 20, 40, 60], "description": "red sedan in center"}},
    {{"category": "person", "bbox": [50, 30, 65, 80], "description": "man in blue shirt"}}
  ]
}}
```

List ALL distinct instances you can identify. Be precise with bounding boxes."""

    response = vlm.chat_with_image(
        prompt=prompt,
        image_paths=[image_path],
        temperature=0.3,
        max_tokens=4096,
    )

    data = parse_json_response(response)
    if data is None or "instances" not in data:
        logger.warning("VLM fallback segmentation failed to parse")
        return []

    img = PILImage.open(image_path).convert("RGB")
    w, h = img.size

    instances = []
    cat_counters: Dict[str, int] = {}

    for item in data["instances"]:
        cat = item.get("category", "object")
        bbox_pct = item.get("bbox", [0, 0, 100, 100])

        # Convert percentage to pixels
        x1 = int(bbox_pct[0] / 100 * w)
        y1 = int(bbox_pct[1] / 100 * h)
        x2 = int(bbox_pct[2] / 100 * w)
        y2 = int(bbox_pct[3] / 100 * h)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        cat_counters[cat] = cat_counters.get(cat, 0) + 1
        inst_id = f"{cat}_{cat_counters[cat]}"

        # Crop patch
        patch = img.crop((x1, y1, x2, y2))
        patch_b64 = encode_pil_image_base64(patch)

        instances.append(Instance(
            instance_id=inst_id,
            category=cat,
            bbox=(x1, y1, x2, y2),
            bbox_normalized=(
                round(x1 / w * 1000),
                round(y1 / h * 1000),
                round(x2 / w * 1000),
                round(y2 / h * 1000),
            ),
            patch_b64=patch_b64,
        ))

    logger.info(f"  VLM fallback found {len(instances)} instances")
    return instances


def stage2_instance_segmentation(
    vlm: VLMClient,
    image_path: str,
    categories: List[Dict[str, Any]],
    config: HopChainConfig,
) -> List[Instance]:
    """Stage 2: Instance segmentation using SAM3 (or VLM fallback).

    In the paper, SAM3 is used to generate segmentation masks and bounding
    boxes for each identified category. We try SAM first, then fall back
    to VLM-based localization.
    """
    # Try SAM first
    instances = _try_sam_segmentation(image_path, categories, config)
    if instances is not None:
        logger.info(f"  SAM found {len(instances)} instances")
        return instances

    # Fallback to VLM
    return _vlm_fallback_segmentation(vlm, image_path, categories)


# ============================================================================
# Stage 3: Multi-Hop Query Generation
# ============================================================================

@dataclass
class MultiHopQuery:
    """A generated multi-hop reasoning query."""
    query_id: int
    query_text: str
    primary_capability: str
    involved_objects: List[str]
    instance_chain: str
    reasoning_hops: List[Dict[str, Any]]
    hypothetical_answer: str
    design_rationale: str
    image_path: str = ""
    combination: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _build_object_list_text(instances: List[Instance]) -> str:
    """Build the {object_list} text for the multi-hop query prompt.

    Each instance is described with its ID and bounding box coordinates
    (in 0-1000 range), matching the paper's format.
    """
    lines = []
    for i, inst in enumerate(instances, 1):
        bn = inst.bbox_normalized
        lines.append(
            f"- **{inst.instance_id}** (category: {inst.category}, "
            f"bbox: [{bn[0]}, {bn[1]}, {bn[2]}, {bn[3]}]): "
            f"See Image {i+1} for the cropped patch of this instance."
        )
    return "\n".join(lines)


def stage3_generate_queries(
    vlm: VLMClient,
    image_path: str,
    combination: List[Instance],
    config: HopChainConfig,
) -> List[MultiHopQuery]:
    """Stage 3: Generate multi-hop queries for a combination of instances.

    In the paper, Qwen3-VL-235B receives the original image plus cropped
    patches of each instance. The prompt enforces strict structural
    constraints on hop types, dependency chains, and answer format.
    """
    num_queries = config.queries_per_combo
    object_list_text = _build_object_list_text(combination)

    prompt = MULTIHOP_QUERY_PROMPT.format(
        num_queries_word=number_to_word(num_queries),
        num_queries=num_queries,
        object_list=object_list_text,
        target_hop_count_info=config.target_hops,
    )

    # Build image list: original + patches
    image_b64s = [encode_image_base64(image_path)]
    for inst in combination:
        if inst.patch_b64:
            image_b64s.append(inst.patch_b64)

    logger.info(
        f"[Stage 3] Generating {num_queries} queries for combination: "
        f"{[inst.instance_id for inst in combination]}"
    )

    response = vlm.chat_with_image(
        prompt=prompt,
        image_b64s=image_b64s,
        temperature=0.7,
        max_tokens=8192,
    )

    data = parse_json_response(response)
    if data is None or "sub_queries" not in data:
        logger.warning("Failed to parse multi-hop query response")
        return []

    queries = []
    for sq in data["sub_queries"]:
        queries.append(MultiHopQuery(
            query_id=sq.get("id", 0),
            query_text=sq.get("query", ""),
            primary_capability=sq.get("primary_capability", ""),
            involved_objects=sq.get("involved_objects", []),
            instance_chain=sq.get("instance_chain", ""),
            reasoning_hops=sq.get("reasoning_hops", []),
            hypothetical_answer=str(sq.get("hypothetical_answer", "")),
            design_rationale=sq.get("design_rationale", ""),
            image_path=image_path,
            combination=[inst.instance_id for inst in combination],
        ))

    logger.info(f"  Generated {len(queries)} multi-hop queries")
    return queries


# ============================================================================
# Stage 4: Ground-Truth Annotation & Difficulty Calibration
# ============================================================================

def stage4_annotate_and_calibrate(
    vlm: VLMClient,
    queries: List[MultiHopQuery],
    config: HopChainConfig,
) -> List[Dict[str, Any]]:
    """Stage 4: Verify answers and calibrate difficulty.

    In the paper, 4 human annotators independently solve each query and
    only unanimous answers are kept. Here we simulate this with multiple
    VLM samples (using the strong model as "annotators").

    Then, a weaker model samples responses to calibrate difficulty —
    queries that are too easy (100% accuracy on weak model) are removed.
    """
    verified_queries = []

    for query in queries:
        logger.info(f"[Stage 4] Verifying query: {query.query_text[:80]}...")

        # --- Step 4a: Multi-sample annotation ---
        answer_prompt = (
            f"Look at the image carefully and answer this question. "
            f"Show your step-by-step reasoning, then provide your final numerical answer.\n\n"
            f"Question: {query.query_text}\n\n"
            f"Your final answer must be a single number. End your response with: "
            f"ANSWER: <number>"
        )

        annotator_answers = []
        for i in range(config.annotation_agreement):
            response = vlm.chat_with_image(
                prompt=answer_prompt,
                image_paths=[query.image_path],
                temperature=0.5,  # Some variation for diverse "annotators"
                max_tokens=2048,
            )

            # Extract numerical answer
            answer = _extract_numerical_answer(response)
            annotator_answers.append(answer)

        # Check agreement
        if None in annotator_answers:
            logger.info(f"  Skipped: some annotators failed to produce a number")
            continue

        if len(set(annotator_answers)) != 1:
            logger.info(
                f"  Skipped: annotators disagreed {annotator_answers}"
            )
            continue

        ground_truth = annotator_answers[0]
        logger.info(f"  Annotators agreed: answer = {ground_truth}")

        # --- Step 4b: Difficulty calibration with weak model ---
        weak_correct = 0
        for i in range(config.weak_model_samples):
            response = vlm.chat_with_image(
                prompt=answer_prompt,
                image_paths=[query.image_path],
                model=config.weak_model,
                temperature=0.8,
                max_tokens=2048,
            )
            weak_answer = _extract_numerical_answer(response)
            if weak_answer == ground_truth:
                weak_correct += 1

        weak_accuracy = weak_correct / config.weak_model_samples
        logger.info(f"  Weak model accuracy: {weak_accuracy:.1%}")

        if weak_accuracy >= config.weak_model_threshold:
            logger.info(f"  Removed: too easy (weak model {weak_accuracy:.0%} correct)")
            continue

        # Passed all filters!
        verified_queries.append({
            "image": query.image_path,
            "query": query.query_text,
            "answer": ground_truth,
            "primary_capability": query.primary_capability,
            "involved_objects": query.involved_objects,
            "instance_chain": query.instance_chain,
            "reasoning_hops": query.reasoning_hops,
            "design_rationale": query.design_rationale,
            "combination": query.combination,
            "weak_model_accuracy": weak_accuracy,
            "num_hops": len(query.reasoning_hops),
        })

    logger.info(f"[Stage 4] Verified {len(verified_queries)}/{len(queries)} queries")
    return verified_queries


def _extract_numerical_answer(text: str) -> Optional[int]:
    """Extract the final numerical answer from model response."""
    # Try ANSWER: pattern
    match = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        try:
            return int(float(match.group(1)))
        except ValueError:
            pass

    # Try last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return int(float(numbers[-1]))
        except ValueError:
            pass

    return None


# ============================================================================
# Full Pipeline
# ============================================================================

def sample_combinations(
    instances: List[Instance],
    config: HopChainConfig,
) -> List[List[Instance]]:
    """Sample combinations of instances for multi-hop query generation.

    Each combination has 3-6 instances (configurable). We sample up to
    max_combos_per_image combinations per image.
    """
    combos = []
    n = len(instances)

    for size in range(config.combo_size_min, min(config.combo_size_max + 1, n + 1)):
        all_combos = list(itertools.combinations(instances, size))
        random.shuffle(all_combos)

        # Prefer combinations with diverse categories
        scored = []
        for combo in all_combos:
            cats = set(inst.category for inst in combo)
            scored.append((len(cats), combo))
        scored.sort(key=lambda x: -x[0])

        for _, combo in scored:
            combos.append(list(combo))
            if len(combos) >= config.max_combos_per_image:
                return combos

    return combos[:config.max_combos_per_image]


def run_pipeline(
    image_paths: List[str],
    config: HopChainConfig,
    skip_stage4: bool = False,
) -> List[Dict[str, Any]]:
    """Run the full HopChain pipeline on a set of images.

    Args:
        image_paths: List of image file paths to process.
        config: Pipeline configuration.
        skip_stage4: If True, skip annotation & calibration (for speed).

    Returns:
        List of verified multi-hop query dictionaries.
    """
    vlm = VLMClient(config)
    all_queries: List[MultiHopQuery] = []
    all_verified: List[Dict[str, Any]] = []

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {img_path}")
        logger.info(f"{'='*60}")

        # Stage 1: Image Selection & Category Identification
        analysis = stage1_select_and_identify(vlm, img_path, config)

        if not analysis.passed_filter:
            logger.info(f"Image did not pass filter, skipping.")
            continue

        # Save Stage 1 results
        stage1_path = output_dir / "stage1_results.jsonl"
        with open(stage1_path, "a") as f:
            f.write(json.dumps({
                "image": img_path,
                "complexity_score": analysis.complexity_score,
                "quality": analysis.quality_rating,
                "categories": analysis.categories,
            }, ensure_ascii=False) + "\n")

        # Stage 2: Instance Segmentation
        instances = stage2_instance_segmentation(
            vlm, img_path, analysis.categories, config
        )

        if len(instances) < config.combo_size_min:
            logger.info(
                f"Too few instances ({len(instances)}), need >= {config.combo_size_min}"
            )
            continue

        # Save Stage 2 results
        stage2_path = output_dir / "stage2_results.jsonl"
        with open(stage2_path, "a") as f:
            f.write(json.dumps({
                "image": img_path,
                "instances": [inst.to_dict() for inst in instances],
            }, ensure_ascii=False) + "\n")

        # Stage 3: Multi-Hop Query Generation
        combinations = sample_combinations(instances, config)
        logger.info(f"Sampled {len(combinations)} instance combinations")

        for combo in combinations:
            queries = stage3_generate_queries(vlm, img_path, combo, config)
            all_queries.extend(queries)

            # Save Stage 3 results incrementally
            stage3_path = output_dir / "stage3_queries.jsonl"
            with open(stage3_path, "a") as f:
                for q in queries:
                    f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Total queries so far: {len(all_queries)}")

    # Stage 4: Ground-Truth Annotation & Difficulty Calibration
    if not skip_stage4 and all_queries:
        all_verified = stage4_annotate_and_calibrate(vlm, all_queries, config)

        # Save final verified dataset
        final_path = output_dir / "hopchain_verified.jsonl"
        with open(final_path, "w") as f:
            for item in all_verified:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_verified)} verified queries to {final_path}")
    elif all_queries:
        # Save unverified queries as the output
        final_path = output_dir / "hopchain_unverified.jsonl"
        with open(final_path, "w") as f:
            for q in all_queries:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_queries)} unverified queries to {final_path}")

    # Summary
    summary = {
        "total_images": len(image_paths),
        "images_passed_filter": sum(
            1 for _ in all_queries  # rough count
        ),
        "total_queries_generated": len(all_queries),
        "total_queries_verified": len(all_verified),
        "config": asdict(config),
    }
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return all_verified if all_verified else [q.to_dict() for q in all_queries]


# ============================================================================
# RLVR Training Data Formatting
# ============================================================================

def format_for_rlvr(
    verified_queries: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Format verified queries into RLVR training format.

    Each item becomes:
    {
        "image": <path>,
        "query": <multi-hop question>,
        "answer": <ground-truth numerical answer>,
        "reward_type": "exact_match",
    }

    The reward function is:
        R(output, answer) = 1.0 if is_equivalent(output, answer) else 0.0
    """
    rlvr_data = []
    for item in verified_queries:
        rlvr_data.append({
            "image": item["image"],
            "query": item["query"],
            "answer": item["answer"],
            "reward_type": "exact_match",
            "num_hops": item.get("num_hops", 0),
            "primary_capability": item.get("primary_capability", ""),
        })

    with open(output_path, "w") as f:
        for item in rlvr_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Formatted {len(rlvr_data)} RLVR training samples to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HopChain: Multi-Hop Data Synthesis for VLM Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Process images in a directory
  python hopchain.py --image-dir ./images --output-dir ./output

  # Process a single image (skip Stage 4 for speed)
  python hopchain.py --images img1.jpg img2.jpg --skip-stage4

  # Use custom API
  python hopchain.py --api-base http://localhost:8000/v1 --model qwen3-vl \\
      --image-dir ./images

  # Load config from file
  python hopchain.py --config hopchain_config.json
""",
    )

    # Input
    parser.add_argument("--images", nargs="+", help="Image file paths")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--config", type=str, help="JSON config file")

    # API
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--weak-model", type=str, default="gpt-4o-mini")

    # Pipeline
    parser.add_argument("--min-complexity", type=int, default=5)
    parser.add_argument("--combo-size-min", type=int, default=3)
    parser.add_argument("--combo-size-max", type=int, default=6)
    parser.add_argument("--max-combos", type=int, default=5)
    parser.add_argument("--queries-per-combo", type=int, default=4)
    parser.add_argument("--target-hops", type=str, default="4-6")
    parser.add_argument("--skip-stage4", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="./hopchain_output")

    args = parser.parse_args()

    # Build config
    if args.config:
        config = HopChainConfig.from_file(args.config)
    else:
        config = HopChainConfig()

    # Override with CLI args
    if args.api_base:
        config.api_base = args.api_base
    if args.api_key:
        config.api_key = args.api_key
    if args.model:
        config.vlm_model = args.model
    if args.weak_model:
        config.weak_model = args.weak_model
    config.min_complexity_score = args.min_complexity
    config.combo_size_min = args.combo_size_min
    config.combo_size_max = args.combo_size_max
    config.max_combos_per_image = args.max_combos
    config.queries_per_combo = args.queries_per_combo
    config.target_hops = args.target_hops
    config.output_dir = args.output_dir

    # Collect images
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            image_paths.extend(str(p) for p in sorted(img_dir.glob(ext)))

    if not image_paths:
        parser.error("No images provided. Use --images or --image-dir")

    logger.info(f"HopChain pipeline starting with {len(image_paths)} images")
    logger.info(f"Config: {json.dumps(asdict(config), indent=2)}")

    # Run pipeline
    results = run_pipeline(image_paths, config, skip_stage4=args.skip_stage4)

    # Format for RLVR if we have verified queries
    if results and "answer" in results[0]:
        rlvr_path = Path(config.output_dir) / "rlvr_train.jsonl"
        format_for_rlvr(results, str(rlvr_path))

    logger.info(f"Pipeline complete! Results in {config.output_dir}/")


if __name__ == "__main__":
    main()
