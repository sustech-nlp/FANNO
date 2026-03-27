#!/usr/bin/env python3
"""
Dataset cleaning utility.

- Ensures conversations end with a GPT turn; if the last turn is human, drop that final human.
- Optional flag to drop records that still do not end with GPT after trimming.

Usage examples:
  python scripts/clean_dataset.py clean --input synthetic_data_1k.jsonl --output synthetic_data_1k_clean.jsonl
  python scripts/clean_dataset.py clean --input synthetic_data_1k.jsonl --output synthetic_data_1k_clean.jsonl --drop_if_no_gpt True
"""

import json
from pathlib import Path
from typing import Optional

import fire


class DatasetCleaner:
    def clean(
        self,
        input: str = "synthetic_data_1k.jsonl",
        output: str = "synthetic_data_1k_clean.jsonl",
        drop_if_no_gpt: bool = False,
    ) -> dict:
        """Clean dataset so conversations end with GPT; drop the trailing human question if needed."""
        input_path = Path(input)
        output_path = Path(output)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        total = 0
        trimmed = 0
        dropped = 0
        written = 0

        with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
            
            for line in src:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue

                conv = record.get("conversations", [])
                if not conv:
                    dropped += 1
                    continue


                
                if conv:
                    find = False
                    for item in conv:
                        value = item.get("value")
                
                        if value is None or value.strip() == "":
                            
                            find = True
                            
                            break
                    if find:
                        dropped +=1
                        continue
                

                # Remove trailing human turn if present
                if conv and conv[-1].get("from") == "human":
                    conv = conv[:-1]
                    trimmed += 1
                             
                # # remove until last one is gpt
                # if conv:
                #     while len(conv) > 0 and conv[-1].get("from") != "gpt":
                #         conv = conv[:-1]
                #         trimmed += 1
                                            

                if not conv:
                    dropped += 1
                    continue

                # Optionally drop if still not ending with GPT
                if drop_if_no_gpt and conv[-1].get("from") != "gpt":
                    dropped += 1
                    continue

                record["conversations"] = conv
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

        return {
            "total": total,
            "trimmed": trimmed,
            "dropped": dropped,
            "written": written,
            "output": str(output_path),
        }


if __name__ == "__main__":
    fire.Fire(DatasetCleaner)
