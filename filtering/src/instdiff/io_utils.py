import json
from typing import Iterable, List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    try:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except OSError as exc:
        raise RuntimeError(f"读取失败: {path}") from exc


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RuntimeError(f"写入失败: {path}") from exc
