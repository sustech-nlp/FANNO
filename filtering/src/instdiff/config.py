from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("YAML 顶层需要是字典")
        return data
    except OSError as exc:
        raise RuntimeError(f"读取配置失败: {path}") from exc
