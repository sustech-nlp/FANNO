#!/usr/bin/env bash

#!/usr/bin/env bash

# Fixed-parameter runner for FANNO-Tools generation
# No external arguments are read; edit variables below to change behavior.

set -euo pipefail

python synthetic_self_play.py \
  --input data/unlabel_data.jsonl \
  --output synthetic_data_1k.jsonl \
  --target 1000 \
  --max-turns 8 \
  --min-score 7 \
  --workers 150 \
  --logic-pattern '{"smooth":0.2,"partial_failure":0.2,"error_recovery":0.2,"escalation":0.2,"user_change_mind":0.2}' \
  --seed 42
