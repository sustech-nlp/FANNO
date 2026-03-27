## start model 
# MODEL_ID=Qwen3-8B
# vllm serve $MODEL_ID \
#     --tensor-parallel-size 1 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --host 0.0.0.0 \
#     --port 54545


export OPENAI_API_KEY=dummy
export OPENAI_API_BASE=http://localhost:8000/v1

## 运行 τ²-bench 评估
tau2 run \
    --domain airline \
    --agent-llm openai/$MODEL_NAME \
    --user-llm openai/$MODEL_NAME \
    --num-trials 1 \
    --num-tasks 5 \
    --max-concurrency 32 \
    --task-split base \
    --save-to my_model_airline \
    --verbose \


    ## 他非常重要的参数。
    ## domain 选择（可选：mock, airline, retail, telecom, telecom-workflow）
    ## --num-tasks 5 \
    ## --task-ids 0,1,2,3,4 \
    ## --task-split base \
    ## Agent 类型（可选：llm_agent, llm_agent_solo, llm_agent_gt）
    ## --agent llm_agent \
    ## User 类型（可选：llm_user, user_simulator, dummy_user）
    ## --user user_simulator \
    ## LLM 额外参数
    ## --agent-llm-args '{"temperature": 0.0}' \
    ## --user-llm-args '{"temperature": 0.0}' \
    ## 最大步数和错误数
    ## --max-steps 200 \
    ## --max-errors 10