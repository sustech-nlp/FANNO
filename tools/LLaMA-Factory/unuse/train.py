import os
import yaml
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run LLaMAFactory training with YAML config.")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--dataset", type=str, default="tools_sft", help="训练数据集名称")
    parser.add_argument("--template", type=str, default="qwen3", help="数据集模板")
    parser.add_argument("--total_batch_size", type=int, default=8, help="总训练 batch size")
    parser.add_argument("--per_device_batch_size", type=int, default=1, help="每卡训练 batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--num_epochs", type=float, default=4.0, help="训练轮数")
    parser.add_argument("--max_samples", type=int, default=100000, help="最大训练样本数")
    parser.add_argument("--cutoff_len", type=int, default=32768, help="输入最大长度")
    parser.add_argument("--yaml_path", type=str, default="train_config.yaml", help="生成的 YAML 文件路径")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3,4,5,6,7", help="使用的 GPU id")
    args = parser.parse_args()

    # 计算 GPU 数量
    gpu_num = len(args.cuda_devices.split(","))
    
    # 自动计算 gradient_accumulation_steps
    grad_accum_steps = max(1, args.total_batch_size // (gpu_num * args.per_device_batch_size))
    print(f"使用 {gpu_num} 张 GPU，总 batch_size={args.total_batch_size}, 每卡 batch_size={args.per_device_batch_size}, 自动计算 gradient_accumulation_steps={grad_accum_steps}")

    # 构建 YAML 配置
    config = {
        "model_name_or_path": args.model_path,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "deepspeed": "examples/deepspeed/ds_z3_config.json",
        "dataset": args.dataset,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "max_samples": args.max_samples,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        "output_dir": args.output_dir,
        "logging_steps": 1,
        "save_steps": 50000,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None
    }

    # 保存 YAML 文件
    with open(args.yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"YAML 配置已保存到 {args.yaml_path}")

    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["FORCE_TORCHRUN"] = "1"

    # 执行训练
    train_cmd = f"llamafactory-cli train {args.yaml_path}"
    print(f"开始训练: {train_cmd}")
    subprocess.run(train_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()
