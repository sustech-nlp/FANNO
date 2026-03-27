#!/usr/bin/env python3

import os
import subprocess

os.environ["HF_ENDPOINT"] = "https://huggingface.cn"
os.environ["WANDB_MODE"] = "offline"


def create_yaml_config(
    model_path,
    dataset,
    template,
    output_dir,
    total_batch_size,
    per_device_batch_size,
    learning_rate,
    num_epochs,
    max_samples,
    cutoff_len,
    gpus_per_node,
):
    total_batch_size = int(total_batch_size)
    per_device_batch_size = int(per_device_batch_size)
    gpus_per_node = int(gpus_per_node)

    grad_accum_steps = max(1, total_batch_size // (per_device_batch_size * gpus_per_node))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    model_name = os.path.basename(model_path.rstrip("/"))
    yaml_path = os.path.join(output_dir, f"{model_name}-tools.yaml")

    yaml_content = f"""### model
model_name_or_path: {model_path}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: {dataset}
template: {template}
cutoff_len: {cutoff_len}
max_samples: {max_samples}
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}
save_only_model: true
logging_steps: 1
save_steps: 50000
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: {per_device_batch_size}
gradient_accumulation_steps: {grad_accum_steps}
learning_rate: {learning_rate}
num_train_epochs: {num_epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


def sft(
    model_path,
    dataset,
    template,
    output_dir,
    total_batch_size,
    per_device_batch_size,
    learning_rate,
    num_epochs,
    max_samples,
    cutoff_len,
    gpus_per_node,
    cuda_devices,
):
    yaml_file = create_yaml_config(
        model_path=model_path,
        dataset=dataset,
        template=template,
        output_dir=output_dir,
        total_batch_size=total_batch_size,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_samples=max_samples,
        cutoff_len=cutoff_len,
        gpus_per_node=gpus_per_node,
    )

    print(f"Created config file: {yaml_file}")
    print(f"Training model: {model_path} with dataset: {dataset}")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    os.environ["FORCE_TORCHRUN"] = "1"

    cmd = f"llamafactory-cli train {yaml_file}"
    print(f"Running command: {cmd}")
    print("=" * 50)

    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line.rstrip())

        return_code = process.wait()
        if return_code == 0:
            print("=" * 50)
            print("Training completed successfully!")
        else:
            print("=" * 50)
            print(f"Training failed with return code: {return_code}")
            raise subprocess.CalledProcessError(return_code, cmd)
    except subprocess.CalledProcessError as exc:
        print(f"Training failed with error: {exc}")
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        raise


if __name__ == "__main__":
    MODEL_PATH = "/mnt/msranlphot_intern/zhuhe/models/Meta-Llama-3.1-8B-Instruct"
    DATASET = "tools_sft"
    TEMPLATE = "llama3"
    OUTPUT_ROOT = "/mnt/msranlphot_intern/zhuhe/saves/fanno-tools"

    CUDA_DEVICES = "0,1,2,3"
    GPUS_PER_NODE = len(CUDA_DEVICES.split(","))

    # TOTAL_BATCHS = [16, 32, 64, 128]
    TOTAL_BATCHS = [128]
    PER_DEVICE_BATCH = 4
    # LEARNING_RATES = [5e-6, 1e-5, 2e-5, 5e-5]
    LEARNING_RATES = [5e-5] 
    # EPOCHS = [2, 4]
    EPOCHS = [4]

    MAX_SAMPLES = 100000
    CUTOFF_LEN = 32768

    model_name = os.path.basename(MODEL_PATH.rstrip("/"))
    print("=== Starting Tools Domain Fine-tuning ===")

    for total_batch in TOTAL_BATCHS:
        for lr in LEARNING_RATES:
            for epoch in EPOCHS:
                info = f"tb{total_batch}_lr{lr}_ep{epoch}"
                output_dir = os.path.join(OUTPUT_ROOT, f"{model_name}-{info}")

                print(f"\nTraining config: {model_name}-{info}")
                print(f"Output dir: {output_dir}")

                try:
                    sft(
                        model_path=MODEL_PATH,
                        dataset=DATASET,
                        template=TEMPLATE,
                        output_dir=output_dir,
                        total_batch_size=total_batch,
                        per_device_batch_size=PER_DEVICE_BATCH,
                        learning_rate=lr,
                        num_epochs=epoch,
                        max_samples=MAX_SAMPLES,
                        cutoff_len=CUTOFF_LEN,
                        gpus_per_node=GPUS_PER_NODE,
                        cuda_devices=CUDA_DEVICES,
                    )
                    print(f"✓ Completed training: {info}")
                except Exception as exc:
                    print(f"✗ Error in training {info}: {exc}")
                    continue
