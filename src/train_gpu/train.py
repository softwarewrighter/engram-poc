"""Unsloth-based LoRA training for NVIDIA GPUs.

This module provides fast LoRA fine-tuning using Unsloth, which offers
2-5x speedup over standard HuggingFace training on NVIDIA GPUs.

Usage:
    python -m src.train_gpu.train --model "HuggingFaceTB/SmolLM-135M-Instruct"
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Conditional imports - only fail if actually running on GPU
try:
    import torch
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


@dataclass
class TrainConfig:
    """Training configuration for Unsloth."""
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    output_dir: str = "./adapters-gpu"
    data_dir: str = "./data"

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training parameters
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_seq_length: int = 512

    # Optimization
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False

    # Logging
    logging_steps: int = 10
    save_steps: int = 100


def load_training_data(data_dir: Path) -> Dataset:
    """Load training data from JSONL files.

    Args:
        data_dir: Directory containing train.jsonl

    Returns:
        HuggingFace Dataset
    """
    train_file = data_dir / "train.jsonl"

    examples = []
    with open(train_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # Convert MLX-LM format to text format for SFTTrainer
            messages = data.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0].get("content", "")
                assistant_msg = messages[1].get("content", "")
                # Format as instruction-response
                text = f"### Instruction:\n{user_msg}\n\n### Response:\n{assistant_msg}"
                examples.append({"text": text})

    return Dataset.from_list(examples)


def train_with_unsloth(config: TrainConfig) -> dict:
    """Run LoRA training with Unsloth.

    Args:
        config: Training configuration

    Returns:
        Training results dictionary
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "Unsloth not available. Install with: "
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )

    print(f"Loading model: {config.model_name}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=42,
    )

    # Load training data
    print(f"Loading training data from: {config.data_dir}")
    dataset = load_training_data(Path(config.data_dir))
    print(f"Training examples: {len(dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save adapter
    print(f"\nSaving adapter to: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Also save in safetensors format for compatibility
    model.save_pretrained_merged(
        config.output_dir + "-merged",
        tokenizer,
        save_method="lora",
    )

    results = {
        "model_name": config.model_name,
        "output_dir": config.output_dir,
        "train_samples": len(dataset),
        "epochs": config.epochs,
        "final_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Runtime: {results['train_runtime']:.1f}s")
    print(f"Adapter saved to: {config.output_dir}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter with Unsloth on NVIDIA GPU"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        default="./adapters-gpu",
        help="Directory to save adapter",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )

    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    train_with_unsloth(config)


if __name__ == "__main__":
    main()
