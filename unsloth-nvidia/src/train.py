"""Unsloth LoRA training for Engram PoC.

Fast fine-tuning on NVIDIA GPUs using Unsloth.

Usage:
    python -m src.train
    python -m src.train --model "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --load-in-4bit
"""

import unsloth  # Must be imported first for optimizations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


@dataclass
class TrainConfig:
    """Training configuration."""
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    output_dir: str = "./adapters"
    data_dir: str = "./data"

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_seq_length: int = 512

    # Options
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False

    # Logging
    logging_steps: int = 10
    save_steps: int = 100


def load_training_data(data_dir: Path) -> Dataset:
    """Load training data from JSONL."""
    train_file = data_dir / "train.jsonl"

    examples = []
    with open(train_file) as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0]["content"]
                assistant_msg = messages[1]["content"]
                text = f"### Instruction:\n{user_msg}\n\n### Response:\n{assistant_msg}"
                examples.append({"text": text})

    return Dataset.from_list(examples)


def train(config: TrainConfig) -> dict:
    """Run LoRA training with Unsloth."""
    print("=" * 60)
    print("ENGRAM PoC - Unsloth Training")
    print("=" * 60)
    print()
    print(f"Model:        {config.model_name}")
    print(f"Output:       {config.output_dir}")
    print(f"LoRA Rank:    {config.lora_rank}")
    print(f"Batch Size:   {config.batch_size}")
    print(f"4-bit:        {config.load_in_4bit}")
    print()

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    # Add LoRA
    print("Adding LoRA adapters...")
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

    # Load data
    print(f"Loading data from {config.data_dir}...")
    dataset = load_training_data(Path(config.data_dir))
    print(f"Training examples: {len(dataset)}")
    print()

    # Training args
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

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    # Train
    print("Starting training...")
    print("-" * 60)
    result = trainer.train()
    print("-" * 60)
    print()

    # Save
    print(f"Saving adapter to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    results = {
        "model_name": config.model_name,
        "output_dir": config.output_dir,
        "train_samples": len(dataset),
        "epochs": config.epochs,
        "final_loss": result.training_loss,
        "runtime_seconds": result.metrics.get("train_runtime", 0),
    }

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Loss:     {results['final_loss']:.4f}")
    print(f"Runtime:        {results['runtime_seconds']:.1f}s")
    print(f"Adapter saved:  {config.output_dir}")
    print()

    # Save results
    results_file = Path(config.output_dir) / "train_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train LoRA adapter with Unsloth")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--output-dir", default="./adapters")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

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

    train(config)


if __name__ == "__main__":
    main()
