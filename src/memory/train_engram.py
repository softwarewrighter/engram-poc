"""
Training script for Engram-enhanced models.

Supports:
1. Training Engram memory tables alone
2. Training Engram + LoRA together (combined approach)
3. Differentiated learning rates for memory vs other parameters
4. Auto-detection of CUDA/MPS/CPU with platform-specific optimizations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model_wrapper import inject_engram_into_model, EngramModelWrapper


def get_device_info() -> Tuple[torch.device, str, dict]:
    """
    Detect best available device and return platform-specific settings.

    Returns:
        Tuple of (device, platform_name, recommended_settings)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Recommend settings based on GPU memory
        if gpu_memory >= 16:
            settings = {
                "batch_size": 8,
                "memory_size": 50000,
                "num_workers": 4,
                "pin_memory": True,
                "mixed_precision": True,
            }
        elif gpu_memory >= 8:
            settings = {
                "batch_size": 4,
                "memory_size": 30000,
                "num_workers": 2,
                "pin_memory": True,
                "mixed_precision": True,
            }
        else:
            settings = {
                "batch_size": 2,
                "memory_size": 10000,
                "num_workers": 2,
                "pin_memory": True,
                "mixed_precision": False,
            }

        platform = f"CUDA ({gpu_name}, {gpu_memory:.1f}GB)"

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        platform = "MPS (Apple Silicon)"
        settings = {
            "batch_size": 4,
            "memory_size": 30000,
            "num_workers": 0,  # MPS doesn't benefit from workers
            "pin_memory": False,
            "mixed_precision": False,  # MPS has limited mixed precision support
        }

    else:
        device = torch.device("cpu")
        platform = "CPU"
        settings = {
            "batch_size": 2,
            "memory_size": 10000,
            "num_workers": 0,
            "pin_memory": False,
            "mixed_precision": False,
        }

    return device, platform, settings


def print_device_info():
    """Print detected device information."""
    device, platform, settings = get_device_info()
    print(f"\n{'='*60}")
    print("Device Detection")
    print(f"{'='*60}")
    print(f"Platform: {platform}")
    print(f"Device: {device}")
    print(f"Recommended settings:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")
    return device, platform, settings


@dataclass
class EngramTrainConfig:
    """Configuration for Engram training."""

    # Model
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    memory_size: int = 50000
    inject_layers: Optional[List[int]] = None

    # Training
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    memory_learning_rate: float = 1e-3  # Higher LR for memory tables
    max_seq_length: int = 512
    gradient_clip: float = 1.0

    # Data
    train_file: str = "data/train.jsonl"
    valid_file: str = "data/valid.jsonl"

    # Output
    output_dir: str = "adapters-engram"

    # LoRA integration (optional)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Device
    device: str = "auto"


class EngramDataset(Dataset):
    """Simple dataset for training from JSONL."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                # Handle both message format and simple prompt/completion format
                if "messages" in data:
                    messages = data["messages"]
                    # Convert to text
                    text = ""
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            text += f"User: {content}\n"
                        elif role == "assistant":
                            text += f"Assistant: {content}\n"
                    self.examples.append(text)
                elif "prompt" in data and "completion" in data:
                    text = f"{data['prompt']}{data['completion']}"
                    self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_engram(config: EngramTrainConfig):
    """Main training function with auto device detection."""

    # Detect device and get platform-specific settings
    device, platform, recommended = get_device_info()

    print("=" * 60)
    print("Engram Training")
    print("=" * 60)
    print(f"Platform: {platform}")
    print(f"Model: {config.model_name}")
    print(f"Memory size: {config.memory_size:,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Use LoRA: {config.use_lora}")
    print("=" * 60)

    # Warn if settings differ from recommended
    if config.batch_size > recommended["batch_size"]:
        print(f"\nWarning: batch_size={config.batch_size} may be too large for {platform}")
        print(f"  Recommended: {recommended['batch_size']}")

    if config.memory_size > recommended["memory_size"] * 2:
        print(f"\nWarning: memory_size={config.memory_size:,} may be too large for {platform}")
        print(f"  Recommended: {recommended['memory_size']:,}")

    # Load model with Engram
    model, tokenizer = inject_engram_into_model(
        config.model_name,
        memory_size=config.memory_size,
        inject_layers=config.inject_layers,
        freeze_base=True,
        device=config.device,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print memory stats
    stats = model.get_memory_stats()
    print(f"\nEngram Statistics:")
    print(f"  Layers: {stats['num_engram_layers']}")
    print(f"  Memory slots per layer: {stats['memory_size']:,}")
    print(f"  Total Engram parameters: {stats['total_engram_params']:,}")

    # Optional: Add LoRA
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model

            # Unfreeze base for LoRA
            for param in model.model.parameters():
                param.requires_grad = False

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA to base model
            model.model = get_peft_model(model.model, lora_config)
            print(f"\nLoRA added with rank={config.lora_rank}")
        except ImportError:
            print("Warning: peft not installed, skipping LoRA")
            config.use_lora = False

    # Move to device
    model = model.to(device)
    print(f"\nUsing device: {device}")

    # CUDA-specific optimizations
    use_amp = False
    scaler = None
    if device.type == "cuda" and recommended.get("mixed_precision", False):
        try:
            scaler = torch.amp.GradScaler("cuda")
            use_amp = True
            print("Mixed precision (AMP) enabled")
        except Exception:
            pass

    # Load data
    train_dataset = EngramDataset(config.train_file, tokenizer, config.max_seq_length)
    valid_dataset = EngramDataset(config.valid_file, tokenizer, config.max_seq_length)

    # Platform-specific DataLoader settings
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": recommended.get("num_workers", 0),
        "pin_memory": recommended.get("pin_memory", False),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Valid: {len(valid_dataset)}")
    print(f"  DataLoader workers: {loader_kwargs['num_workers']}")

    # Setup optimizer with differentiated learning rates
    engram_params = model.engram_parameters()

    if config.use_lora:
        # Get LoRA parameters
        lora_params = [p for n, p in model.model.named_parameters() if "lora" in n.lower()]
        optimizer = torch.optim.AdamW([
            {"params": engram_params, "lr": config.memory_learning_rate},
            {"params": lora_params, "lr": config.learning_rate},
        ])
    else:
        optimizer = torch.optim.AdamW([
            {"params": engram_params, "lr": config.memory_learning_rate},
        ])

    # Training loop
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_valid_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Use AMP if available (CUDA only)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(engram_params, config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(engram_params, config.gradient_clip)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        valid_loss = 0
        num_valid = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                valid_loss += outputs.loss.item()
                num_valid += 1

        avg_valid_loss = valid_loss / num_valid if num_valid > 0 else 0

        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Valid Loss: {avg_valid_loss:.4f}")

        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            os.makedirs(config.output_dir, exist_ok=True)
            model.save_engram_weights(os.path.join(config.output_dir, "engram_weights.pt"))
            print(f"  Saved best model (valid loss: {avg_valid_loss:.4f})")

            # Also save config
            with open(os.path.join(config.output_dir, "config.json"), "w") as f:
                json.dump({
                    "model_name": config.model_name,
                    "memory_size": config.memory_size,
                    "inject_layers": config.inject_layers,
                    "use_lora": config.use_lora,
                }, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Weights saved to: {config.output_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Engram-enhanced model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect device and use recommended settings
  python -m src.memory.train_engram --auto

  # Train with Engram + LoRA on CUDA
  python -m src.memory.train_engram --use-lora --epochs 5

  # Quick test run
  python -m src.memory.train_engram --epochs 1 --batch-size 2 --memory-size 10000

  # Check device info only
  python -m src.memory.train_engram --device-info
        """
    )

    # Device options
    parser.add_argument("--device-info", action="store_true",
                        help="Print device info and exit")
    parser.add_argument("--auto", action="store_true",
                        help="Use recommended settings for detected device")

    # Model options
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--memory-size", type=int, default=None,
                        help="Memory table size per layer (default: auto)")
    parser.add_argument("--inject-layers", type=str, default=None,
                        help="Comma-separated layer indices to inject (default: all)")

    # Training options
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: auto based on device)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for LoRA/other params")
    parser.add_argument("--memory-lr", type=float, default=1e-3,
                        help="Learning rate for memory tables (10x higher)")

    # Data options
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--valid-file", default="data/valid.jsonl")
    parser.add_argument("--output-dir", default="adapters-engram")

    # LoRA options
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA adapters (combined approach)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    args = parser.parse_args()

    # Device info only
    if args.device_info:
        print_device_info()
        return

    # Get device recommendations
    device, platform, recommended = get_device_info()

    # Auto mode: use recommended settings
    if args.auto:
        print(f"\nAuto mode: Using recommended settings for {platform}")
        args.batch_size = recommended["batch_size"]
        args.memory_size = recommended["memory_size"]

    # Apply defaults from recommendations if not specified
    if args.batch_size is None:
        args.batch_size = recommended["batch_size"]
    if args.memory_size is None:
        args.memory_size = recommended["memory_size"]

    # Parse inject_layers
    inject_layers = None
    if args.inject_layers:
        inject_layers = [int(x.strip()) for x in args.inject_layers.split(",")]

    config = EngramTrainConfig(
        model_name=args.model,
        memory_size=args.memory_size,
        inject_layers=inject_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        memory_learning_rate=args.memory_lr,
        train_file=args.train_file,
        valid_file=args.valid_file,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    train_engram(config)


if __name__ == "__main__":
    main()
