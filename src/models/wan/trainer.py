"""
LoRA Training Engine for WAN Models with Memory Optimization.
Handles the training loop, loss computation, and optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
import time
import math
from dataclasses import dataclass
from contextlib import nullcontext

from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from peft import get_peft_model_state_dict

from .config import WanLoRAConfig, log_memory_usage
from .model_loader import WanModelLoader
from ..preprocessing.data_manager import DatasetManager, TrainingSample

logger = get_logger(__name__)

@dataclass
class TrainingMetrics:
    """Training metrics container."""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    diffusion_loss: float = 0.0
    routing_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    time_per_step: float = 0.0
    memory_used: float = 0.0
    throughput: float = 0.0  # samples per second

class WanLoRATrainer:
    """
    LoRA trainer for WAN models with advanced memory optimization.
    """
    
    def __init__(
        self,
        config: WanLoRAConfig,
        model_loader: WanModelLoader,
        dataset_manager: DatasetManager,
        output_dir: Union[str, Path],
        logging_dir: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.model_loader = model_loader
        self.dataset_manager = dataset_manager
        self.output_dir = Path(output_dir)
        self.logging_dir = Path(logging_dir) if logging_dir else self.output_dir / "logs"
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.metrics_history = []
        
        # Model components (will be set in setup)
        self.models = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        
        # Accelerate setup
        self.accelerator = None
        self.scaler = None
        
        # Training setup
        self._setup_training()
    
    def _setup_training(self):
        """Initialize training components."""
        logger.info("Setting up training environment...")
        
        # Setup accelerate
        self._setup_accelerator()
        
        # Load models
        self.models = self.model_loader.load_models()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup learning rate scheduler
        self._setup_lr_scheduler()
        
        # Setup data loader
        self._setup_dataloader()
        
        # Prepare with accelerate
        self._prepare_for_training()
        
        # Setup gradient scaler for mixed precision
        if self.config.use_mixed_precision and not self.accelerator.mixed_precision:
            self.scaler = GradScaler()
        
        logger.info("Training setup complete")
    
    def _setup_accelerator(self):
        """Setup accelerate for distributed training."""
        project_config = ProjectConfiguration(
            project_dir=self.output_dir,
            logging_dir=self.logging_dir,
        )
        
        # Determine mixed precision type
        mixed_precision = "no"
        if self.config.use_mixed_precision:
            if self.config.precision == "fp16":
                mixed_precision = "fp16"
            elif self.config.precision == "bf16":
                mixed_precision = "bf16"
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard" if self.config.enable_logging else None,
            project_config=project_config,
            cpu=not torch.cuda.is_available(),
        )
        
        # Set random seed
        if self.config.seed is not None:
            set_seed(self.config.seed)
        
        logger.info(f"Accelerator setup: {self.accelerator.state}")
    
    def _setup_optimizer(self):
        """Setup the optimizer."""
        self.optimizer = self.model_loader.get_optimizer(
            learning_rate=self.config.learning_rate
        )
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
    
    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = self.config.max_train_steps
        if num_training_steps is None:
            # Calculate from epochs and dataset size
            steps_per_epoch = len(self.dataset_manager) // (
                self.config.train_batch_size * self.accelerator.num_processes
            )
            num_training_steps = self.config.num_train_epochs * steps_per_epoch
        
        self.lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"LR Scheduler: {self.config.lr_scheduler_type}, warmup: {self.config.lr_warmup_steps}")
    
    def _setup_dataloader(self):
        """Setup training data loader."""
        self.dataloader = self.dataset_manager.get_dataloader(
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self._collate_fn,
        )
        logger.info(f"DataLoader: {len(self.dataloader)} batches per epoch")
    
    def _collate_fn(self, batch: List[TrainingSample]) -> Dict[str, torch.Tensor]:
        """Custom collate function for training samples."""
        # Extract components from batch
        videos = []
        images = []
        prompts = []
        
        for sample in batch:
            if sample.video_path:
                videos.append(sample.video_tensor)
            elif sample.image_path:
                images.append(sample.image_tensor)
            prompts.append(sample.prompt)
        
        # Stack tensors
        collated = {}
        
        if videos:
            collated["videos"] = torch.stack(videos)
        if images:
            collated["images"] = torch.stack(images)
        
        collated["prompts"] = prompts
        
        return collated
    
    def _prepare_for_training(self):
        """Prepare models and optimizers with accelerate."""
        # Only prepare the transformer (main trainable model)
        transformer = self.models["transformer"]
        
        self.transformer, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(
            transformer, self.optimizer, self.dataloader, self.lr_scheduler
        )
        
        # Update models dict
        self.models["transformer"] = self.transformer
        
        # Move other models to device
        device = self.accelerator.device
        for key, model in self.models.items():
            if key != "transformer" and hasattr(model, "to"):
                self.models[key] = model.to(device)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting LoRA training...")
        log_memory_usage("training start")
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self._load_checkpoint(self.resume_from_checkpoint)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.num_train_epochs):
            self.epoch = epoch
            epoch_metrics = self._train_epoch()
            
            # Save checkpoint
            if self.config.save_strategy == "epoch":
                self._save_checkpoint()
            
            # Early stopping check
            if self._should_stop_early(epoch_metrics):
                logger.info("Early stopping triggered")
                break
        
        # Final save
        self._save_final_model()
        
        total_time = time.time() - start_time
        
        results = {
            "final_loss": self.best_loss,
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "total_time": total_time,
            "metrics_history": self.metrics_history,
        }
        
        logger.info(f"Training completed in {total_time:.2f}s")
        log_memory_usage("training end")
        
        return results
    
    def _train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.transformer.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for step, batch in enumerate(self.dataloader):
            step_start_time = time.time()
            
            # Forward pass
            with self.accelerator.accumulate(self.transformer):
                loss_dict = self._forward_step(batch)
                loss = loss_dict["total_loss"]
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.transformer.parameters(),
                        self.config.max_grad_norm
                    )
                else:
                    grad_norm = 0.0
                
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Update global step
            if self.accelerator.sync_gradients:
                self.global_step += 1
            
            # Calculate metrics
            step_time = time.time() - step_start_time
            throughput = self.config.train_batch_size / step_time
            
            metrics = TrainingMetrics(
                step=self.global_step,
                epoch=self.epoch,
                total_loss=loss_dict["total_loss"].item(),
                diffusion_loss=loss_dict.get("diffusion_loss", 0.0),
                routing_loss=loss_dict.get("routing_loss", 0.0),
                learning_rate=self.lr_scheduler.get_last_lr()[0],
                grad_norm=grad_norm,
                time_per_step=step_time,
                throughput=throughput,
                memory_used=torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
            )
            
            epoch_loss += metrics.total_loss
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(metrics)
            
            # Save checkpoint
            if (self.config.save_strategy == "steps" and 
                self.global_step % self.config.save_steps == 0):
                self._save_checkpoint()
            
            # Validation/preview
            if (self.config.validation_steps > 0 and 
                self.global_step % self.config.validation_steps == 0):
                self._generate_preview()
            
            # Max steps check
            if (self.config.max_train_steps and 
                self.global_step >= self.config.max_train_steps):
                break
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / len(self.dataloader)
        
        logger.info(
            f"Epoch {self.epoch}: avg_loss={avg_loss:.4f}, "
            f"time={epoch_time:.2f}s, steps={self.global_step}"
        )
        
        return TrainingMetrics(
            epoch=self.epoch,
            total_loss=avg_loss,
            time_per_step=epoch_time / len(self.dataloader),
        )
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss components
        """
        # Get model components
        tokenizer = self.models["tokenizer"]
        text_encoder = self.models["text_encoder"]
        vae = self.models["vae"]
        scheduler = self.models["scheduler"]
        
        # Process prompts
        prompts = batch["prompts"]
        
        # Tokenize prompts
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = text_inputs.input_ids.to(self.accelerator.device)
        attention_mask = text_inputs.attention_mask.to(self.accelerator.device)
        
        # Encode text
        with torch.no_grad():
            encoder_hidden_states = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        
        # Process visual inputs
        if "videos" in batch:
            # Video-to-video or text-to-video
            visual_inputs = batch["videos"].to(self.accelerator.device)
        elif "images" in batch:
            # Image-to-video
            visual_inputs = batch["images"].to(self.accelerator.device)
        else:
            raise ValueError("No visual inputs found in batch")
        
        # Encode visual inputs with VAE
        with torch.no_grad():
            latents = vae.encode(visual_inputs).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise with transformer
        model_output = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            return_attention_weights=False,
        )
        
        # Calculate diffusion loss
        target = noise  # Predicting noise
        diffusion_loss = F.mse_loss(model_output.sample, target, reduction="mean")
        
        # Calculate routing loss (if TREAD is enabled)
        routing_loss = torch.tensor(0.0, device=diffusion_loss.device)
        if hasattr(model_output, "routing_info") and model_output.routing_info:
            for layer_info in model_output.routing_info.values():
                if "load_balancing_loss" in layer_info:
                    routing_loss += layer_info["load_balancing_loss"]
        
        # Total loss
        total_loss = diffusion_loss + self.config.routing_loss_weight * routing_loss
        
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "routing_loss": routing_loss,
        }
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        if self.accelerator.is_main_process:
            self.metrics_history.append(metrics)
            
            log_dict = {
                "train/loss": metrics.total_loss,
                "train/diffusion_loss": metrics.diffusion_loss,
                "train/routing_loss": metrics.routing_loss,
                "train/learning_rate": metrics.learning_rate,
                "train/grad_norm": metrics.grad_norm,
                "train/throughput": metrics.throughput,
                "train/memory_gb": metrics.memory_used,
                "global_step": metrics.step,
                "epoch": metrics.epoch,
            }
            
            if self.config.enable_logging:
                self.accelerator.log(log_dict, step=self.global_step)
            
            logger.info(
                f"Step {metrics.step}: loss={metrics.total_loss:.4f}, "
                f"lr={metrics.learning_rate:.2e}, "
                f"throughput={metrics.throughput:.1f} samples/s"
            )
    
    def _generate_preview(self):
        """Generate preview images/videos during training."""
        if not self.config.enable_preview:
            return
        
        logger.info("Generating training preview...")
        
        self.transformer.eval()
        
        try:
            with torch.no_grad():
                # Use the pipeline for generation
                pipeline = self.models["pipeline"]
                
                # Generate with a fixed prompt
                preview_prompt = self.config.preview_prompt or "A beautiful landscape"
                
                result = pipeline(
                    prompt=preview_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=256,  # Smaller for faster generation
                    width=256,
                    num_frames=8 if "video" in self.config.model_type.lower() else 1,
                )
                
                # Save preview
                preview_dir = self.output_dir / "previews"
                preview_dir.mkdir(exist_ok=True)
                
                preview_path = preview_dir / f"step_{self.global_step}.png"
                
                if hasattr(result, 'frames'):
                    # Video result
                    result.frames[0][0].save(preview_path)
                else:
                    # Image result
                    result.images[0].save(preview_path)
                
                logger.info(f"Preview saved to {preview_path}")
        
        except Exception as e:
            logger.warning(f"Failed to generate preview: {e}")
        
        finally:
            self.transformer.train()
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save LoRA weights
        self.model_loader.save_lora_weights(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": self.config.__dict__,
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        logger.info("Checkpoint saved successfully")
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load LoRA weights
        self.model_loader.load_lora_weights(checkpoint_path)
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.accelerator.device)
            
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_loss = training_state["best_loss"]
            
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler_state_dict"])
            
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.warning("No training state found, starting fresh")
    
    def _save_final_model(self):
        """Save the final trained model."""
        if not self.accelerator.is_main_process:
            return
        
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving final model to {final_dir}")
        
        # Save LoRA weights
        self.model_loader.save_lora_weights(final_dir)
        
        # Save config
        config_path = final_dir / "config.json"
        with open(config_path, "w") as f:
            import json
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info("Final model saved successfully")
    
    def _should_stop_early(self, epoch_metrics: TrainingMetrics) -> bool:
        """Check if training should stop early."""
        if not self.config.early_stopping_patience:
            return False
        
        # Simple early stopping based on loss
        if epoch_metrics.total_loss < self.best_loss:
            self.best_loss = epoch_metrics.total_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = getattr(self, 'epochs_without_improvement', 0) + 1
        
        return self.epochs_without_improvement >= self.config.early_stopping_patience
