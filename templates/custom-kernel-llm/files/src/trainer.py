import torch
import time
import math
from pathlib import Path


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, cfg):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.device
        self.step_count = 0
        self.best_val_loss = float("inf")

    def train(self):
        print(f"\nTraining on device: {self.device}")
        print(f"Kernel backend: {self.cfg.kernel_backend}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'=' * 60}\n")

        self._warmup()

        for epoch in range(self.cfg.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate()

            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Perplexity: {math.exp(val_loss):.2f}"
            )

            self._save_checkpoint(epoch, val_loss)

        print("\nTraining complete!")

    def _warmup(self, n_steps=5):
        print("Warming up...")
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= n_steps:
                    break
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                self.model(input_ids, targets)
                if self.device == "cuda":
                    torch.cuda.synchronize()
        print("Warmup complete.\n")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            logits, loss = self.model(input_ids, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1
            self.step_count += 1

        return total_loss / n_batches

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                logits, loss = self.model(input_ids, targets)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _save_checkpoint(self, epoch, val_loss):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "cfg": self.cfg,
                },
                path,
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
