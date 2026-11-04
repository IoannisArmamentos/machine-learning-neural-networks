"""One training + evaluation run, following the original scripts' loops."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import torch
from torch.utils.data import DataLoader

from a2.models import ModelSpec
from a2.stats import accuracy_percent
from a2.timing import SyncTimer, device_synchronizer


@dataclass
class RunResult:
    train_seconds: float
    epoch_seconds: list[float]
    epoch_mean_loss: list[float]
    inference_seconds: float
    n_correct: int
    n_test: int
    indices: list[int]
    true_labels: list[int]
    predicted_labels: list[int]

    @property
    def accuracy_percent(self) -> float:
        return accuracy_percent(self.n_correct, self.n_test)


def seed_everything(seed: int) -> None:
    # torch.manual_seed also seeds every CUDA/ROCm device. No other RNG is used.
    torch.manual_seed(seed)


def train(model, spec: ModelSpec, loader: DataLoader, device: torch.device,
          log_interval: int = 100, max_batches: int | None = None):
    criterion = spec.make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    sync = device_synchronizer(device)
    n_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    epoch_seconds: list[float] = []
    epoch_loss: list[float] = []

    with SyncTimer(sync) as total:
        for epoch in range(spec.epochs):
            with SyncTimer(sync) as ep:
                model.train()
                # Accumulate on-device: avoids adding a host sync on every step.
                loss_sum = torch.zeros((), device=device)
                for i, (images, labels) in enumerate(islice(loader, max_batches)):
                    images = spec.prepare_input(images).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach()
                    if log_interval and (i + 1) % log_interval == 0:
                        print(f"  Epoch [{epoch + 1}/{spec.epochs}], Step[{i + 1}/{n_steps}], "
                              f"Loss: {loss.item():.4f}", flush=True)
            epoch_seconds.append(ep.seconds)
            # .item() happens after the epoch timer stopped, but inside the total
            # timer; it is one tiny sync per epoch (negligible).
            epoch_loss.append(loss_sum.item() / n_steps)
    return total.seconds, epoch_seconds, epoch_loss


@torch.no_grad()
def evaluate(model, spec: ModelSpec, loader: DataLoader, device: torch.device,
             max_batches: int | None = None):
    """Times the full test pass. Results stay on-device until the timer stops."""
    sync = device_synchronizer(device)
    model.eval()
    preds, labels_all, idx_all = [], [], []
    with SyncTimer(sync) as timer:
        for images, labels, indices in islice(loader, max_batches):
            images = spec.prepare_input(images).to(device)
            outputs = model(images)
            preds.append(outputs.argmax(dim=1))
            labels_all.append(labels)
            idx_all.append(indices)
    predicted = torch.cat(preds).cpu()
    true = torch.cat(labels_all)
    indices = torch.cat(idx_all)
    n_correct = int((predicted == true).sum())
    return timer.seconds, n_correct, indices.tolist(), true.tolist(), predicted.tolist()


def warmup(spec: ModelSpec, train_ds, device: torch.device, n_batches: int) -> None:
    """Untimed throw-away steps so one-time GPU kernel compilation/autotuning is
    not charged to run 1. Runs *before* seeding, on a separate model instance."""
    if n_batches <= 0:
        return
    model = spec.build().to(device)
    criterion = spec.make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=False)
    model.train()
    for images, labels in islice(loader, n_batches):
        images, labels = spec.prepare_input(images).to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for images, _ in islice(loader, n_batches):
            model(spec.prepare_input(images).to(device))
    device_synchronizer(device)()
    del model, optimizer


def run_once(spec: ModelSpec, train_loader, test_loader, device: torch.device, seed: int,
             log_interval: int = 100, max_train_batches: int | None = None,
             max_test_batches: int | None = None) -> RunResult:
    seed_everything(seed)  # before model init so weight init is seeded
    model = spec.build().to(device)
    train_s, epoch_s, epoch_loss = train(model, spec, train_loader, device,
                                         log_interval, max_train_batches)
    infer_s, n_correct, idx, true, pred = evaluate(model, spec, test_loader, device,
                                                   max_test_batches)
    return RunResult(train_s, epoch_s, epoch_loss, infer_s, n_correct, len(true), idx, true, pred)
