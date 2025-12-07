import math

import torch

from electrodrive.learn import train as train_module
from electrodrive.learn.specs import ExperimentConfig, TrainerSpec


class _DummyModel(torch.nn.Module):
    """
    Minimal model with compute_loss for scheduler tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def compute_loss(self, batch, loss_weights):
        pred = self.linear(batch["X"])
        diff = pred - batch["V_gt"]
        loss = (diff**2).mean()
        return {"total": loss}


class _DummyLoader:
    def __init__(self, num_batches: int, template: dict[str, torch.Tensor]) -> None:
        self.num_batches = num_batches
        self.template = template

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "X": self.template["X"].clone(),
                "V_gt": self.template["V_gt"].clone(),
            }


def test_scheduler_total_steps_matches_optimizer_steps(monkeypatch, tmp_path):
    num_batches = 10
    max_epochs = 3
    schedulers: list = []

    def fake_build_dataloaders(config):
        template = {
            "X": torch.ones(2, 1),
            "V_gt": torch.zeros(2, 1),
        }
        return _DummyLoader(num_batches, template), []

    monkeypatch.setattr(train_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(train_module, "initialize_model", lambda cfg: _DummyModel())

    class RecordingScheduler:
        def __init__(self, optimizer, T_max, **kwargs):
            self.optimizer = optimizer
            self.T_max = T_max
            self.kwargs = kwargs
            self.step_calls = 0
            self.last_epoch = -1
            schedulers.append(self)

        def step(self):
            self.step_calls += 1
            self.last_epoch += 1

    monkeypatch.setattr(
        train_module.optim.lr_scheduler,
        "CosineAnnealingLR",
        RecordingScheduler,
    )

    def run_case(accum_steps: int, run_dir):
        trainer = TrainerSpec(
            max_epochs=max_epochs,
            accum_steps=accum_steps,
            points_per_step=4,
            lr_scheduler="cosine",
        )
        config = ExperimentConfig(
            exp_name=f"scheduler_case_{accum_steps}",
            trainer=trainer,
            device="cpu",
        )
        exit_code = train_module.train(config, run_dir)
        assert exit_code == 0
        assert schedulers, "scheduler was not constructed"
        sched = schedulers[-1]
        expected_steps_per_epoch = math.ceil(max(num_batches, 1) / max(accum_steps, 1))
        expected_total_steps = max_epochs * expected_steps_per_epoch
        assert sched.T_max == expected_total_steps
        assert sched.step_calls == expected_total_steps
        assert sched.last_epoch == expected_total_steps - 1

    run_case(accum_steps=1, run_dir=tmp_path / "accum1")
    run_case(accum_steps=4, run_dir=tmp_path / "accum4")
