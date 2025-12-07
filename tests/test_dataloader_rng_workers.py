import contextlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

import electrodrive.learn.dataset as D
from electrodrive.learn.encoding import ENCODING_DIM
from electrodrive.learn.specs import ExperimentConfig


class _ToyDataset(Dataset):
    def __init__(
        self,
        config,
        split: str,
    ):
        self.n_points = int(
            config.dataset.samples_per_problem_jit
        )
        if split == "train":
            self.n = int(
                config.dataset.n_train_problems
            )
        else:
            self.n = int(
                config.dataset.n_val_problems
            )
        self.dtype = getattr(
            torch, config.train_dtype
        )
        # Initial rng will be replaced by worker_init_fn inside each worker
        # via _collocation_worker_init_fn; we keep this for attribute presence.
        self.rng = np.random.default_rng(
            int(config.seed)
            + (0 if split == "train" else 1)
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        info = get_worker_info()
        worker_id = info.id if info is not None else -1

        # Draw from the per-worker dataset RNG. If worker_init_fn is wired
        # correctly, each worker gets a different seed and thus a different
        # random stream.
        rand_val = int(
            self.rng.integers(0, 10_000)
        )

        X = torch.full(
            (self.n_points, 3),
            float(rand_val),
            dtype=self.dtype,
        )
        zeros = torch.zeros(
            self.n_points,
            dtype=self.dtype,
        )
        encoding = torch.zeros(
            self.n_points,
            ENCODING_DIM,
            dtype=self.dtype,
        )

        return {
            "X": X,
            "V_gt": zeros,
            "is_boundary": torch.zeros(
                self.n_points,
                dtype=torch.bool,
            ),
            "mask_finite": torch.ones(
                self.n_points,
                dtype=torch.bool,
            ),
            "encoding": encoding,
            "worker_id": torch.full(
                (self.n_points,),
                worker_id,
                dtype=torch.int64,
            ),
            "rand_val": torch.full(
                (self.n_points,),
                rand_val,
                dtype=torch.int64,
            ),
        }


def test_dataloader_rng_workers(monkeypatch):
    # Use a lightweight dataset to isolate worker seeding behaviour.
    monkeypatch.setattr(
        D,
        "ElectrostaticsJITDataset",
        _ToyDataset,
    )

    config = ExperimentConfig(
        exp_name="rng_workers_test"
    )
    config.device = "cpu"
    config.seed = 1234
    config.dataset.n_train_problems = 6
    config.dataset.n_val_problems = 0
    config.dataset.samples_per_problem_jit = 4
    config.trainer.batch_size = 8
    config.trainer.num_workers = 2
    config.trainer.pin_memory = False

    def _collect():
        # Global RNGs should not affect dataset.rng directly, but we reset
        # them for determinism in any helper code.
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        train_loader, _ = D.build_dataloaders(
            config
        )
        seen = []
        for batch in train_loader:
            if not batch:
                continue
            seen.append(
                (
                    int(
                        batch["rand_val"][
                            0
                        ].item()
                    ),
                    int(
                        batch[
                            "worker_id"
                        ][0].item()
                    ),
                )
            )

        # Best-effort cleanup of workers to avoid resource leaks on some platforms.
        with contextlib.suppress(
            Exception
        ):
            if (
                hasattr(
                    train_loader,
                    "_iterator",
                )
                and train_loader._iterator
                is not None
            ):
                train_loader._iterator._shutdown_workers()

        return seen

    seq1 = _collect()
    seq2 = _collect()

    # 1) Full loader determinism across runs.
    assert (
        seq1 == seq2
    ), "Sequences differ despite identical seeds with multi-worker loader."

    # 2) Group values by worker to inspect per-worker streams.
    by_worker1 = defaultdict(list)
    for rand_val, wid in seq1:
        by_worker1[wid].append(rand_val)

    by_worker2 = defaultdict(list)
    for rand_val, wid in seq2:
        by_worker2[wid].append(rand_val)

    # Per-worker sequences should also be deterministic across runs.
    assert (
        by_worker1 == by_worker2
    ), "Per-worker RNG streams changed between identical seeded runs."

    worker_ids = set(by_worker1.keys())
    assert (
        len(worker_ids) >= 2
    ), "Expected multiple workers to contribute distinct batches."
    assert (
        -1 not in worker_ids
    ), "Got worker_id=-1; DataLoader did not use multiple worker processes as expected."

    # 3) At least two workers should have different RNG sequences.
    workers = sorted(worker_ids)
    if len(workers) >= 2:
        w0, w1 = workers[0], workers[1]
        assert (
            by_worker1[w0] != by_worker1[w1]
        ), (
            "Worker RNG streams appear identical between workers; "
            "expected distinct sequences when seeding by worker_id."
        )
