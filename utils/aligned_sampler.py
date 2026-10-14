"""Shared shuffling for the per-modality training loaders of the fusion trainers.

The fusion trainers build one DataLoader per modality and iterate over
zip(*loaders). With shuffle=True every loader drew its own permutation, so the
modalities in a training batch belonged to different sequences and only the
first modality matched the label. All training loaders now share one sampler
that yields the same permutation to every loader within an epoch and a new
permutation in the next epoch.
"""
import torch
from torch.utils.data import DataLoader, Sampler


class AlignedRandomSampler(Sampler):
    def __init__(self, n_items, n_loaders, seed):
        self.n, self.k, self.seed = n_items, n_loaders, seed
        self.calls = 0

    def __iter__(self):
        epoch = self.calls // self.k          # zip(*loaders) calls iter() once per loader per epoch
        self.calls += 1
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())

    def __len__(self):
        return self.n


def align_train_loaders(loaders, seed=None):
    n = len(loaders[0].dataset)
    assert all(len(l.dataset) == n for l in loaders), 'modality datasets differ in length'
    if seed is None:
        seed = torch.initial_seed() % (2 ** 31)
    sampler = AlignedRandomSampler(n, len(loaders), seed)
    return [DataLoader(l.dataset, batch_size=l.batch_size, sampler=sampler,
                       drop_last=l.drop_last, num_workers=l.num_workers,
                       pin_memory=l.pin_memory, worker_init_fn=l.worker_init_fn)
            for l in loaders]
