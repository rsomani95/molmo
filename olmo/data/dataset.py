import aiohttp
import os
import warnings
from os.path import join

import datasets
from datasets import DownloadConfig
import numpy as np

if "MOLMO_DATA_DIR" in os.environ:
    DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "torch_datasets")
else:
    warnings.warn("MOLMO_DATA_DIR is not set, data loading might fail")
    DATA_HOME = None


# Comprehensive timeout configuration for fsspec/aiohttp downloads.
# Key insight: for large file downloads, we need to:
# 1. Disable `total` timeout (set to None) - this runs from request start and will
#    fail for large files regardless of download speed
# 2. Set generous `sock_read` timeout to detect stalls (not overall time)
# 3. Set fsspec's own timeout parameter as backup
_AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=None,       # Disable total timeout - large files can take arbitrarily long
    connect=300,      # 5 min to establish connection
    sock_connect=300, # 5 min for socket connection
    sock_read=600     # 10 min between data chunks - detects stalls, not total time
)

STORAGE_OPTIONS = {
    'timeout': 3600,  # fsspec-level timeout (backup)
    'client_kwargs': {
        'timeout': _AIOHTTP_TIMEOUT
    }
}

# DownloadConfig for HuggingFace datasets with extended timeouts
DOWNLOAD_CONFIG = DownloadConfig(
    storage_options=STORAGE_OPTIONS,
    max_retries=5,
)


class Dataset:
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get(item, np.random)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, item, rng):
        # `rng` is used to support deterministic data augmentation for tasks that require it.
        # Used to avoid the hazards of relying on the global rng state for determinism
        raise NotImplementedError()


class DeterministicDataset:
    """Dataset wrapper that supports padding and control the random seed based on the epoch"""

    def __init__(self, dataset: Dataset, preprocessor, seed, n_pad=0):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_pad = n_pad

    def __len__(self):
        return len(self.dataset) + self.n_pad

    def __getitem__(self, idx):
        return self.get(idx, 0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, idx, epoch=0):
        rng = np.random.RandomState(self.seed + idx + len(self.dataset)*epoch)
        if idx >= len(self.dataset):
            # Padding example
            item = self.dataset.get(0, rng)
            if "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["valid"] = False
        else:
            item = self.dataset.get(idx, rng)
        if self.preprocessor:
            item = self.preprocessor(item, rng)
        return item


class DatasetBase(Dataset):
    def __init__(self, split, sample: int=None):
        super().__init__()
        self.split = split
        self.sample = sample
        self.data = self.load()[:self.sample]

    def load(self):
        raise NotImplementedError()

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def get(self, item, rng):
        raise NotImplementedError()


class HfDataset(Dataset):
    PATH = None

    @classmethod
    def download(cls, n_procs=None):
        datasets.load_dataset_builder(cls.PATH).download_and_prepare(
            download_config=DOWNLOAD_CONFIG
        )

    def __init__(self, split: str, keep_in_memory=True, **kwargs):
        self.split = split
        # Remove any timeout-related kwargs to ensure our DOWNLOAD_CONFIG is used
        kwargs.pop('download_config', None)
        kwargs.pop('storage_options', None)
        self.dataset = datasets.load_dataset(
            self.PATH, split=split, keep_in_memory=keep_in_memory,
            download_config=DOWNLOAD_CONFIG,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)
