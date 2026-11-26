"""Which GPUs have room, so `experiments run` can spread runs over them and configs need not pin a device.

Each run is started with ``CUDA_VISIBLE_DEVICES`` set to its GPU, so the parser's ``device: auto`` lands on it.
"""

from __future__ import annotations

import subprocess


def gpu_memory() -> dict[int, tuple[int, int]]:
    """Free and total MiB per GPU, numbered as nvidia-smi does (PCI bus order); empty when nvidia-smi can't tell."""
    query = ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits']
    try:
        out = subprocess.run(query, capture_output=True, text=True, check=True, timeout=30).stdout
        rows = (line.split(',') for line in out.splitlines() if line.strip())
        return {int(index): (int(free), int(total)) for index, free, total in rows}
    except (OSError, subprocess.SubprocessError, ValueError):  # no driver, or '[N/A]' memory on MIG
        return {}


def free_gpus(min_free: float = 0.5) -> list[int]:
    """GPUs with at least ``min_free`` of their memory free, most free first."""
    memory = gpu_memory()
    roomy = [index for index, (free, total) in memory.items() if free >= min_free * total]
    return sorted(roomy, key=lambda index: -memory[index][0])
