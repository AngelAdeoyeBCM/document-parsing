"""Which GPUs have room, asked when a run starts so configs need not pin a device."""

from __future__ import annotations

import os
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


def auto_device() -> str:
    """``cuda:N`` for the visible GPU with the most free memory; ``auto`` (Docling decides) when that is unknown."""
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')  # so CUDA numbers GPUs as nvidia-smi does
    memory = gpu_memory()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        try:
            order = [int(index) for index in visible.split(',') if index.strip()]
        except ValueError:  # UUIDs or MIG ids
            return 'auto'
        memory = {order.index(index): room for index, room in memory.items() if index in order}
    if not memory:
        return 'auto'
    return f'cuda:{max(memory, key=lambda index: memory[index][0])}'
