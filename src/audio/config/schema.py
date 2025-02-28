from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    wavlm: List[str] = field(default_factory=lambda: [])
    wavlm: str = "microsoft/wavlm-base-plus-sv"

@dataclass
class RuntimeConfig:
    device: List[str] = field(default_factory=lambda: [])
    cuda_alloc_conf: str = "expandable_segments:True"

@dataclass
class RootConfig:
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig = ModelConfig()
