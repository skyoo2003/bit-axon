"""Bit-Axon 3B: Minimal Bits, Maximal Impulse — sLLM engine for Apple Silicon."""

__version__ = "0.1.0"

from bit_axon.config import BitAxonConfig
from bit_axon.model import BitAxonModel
from bit_axon.tokenizer import QwenTokenizerWrapper

__all__ = [
    "BitAxonConfig",
    "BitAxonModel",
    "QwenTokenizerWrapper",
    "__version__",
]
