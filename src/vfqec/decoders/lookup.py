"""Minimum-weight repetition decoding, including a deterministic even-N tie break."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def lookup(n: int) -> np.ndarray:
    table = np.zeros(2 ** (n - 1), dtype=int)
    seen = set()
    for error in sorted(range(2**n), key=lambda v: (v.bit_count(), v)):
        syndrome = sum((((error >> i) ^ (error >> (i + 1))) & 1) << i for i in range(n - 1))
        if syndrome not in seen:
            table[syndrome] = error
            seen.add(syndrome)
    return table


class RepetitionDecoder:
    def __init__(self, code):
        self.code = code
        self.table = lookup(code.n)

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        index = syndrome @ (1 << np.arange(self.code.m))
        masks = self.table[index]
        correction = ((masks[:, None] >> np.arange(self.code.n)) & 1).astype(np.uint8)
        zeros = np.zeros_like(correction)
        return (correction, zeros) if self.code.basis == "Z" else (zeros, correction)
