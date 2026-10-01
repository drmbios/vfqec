"""Spatial CSS minimum-weight perfect matching via PyMatching."""

import numpy as np
import pymatching


class SurfaceDecoder:
    """Decode each noisy round spatially; this is not a spacetime decoder."""

    def __init__(self, code):
        self.nx = len(code.x_checks)
        self.x_matching = pymatching.Matching(code.check_matrix("Z"))
        self.z_matching = pymatching.Matching(code.check_matrix("X"))

    def decode(self, syndrome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.x_matching.decode_batch(syndrome[:, self.nx :]).astype(np.uint8),
            self.z_matching.decode_batch(syndrome[:, : self.nx]).astype(np.uint8),
        )


def decoder_for(code):
    from vfqec.decoders.lookup import RepetitionDecoder

    return RepetitionDecoder(code) if code.name == "repetition" else SurfaceDecoder(code)
