from typing import Callable
import numpy as np
import soundfile as sf

SUBTYPE_TO_NP = {
    'PCM_16': np.int16,      # 16-bit signed (common in WAV)
    'PCM_U8': np.uint8,      # 8-bit unsigned
    'PCM_24': np.int32,      # 24-bit is often read as 32-bit
    'PCM_32': np.int32,      # 32-bit signed
    'FLOAT': np.float32,     # 32-bit float
    'DOUBLE': np.float64,    # 64-bit float
}

def get_wav_meta(fpath) -> dict:
    """
    return number of samples
    """
    with sf.SoundFile(fpath) as f:
        n_samples = len(f)
        subtype = SUBTYPE_TO_NP[f.subtype]
        metadata = {
            "num_samples": len(f),             # Total number of samples
            "sample_rate": f.samplerate,       # Sample rate (Hz)
            "num_channels": f.channels,        # Number of channels (1=mono, 2=stereo)
            "duration": len(f) / f.samplerate, # Duration in seconds
            "subtype": subtype,                # Data format (e.g., 'PCM_16', 'FLOAT')
            "format": f.format,                # File format ('WAV', 'FLAC', etc.)
        }
        return metadata


def get_num_frames(N, frame_size, stride=1):
    """
    num. valid frames a sample of length N
    can yield
    """
    assert N >= frame_size
    return ((N - frame_size) // stride) + 1

# --------------------------------------------------

def gen_fuzzy_split_fn(
    rng: np.random.Generator,
    split_spec: tuple[float, ...],
    target_split: int
    ):
    """
    Construct a probabilistic selector for multi-way dataset splits.

    Args:
        rng: numpy random generator
        split_spec: tuple of split proportions, e.g. (0.8, 0.1, 0.1) for train/val/test
        target_split: which split to return True for (0-indexed)

    Returns:
        I(x in split)

    Example:
        # Create train split selector (80% of data)
        train_fn = _gen_fuzzy_split_fn(rng, (0.8, 0.1, 0.1), target_split=0)

        # Create validation split selector (10% of data)
        val_fn = _gen_fuzzy_split_fn(rng, (0.8, 0.1, 0.1), target_split=1)
    """
    # Validate inputs
    assert len(split_spec) > 0, "split_spec must contain at least one split"
    assert all(0 <= p <= 1 for p in split_spec), "All split proportions must be in [0, 1]"
    assert abs(sum(split_spec) - 1.0) < 1e-6, f"Split proportions must sum to 1.0, got {sum(split_spec)}"
    assert 0 <= target_split < len(split_spec), f"target_split must be in [0, {len(split_spec)-1}]"

    # Handle edge case: target split has 0 probability
    if split_spec[target_split] == 0:
        return lambda x: False

    # Handle edge case: target split has 1.0 probability
    if split_spec[target_split] == 1.0:
        return lambda x: True

    # Compute cumulative boundaries for each split
    # e.g., (0.8, 0.1, 0.1) -> boundaries at [0.8, 0.9, 1.0]
    cumulative = []
    total = 0.0
    for p in split_spec:
        total += p
        cumulative.append(total)

    # Define boundaries for target split
    lower_bound = cumulative[target_split - 1] if target_split > 0 else 0.0
    upper_bound = cumulative[target_split]

    def split_fn(dat):
        """Returns True if element belongs to target split."""
        rand_val = rng.uniform(0, 1)
        return lower_bound <= rand_val < upper_bound

    return split_fn


def gen_simple_dir_filter(filter_mode: str, flist: set):

    def filter_fn(dat):
        """
        true if selected
        """
        dirname = dat["folder"]

        if dirname == "NoFX":
            return True

        if filter_mode == "whitelist":
            return dirname in flist

        return dirname not in flist

    return filter_fn


def gen_processing_fn(data_fn: Callable, shape_fn: Callable):
    """
    Combine processing and preallocation function into one
    """
    def _draft_processing_fn(metadata: dict, data: np.ndarray | None = None, shape_only=False):
        if shape_only:
            return shape_fn(metadata["num_samples"])

        if data is None:
            raise ValueError("data needs to be defined for processing mode")

        return data_fn(data, metadata)

    return _draft_processing_fn

# --------------------------------------------------
#               IDMT specific stuff
# --------------------------------------------------

# As per IDMT file name format
def parse_idmt(fname):
    name = fname.split('.')[0]
    if len(name) == len(fname):
        raise ValueError(f"invalid file name format: fname")

    ids = name.split('-')
    if len(ids) != 4:
        raise ValueError(f"invalid file name format: fname")

    # (inst_id, note_id, effect_id, sample_id)
    return ids



