from multiprocessing import Value
from typing import Callable
from scipy.io import wavfile
import numpy as np

import os
from pathlib import Path

# local
from .utils.audio import get_wav_meta


class AudioFolder:
    """
    General loader for audio stored in a filesystem
    """
    path: Path
    frame_size: int
    classes: list[str]
    sample_metadata: list[dict]

    def __init__(
        self,
        path: str,
        audio_dtype = np.int16,
        lazy: bool = True,
        transform: Callable | None = None
    ) -> None:
        self.path = Path(path)
        self.sample_folders = {}
        self.audio_dtype = audio_dtype

        self._data = np.array([])
        self._metadata = []
        self._build_index()

        if lazy:
            return

        for i in range(self.__len__()):
            self._load_sample(i)

    def __len__(self):
        return self._N_samples

    def get_dirs(self):
        """
        Get dirs indexed by audio folder
        """

        return set(self.sample_folders.keys())

    # TODO: let absolute path also work
    def get_dir_samples_i(self, dir: str):
        """
        Retrieve sample ids by folder
        """
        if dir not in self.sample_folders:
            raise KeyError(f"folder not found: {dir}")
        return self.sample_folders[dir]

    # --------------------------------------------------

    def get_num_frames(self, sample_i, frame_size = 1024, stride=1):
        assert stride >= 1

        start = abs(self._data_offsets[sample_i])
        end = abs(self._data_offsets[sample_i + 1])
        sample_len = end - start

        return sample_len - frame_size

    def get_sample(self, sample_i):
        self._update_file_cache(sample_i)

        start = abs(self._data_offsets[sample_i])
        end = abs(self._data_offsets[sample_i + 1])

        return self._data[start: end], self.get_metadata(sample_i)

    def get_metadata(self, sample_i):
        return self._metadata[sample_i]

    def get_frame(self, sample_i, frame_i, frame_size: int):
        self._update_file_cache(sample_i)

        # TODO: frame bounds check
        frame_view = np.lib.stride_tricks.sliding_window_view(
            self._data,
            window_shape=frame_size
        )

        return frame_view[self._data_offsets[sample_i] + frame_i]

    # --------------------------------------------------

    # cacheless debug fn
    def _get_sample_nc(self, sample_i):
        start = abs(self._data_offsets[sample_i])
        end = abs(self._data_offsets[sample_i + 1])

        return self._data[start: end], self.get_metadata(sample_i)

    # --------------------------------------------------

    def _build_index(self):
        f_count = 0
        for d, _, files in os.walk(self.path):
            if not self._valid_folder(d, files):
                continue

            d_relative = str(Path(d).relative_to(self.path))
            if d_relative not in self.sample_folders:
                self.sample_folders[d_relative] = []

            added_meta = [
                {"folder": d_relative, "file": f} for f in files if self._is_audio(f)
            ]
            self.sample_folders[d_relative].extend(
                x for x in range(f_count, f_count + len(added_meta))
            )
            self._metadata.extend(added_meta)
            f_count += len(added_meta)

        assert f_count == len(self._metadata)
        self._N_samples = f_count
        self._build_dat_index()

    def _build_dat_index(self):
        """
        initialize packed arr for audio data
        """
        data_offsets = [0]

        for i in range(len(self._metadata)):
            sample_path = self._get_sample_path(i)
            meta = get_wav_meta(sample_path)

            if meta["subtype"] != self.audio_dtype:
                raise ValueError(
                    f"Audio sample ({sample_path.name}) - ({meta["subtype"]}) has unexpected bit depth: {meta["subtype"]}"
                )

            self._metadata[i].update(meta)

            # BACK: max(1, get_size(meta))
            n_samples = max(1, meta["num_samples"])
            data_offsets.append(data_offsets[-1] + n_samples)

        assert len(data_offsets) > 1

        # alloc
        data = np.empty((data_offsets[-1],), dtype=self.audio_dtype)

        self._data = data
        self._data_offsets = -np.array(data_offsets)

    # TODO:
    def _get_sample_size(metadat: dict):
        pass

    # ----------------File Loading----------------------

    def _get_sample_path(self, i):
        if i < 0 or i >= self._N_samples:
            raise IndexError(f"sample index out of bounds: {i}")

        m = self._metadata[i]

        fd = m["folder"]
        fname = m["file"]


        return self.path / fd / fname

    def _sample_is_loaded(self, i):
        if i < 0 or i >= self._N_samples:
            raise IndexError(f"sample index out of bounds: {i}")

        return self._data_offsets[i + 1] > 0

    def _load_sample(self, i):
        if self._sample_is_loaded(i):
            return

        sample_path = self._get_sample_path(i)
        _, dat = wavfile.read(sample_path)

        start = abs(self._data_offsets[i])
        end = abs(self._data_offsets[i + 1])
        self._data[start: end] = dat

        self._data_offsets[i + 1] = end

    def _update_file_cache(self, sample_i: int | np.ndarray):
        if type(sample_i) == int:
            sample_i = np.array([sample_i])

        for s in np.unique(sample_i):
            self._load_sample(int(s))

    # --------------------------------------------------

    @staticmethod
    def _is_audio(f: str | list[str]):
        ext = ('.wav',)
        if type(f) == str:
            return f.lower().endswith(ext)

        return all(fi.lower().endswith(ext) for fi in f)

    def _valid_folder(self, d: str, files: list):
        return (
            len(files) > 0 and
            Path(d).resolve() != self.path.resolve()
        )


if __name__ == "__main__":
    DS_PATH = r"/home/roy/m/phat/dsets/guitar_merged"
    d = AudioFolder(DS_PATH)
    r = d.get_sample(10)

    print("finished")
    print(r)

