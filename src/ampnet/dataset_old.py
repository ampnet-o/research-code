from .loader import AudioFolder
from .utils.audio import parse_idmt, get_num_frames

import math
from pathlib import Path

import numpy as np

# --------------------------------------------------

class PairedAudioFolder:
    def __init__(
            self,
            ds_path,
            src_folder = "NoFX",
            dir_filter = [],
            dir_filter_mode = "blacklist", # or whitelist
            frame_size = 4096,
            stride = 256,
            batch_size = 128,
            seed = 42,
            debug = False,
            lazy = False,
            ) -> None:

        if not Path(ds_path).exists():
            raise ValueError(f"invalid dataset path: {ds_path}")

        self.stride = stride
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.debug = debug
        self.af = AudioFolder(ds_path, lazy=lazy)

        # lower level
        self.frame_idx_dtype = np.uint32
        self._rng = np.random.default_rng(seed)

        # impl specific
        self.fpairs = self._build_fpairs(src_folder, dir_filter, dir_filter_mode)
        self.frame_pairs = self._build_frame_index()
        self._shuffle_fr_pairs()

        if self.debug:
            self._idxck()
            print("consistency checks passed")

    # lengths
    def get_num_batches(self):
        """
        batch-wise length
        """
        return len(self.frame_pairs) // self.batch_size

    def get_num_files(self):
        """
        number of !indexed! files
        """
        return len(self.fpairs.keys()) + sum(len(x) for x in self.fpairs.values())

    # indexers
    def get_fr_batch(self, i: int):
        """
        Get frame batch at a specific index
        """
        idx = self.get_fr_batch_i(i)
        src = self.af.get_frame(idx[:, 0], idx[:, 2], self.frame_size)
        wet = self.af.get_frame(idx[:, 1], idx[:, 2], self.frame_size)
        return src, wet

    def get_fr_batch_i(self, i: int):
        """
        Get batch indexes at index
        """
        return self.frame_pairs[i:i+self.batch_size]

    # iterators
    def iter_fpairs_i(self):
        """
        iterate over file pairs by file id
        """
        for k in self.fpairs:
            v = self.fpairs[k]
            if len(v) <= 1:
                raise ValueError(f"invalid pair (too short): {k}")

            # src, wet signals
            yield v[0], v[1:]

    def iter_fr_pairs(self, split=0):
        """
        iterate over frame pairs
        """
        for i in range(
            0,
            len(self.frame_pairs) // self.batch_size,
            self.batch_size
        ):
            yield self.get_fr_batch(i)

    # --------------------------------------------------
    #                      utils
    # --------------------------------------------------

    def _shuffle_fr_pairs(self):
        self._rng.shuffle(self.frame_pairs)

    # --------------------------------------------------
    #                  impl specific
    # --------------------------------------------------

    def _build_fpairs(self, src_folder, dir_filter, filter_mode):
        """
        make src -> wet file pairs index
        """
        # audiofolder id -> sample id by fname
        pairs = {}
        sourced_pairs = []

        if len(dir_filter) == 0 and filter_mode == "blacklist":
            f_filter_check = lambda x: False
        elif filter_mode == "whitelist":
            f_filter_check = lambda x: x in set(dir_filter)
        else:
            f_filter_check = lambda x: x not in set(dir_filter)


        for afid in range(len(self.af)):
            meta = self.af.get_metadata(afid)
            sample_dir = meta["folder"]

            if sample_dir != src_folder and not f_filter_check(sample_dir):
                continue

            # guitar id, note id, effect id
            gid, nid, eid, _ = parse_idmt(meta["file"])

            src_id = f"{gid}:{nid}"

            if src_id not in pairs:
                pairs[src_id] = [afid]

            else:
                pairs[src_id].append(afid)

            if sample_dir == src_folder:
                sourced_pairs.append(src_id)
                arr = pairs[src_id]
                arr[-1], arr[0] = arr[0], arr[-1]

        # pigeonholing
        if self.debug:
            assert len(sourced_pairs) == len(pairs)
            assert len(set(sourced_pairs)) == len(pairs)

        return pairs


    # NOTE this assumes:
    # src len == wet len
    # non-empty file set
    # src -> wet pairs are constant wrt sources
    def _build_frame_index(self):
        N = self.get_num_files()
        N_samples = self.af.get_metadata(0)['num_samples']
        N_frames = get_num_frames(N_samples, self.frame_size, self.stride)

        # to be flattened (src fid, wet fid, frame idx)
        fr_idx = np.zeros((N, N_frames, 3), dtype=self.frame_idx_dtype)

        i = 0
        for src, wet in self.iter_fpairs_i():
            valid_frames = np.arange(
                0,
                N_frames*self.stride,
                self.stride,
                dtype=np.uint32
            )

            # TODO: this is really stupid
            for w in wet:
                _row = fr_idx[i, :]
                _row[:, 0] = src
                _row[:, 1] = w
                _row[:, 2] = valid_frames

                i += 1

        return fr_idx[:i].reshape((-1, 3))

    def _idxck(self):
        """
        consistency checks
        """

        # frame pair array shouldn't contain [0, 0, 0]
        zeros = np.where(0 == np.sum(self.frame_pairs, axis=1))
        assert len(zeros[0]) == 0


