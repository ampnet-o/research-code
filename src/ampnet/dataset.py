from typing import Callable, Iterable
from .loader import AudioFolder
from .utils.audio import parse_idmt, get_num_frames, gen_fuzzy_split_fn

from pathlib import Path

import numpy as np

# --------------------------------------------------

class PairedAudioFolder:
    def __init__(
            self,
            ds_path: str,
            src_folder = "NoFX",
            filter_f = None,
            frame_size = 4096,
            stride = 256,
            batch_size = 128,
            seed = 42,
            debug = False,
            lazy = False,
            split_spec = (0.8, 0.1, 0.1),
            split = 0
        ) -> None:

        if not Path(ds_path).exists():
            raise ValueError(f"invalid dataset path: {ds_path}")

        self.stride = stride
        self.frame_size = frame_size
        self.batch_size = batch_size
        self.debug = debug
        self.cwd = ds_path
        self.af = AudioFolder(ds_path, lazy=lazy)

        # lower level, impl specific
        self.frame_idx_dtype = np.uint32
        self._rng = np.random.default_rng(seed)

        split_fn = gen_fuzzy_split_fn(self._rng, split_spec, split)
        if filter_f is None:
            self._split_fn = split_fn
        else:
            # M = file metadata
            self._split_fn = lambda M: filter_f(M) and split_fn(M)

        self.fpairs = self._build_fpairs(src_folder, filter_f)
        self.frame_pairs = self._build_frame_index()
        self._shuffle_fr_pairs()

        if self.debug:
            self._idxck()
            print("consistency checks passed")

    # --------------------------------------------------
    #                    porcelain
    # --------------------------------------------------

    def get_num_batches(self):
        """
        batch-wise length
        """
        return len(self.frame_pairs) // self.batch_size

    def get_num_files(self):
        """
        number of indexed (!!) files
        """
        return len(self.fpairs.keys()) + sum(len(x[1]) for x in self.fpairs.values())

    # indexers
    def get_fr_batch(self, i: int, enumerate=False):
        """
        Get frame batch at a specific index

        return:
            - afids (opt), (src, wet)
        """
        idx = self.get_fr_batch_i(i)
        src = self.af.get_frame(idx[:, 0], idx[:, 2], self.frame_size)
        wet = self.af.get_frame(idx[:, 1], idx[:, 2], self.frame_size)

        if enumerate:
            return idx, (src, wet)

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
            if len(v) == 0:
                raise ValueError(f"invalid pair (unpaired): {k}")

            # src, wet signals
            yield v

    def iter_fr_pairs(self, split=0):
        """
        iterate over frame pairs
        """

        for i in range(
            0,
            len(self.frame_pairs),
            self.batch_size
        ):
            yield self.get_fr_batch(i)

    # --------------------------------------------------
    #                      utils
    # --------------------------------------------------

    def _shuffle_fr_pairs(self):
        self._rng.shuffle(self.frame_pairs)

    # --------------------------------------------------
    #                     plumbing
    # --------------------------------------------------

    def _build_fpairs(self, src_folder, filter_fn: None | Callable):
        """
        make src -> wet file pairs index. One time call
        """
        if filter_fn is None: filter_fn = self._split_fn

        # audiofolder id -> sample id by fname
        src_afids = self.af.get_dir_samples_i(src_folder)
        pairs = {}

        self._update_idmt2afid(
            pairs,
            (x for x in src_afids if filter_fn(self.af.get_metadata(x))),
            init=True
        )


        for dir in (self.af.get_dirs() - {src_folder}):
            wet_afids = self.af.get_dir_samples_i(dir)
            self._update_idmt2afid(
                pairs,
                (x for x in wet_afids if filter_fn(self.af.get_metadata(x)))
            )

        # final filter for unpaired sources
        return {k: pairs[k] for k in pairs if len(pairs[k][1]) > 0}

    def _update_idmt2afid(self, pairs: dict, afids: Iterable, init=False):
        for afid in afids:
            meta = self.af.get_metadata(afid)

            # guitar id, note id, effect id
            gid, nid, eid, _ = parse_idmt(meta["file"])
            idmt_id = f"{gid}:{nid}"

            if init:
                pairs[idmt_id] = (afid, [])
            elif idmt_id in pairs:
                pairs[idmt_id][1].append(afid)

        return pairs


    # NOTE this assumes quite a bit:
    # - src len == wet len
    # - non-empty file set
    # - src -> wet pairs are constant wrt sources (???)
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


