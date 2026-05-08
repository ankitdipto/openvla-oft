"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Optional, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        action_token_begin_idx: Optional[int] = None,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.action_token_begin_idx: int = self._resolve_action_token_begin_idx(action_token_begin_idx)

    def _resolve_action_token_begin_idx(self, action_token_begin_idx: Optional[int]) -> int:
        if action_token_begin_idx is not None:
            return int(action_token_begin_idx)

        # MiniVLA Qwen checkpoints can reserve `<extra_0> ... <extra_255>` for
        # action prediction. If they exist contiguously, use that explicit range.
        inferred_extra_begin = self._infer_extra_action_token_begin_idx()
        if inferred_extra_begin is not None:
            return inferred_extra_begin

        # Legacy OpenVLA contract: overwrite the final `n_bins` tokens.
        return int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def _infer_extra_action_token_begin_idx(self) -> Optional[int]:
        convert_token = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert_token is None:
            return None

        extra_token_ids = []
        for idx in range(self.n_bins):
            token_id = convert_token(f"<extra_{idx}>")
            if token_id is None or token_id == self.tokenizer.unk_token_id or token_id < 0:
                return None
            extra_token_ids.append(int(token_id))

        expected_ids = list(range(extra_token_ids[0], extra_token_ids[0] + self.n_bins))
        if extra_token_ids != expected_ids:
            return None

        # Keep the legacy convention that valid action tokens are those with
        # ids strictly greater than `action_token_begin_idx`.
        return extra_token_ids[0] - 1

    @property
    def action_token_end_idx(self) -> int:
        return self.action_token_begin_idx + self.n_bins

    @property
    def stop_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def prompt_suffix_token_id(self) -> Optional[int]:
        # OpenVLA's Llama-family checkpoints expect an extra empty token before
        # action prediction. Qwen-style checkpoints do not.
        if self.tokenizer.__class__.__name__ == "LlamaTokenizerFast":
            return 29871
        return None

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            token_ids = list(self.action_token_begin_idx + (self.n_bins + 1 - discretized_action))
            return self.tokenizer.decode(token_ids)
        else:
            token_ids = self.action_token_begin_idx + (self.n_bins + 1 - discretized_action)
            return self.tokenizer.batch_decode(token_ids.tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.action_token_begin_idx + self.n_bins + 1 - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
