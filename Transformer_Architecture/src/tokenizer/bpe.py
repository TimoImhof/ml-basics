from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray


@dataclass
class BytePairEncoder_Config:
    vocab: Set[str] = field(default_factory=set)
    vocab_size: int = 37000
    end_of_word_token: str = "</w>"
    out_of_vocab_idx: int = 0


class BytePairEncoder:
    def __init__(self, config: BytePairEncoder_Config):
        self.config = config
        # bpe vocabulary building
        self.vocab = config.vocab
        self.tokenized_corpus = []
        self.learned_merges = []
        # encoding and decoding mappings
        self.mapping_token_to_idx = {}
        self.mapping_idx_to_token = {}

    def _find_pairs(self) -> dict[Tuple, int]:
        pairs = defaultdict(int)
        len_tokenized_corpus = len(self.tokenized_corpus)
        idx = 0
        while idx < len_tokenized_corpus - 1:
            candidate_pair = tuple(
                [self.tokenized_corpus[idx], self.tokenized_corpus[idx + 1]]
            )
            pairs[candidate_pair] += 1
            idx += 1
        return pairs

    def _apply_merge_to_input(self, new_char: str, input: list) -> List[str]:
        len_input = len(input)
        idx = 0
        while idx + 1 < len_input:
            candiate_merged_token = input[idx] + input[idx + 1]
            if candiate_merged_token == new_char:
                input[idx] = candiate_merged_token
                del input[idx + 1]
                len_input -= 1
            idx += 1
        return input

    def _find_max_pair(self, pairs: dict[Tuple, int]) -> Optional[Tuple]:
        pairs_sorted_desc = sorted(pairs, key=lambda x: pairs[x], reverse=True)
        for pair in pairs_sorted_desc:
            if self.config.end_of_word_token in pair:
                continue
            return pair
        return

    def _apply_char_lvl_encoding(self, text: str) -> list:
        encoded_text = []
        for char in text:
            if char == " ":
                encoded_text.append(self.config.end_of_word_token)
            else:
                encoded_text.append(char)
        return encoded_text

    def build_vocabulary(self, corpus: str):
        self.tokenized_corpus = self._apply_char_lvl_encoding(corpus)
        self.vocab = set(self.tokenized_corpus)

        # Create mapping for base vocabulary
        mapping_idx = 1
        for tok in list(self.vocab):
            self.mapping_idx_to_token[mapping_idx] = tok
            self.mapping_token_to_idx[tok] = mapping_idx
            mapping_idx += 1

        with tqdm(
            initial=len(self.vocab),
            total=self.config.vocab_size,
            desc="Vocabulary size",
        ) as progress_bar:

            while len(self.vocab) < self.config.vocab_size:
                pairs = self._find_pairs()
                if not pairs:
                    break
                most_frequent_pair = self._find_max_pair(pairs)
                if most_frequent_pair is None:
                    break
                self.learned_merges.append(most_frequent_pair)
                new_token = "".join(most_frequent_pair)
                self.vocab.add(new_token)
                self._apply_merge_to_input(new_token, self.tokenized_corpus)

                # Extend mapping for merges
                self.mapping_idx_to_token[mapping_idx] = new_token
                self.mapping_token_to_idx[new_token] = mapping_idx
                mapping_idx += 1

                progress_bar.update(1)

    def encode(self, text: str) -> np.ndarray[int]:
        char_lvl_encoded_text = self._apply_char_lvl_encoding(text)
        for merge in self.learned_merges:
            new_char = "".join(merge)
            char_lvl_encoded_text = self._apply_merge_to_input(
                new_char, char_lvl_encoded_text
            )

        return np.array(
            [
                self.mapping_token_to_idx.get(tok, self.config.out_of_vocab_idx)
                for tok in char_lvl_encoded_text
            ]
        )

    def decode(self, input_ids: NDArray[np.int32]) -> str:
        decoded_text = ""
        for id in input_ids:
            decoded_token = self.mapping_idx_to_token[id]
            if decoded_token == self.config.end_of_word_token:
                decoded_text += " "
            else:
                decoded_text += decoded_token
        return decoded_text
