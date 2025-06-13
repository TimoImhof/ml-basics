from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np


@dataclass
class BytePairEncoder_Config:
    vocab: Set[str] = field(default_factory=set)
    vocab_size: int = 37000
    eow_token: str = "</w>"


class BytePairEncoder:
    def __init__(self, config: BytePairEncoder_Config):
        self.config = config
        self.vocab = config.vocab
        self.tokenized_corpus = []
        self.learned_merges = []
        self.vocab_mapping = {}

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

    def _apply_merge(self, new_char: str) -> List[str]:
        len_tokenized_corpus = len(self.tokenized_corpus)
        idx = 0
        while idx + 1 < len_tokenized_corpus:
            candiate_merged_token = (
                self.tokenized_corpus[idx] + self.tokenized_corpus[idx + 1]
            )
            if candiate_merged_token == new_char:
                self.tokenized_corpus[idx] = candiate_merged_token
                del self.tokenized_corpus[idx + 1]
                len_tokenized_corpus -= 1
            idx += 1

    def _find_max_pair(self, pairs: dict[Tuple, int]) -> Optional[Tuple]:
        pairs_sorted_desc = sorted(pairs, key=lambda x: pairs[x], reverse=True)
        for pair in pairs_sorted_desc:
            if self.config.eow_token in pair:
                continue
            return pair
        return

    def _apply_char_lvl_encoding(self, text: str):
        self.tokenized_corpus = []
        for sentence in text.split("."):
            for word in sentence.strip().split():
                tokens = list(word) + [
                    self.config.eow_token
                ]  # Ensure tokens cannot be merged across words
                self.tokenized_corpus.extend(tokens)

    def build_vocabulary(self, corpus: str):
        self._apply_char_lvl_encoding(corpus)
        self.vocab = set(self.tokenized_corpus)

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
                new_char = "".join(most_frequent_pair)
                self.vocab.add(new_char)
                self._apply_merge(new_char)

                progress_bar.update(1)

        for i, token in enumerate(self.vocab):
            self.vocab_mapping[token] = i

    def _convert_encoded_text_to_indices(self) -> np.ndarray:
        numeric_repr = []
        for token in self.tokenized_corpus:
            numeric_repr.append(self.vocab_mapping.get(token, self.config.vocab_size))
        return np.array(numeric_repr)

    def encode(self, text: str) -> np.ndarray:
        self._apply_char_lvl_encoding(text)
        for merge in self.learned_merges:
            new_char = "".join(merge)
            self._apply_merge(new_char)
        return self._convert_encoded_text_to_indices()
