from typing import List
import bpe_logic
from bpe_logic import BytePairEncoderConfig


class BPEWorker:
    def __init__(self, config: BytePairEncoderConfig):
        self.config = config

    def create_char_lvl_encoding(self, text: str) -> List[str]:
        return bpe_logic.apply_char_lvl_encoding_on_str(text, self.config.eow_token)

    def count_pairs(self, tokenized_text: List[str]) -> List[str]:
        return bpe_logic.count_pairs_in_list_of_str(tokenized_text)

    def apply_merge_rule(self, tokenized_text: List[str], merge_rule: str) -> List[str]:
        return bpe_logic.apply_merge_rule_on_list_of_str(tokenized_text, merge_rule)
