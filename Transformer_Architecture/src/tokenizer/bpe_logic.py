from typing import List, DefaultDict
from collections import defaultdict


def count_pairs_in_list_of_str(input_list: List[str]) -> DefaultDict[str, int]:
    pairs = defaultdict(int)
    for i in range(len(input_list) - 1):
        candidate_pair = tuple([input_list[i], input_list[i + 1]])
        pairs[candidate_pair] += 1
    return pairs


def apply_merge_rule_on_list_of_str(
    input_list: List[str], merge_rule: str
) -> List[str]:
    len_input = len(input_list)
    idx = 0
    while idx + 1 < len_input:
        merged_candidate = input_list[idx] + input_list[idx + 1]
        if merged_candidate == merge_rule:
            input_list[idx] = merged_candidate
            del input_list[idx + 1]
            len_input -= 1
        idx += 1
    return input_list


def apply_char_lvl_encoding_on_str(text: str, eow_token: str) -> List[str]:
    encoded_text = []
    for char in text:
        if char == " ":
            encoded_text.append(eow_token)
        else:
            encoded_text.append(char)
    return encoded_text
