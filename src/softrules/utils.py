from typing import Dict
import json
from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.utils import line_to_hash

def read_fsre_data(path: str, rules: Dict[str, dict], preprocessing_type: str, **kwargs):
    """
    Read data in the style of the few-shot relation extraction datasets (e.g. `_train_data.json` from Few-Shot TACRED)
    """
    with open(path) as fin:
        train_data = json.load(fin)
    
    data = []
    idx = -1
    for (relation, sentences) in train_data.items():
        for sentence in sentences:
            for rule in rules[line_to_hash(sentence, use_all_fields=True)]:
                idx += 1
                data.append({
                    'id': idx,
                    'rule': rule['query'].lower(),
                    'sentence': preprocess_line(sentence, preprocessing_type),
                })

    return data

def read_randomly_generated_data(path: str, preprocessing_type: str, **kwargs):
    """
    Read rules + sentences that were automatically generated (e.g. no relation associated, etc)
    """
    data = []
    idx = -1

    with open(path) as fin:
        for line in fin:
            idx += 1
            loaded_line = json.loads(line)
            preprocessed_line = preprocess_line(loaded_line, preprocessing_type)
            data.append({
                'id': idx,
                'rule': loaded_line['query'].lower(),
                'sentence': preprocess_line(loaded_line, preprocessing_type),
            })

    return data