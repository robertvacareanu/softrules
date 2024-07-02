"""
Taking a look at how the rules look like on TACRED
"""

import random
import json

from src.utils import line_to_hash
from src.baseline.entity_marker_cross_encoder import preprocess_line
from collections import defaultdict, Counter
from src.softrules.different_encoder.model_cliplike import read_rules


if __name__ == "__main__":

    train_path = "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED_full/train.json"

    rules_path = [
        '/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/surface.jsonl',
        '/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/enhanced_syntax.jsonl',
    ]

    with open(train_path) as fin:
        train = json.load(fin)


    rules = read_rules(rules_path)
    rules = dict(rules)

    # Get the rules over the sentences in training; Decide whether to include no relation or not
    train_rules = [y for line in train for y in rules[line_to_hash(line, use_all_fields=True)] if line['relation'] != 'no_relation']


    ddr = defaultdict(list)
    for x in train_rules:
        ddr[x['relation']].append(x['query'])

    for key in ddr.keys():
        random.seed(1)
        print("-"*20)
        print(f'"{key}"')
        for x in random.sample(ddr[key], k=min(10, len(ddr[key]))):
            print(f'"{x}"')
        print("-"*20)
        print("\n\n")