"""
Check the number of entity types, the number of entities
The idea is to have some sort of understanding of how the data looks like in order to improve it
"""

import json
import tqdm
import numpy as np
from collections import Counter, defaultdict

if __name__ == "__main__":
    entity_types_counts = defaultdict(int)
    entities_counts = defaultdict(int)
    rules_and_entities_counts = defaultdict(int)
    with open('/storage/rvacareanu/data/softrules/rules/random_231201/enhanced_syntax_all.jsonl') as fin:
        for line in tqdm.tqdm(fin):
            loaded = json.loads(line)
            entity_types_counts[(loaded['subj_type'], loaded['obj_type'])] += 1
            key = (
                ' '.join(loaded['token'][loaded['subj_start']:(loaded['subj_end']+1)]),
                ' '.join(loaded['token'][loaded['obj_start']:(loaded['obj_end']+1)]), 
                loaded['subj_type'], 
                loaded['obj_type']
            )
            entities_counts[key] += 1
            rules_and_entities_counts[(loaded['query'], *key)] += 1
    
    print(entity_types_counts)
    print(sorted(entities_counts.items(), key=lambda x: -x[1])[:100])
    print(sorted(rules_and_entities_counts.items(), key=lambda x: -x[1])[:10])