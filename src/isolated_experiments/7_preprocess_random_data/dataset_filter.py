"""
Check the number of entity types, the number of entities
The idea is to have some sort of understanding of how the data looks like in order to improve it
"""

import json
import tqdm
import random
import numpy as np
from collections import Counter, defaultdict

def remove_full_duplicates(filepath, savepath):
    """
    Remove complete duplicates
    """
    rules_matches_tokens_and_entities_counts = defaultdict(int)
    rules_matches_tokens_and_entities_orig = {}
    i = 0
    with open(filepath) as fin:
        for line in tqdm.tqdm(fin):
            # i+=1
            # if i > 10_000_000:
            #     continue
            loaded = json.loads(line)

            key = (
                loaded['query'], 
                ' '.join(loaded['matched_tokens']), 
                ' '.join(loaded['token']),
                ' '.join(loaded['token'][loaded['subj_start']:(loaded['subj_end']+1)]),
                ' '.join(loaded['token'][loaded['obj_start']:(loaded['obj_end']+1)]), 
                loaded['subj_type'], 
                loaded['obj_type'],
                loaded['subj_start'],
                loaded['subj_end'],
                loaded['obj_start'],
                loaded['obj_end'],
            )

            rules_matches_tokens_and_entities_counts[key] += 1
            if rules_matches_tokens_and_entities_counts[key] > 1:
                assert(loaded == rules_matches_tokens_and_entities_orig[key])
            else:
                rules_matches_tokens_and_entities_orig[key] = loaded

    with open(savepath, 'w+') as fout:
        for line in tqdm.tqdm(rules_matches_tokens_and_entities_orig.values()):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


def subsample_rulesandentities(filepath, savepath, seed=1):
    """
    Sub-sample based on (<rule>, <entity1>, <entity2>, <entity1_type>, <entity2_type>)
    Keep only np.sqrt(count)
    """
    random.seed(seed)
    rules_and_entities = defaultdict(list)
    i = 0
    with open(filepath) as fin:
        for line in tqdm.tqdm(fin):
            # i+=1
            # if i > 10_000_000:
            #     continue
            loaded = json.loads(line)

            key = (
                loaded['query'],
                ' '.join(loaded['token'][loaded['subj_start']:(loaded['subj_end']+1)]),
                ' '.join(loaded['token'][loaded['obj_start']:(loaded['obj_end']+1)]), 
                loaded['subj_type'], 
                loaded['obj_type']
            )

            rules_and_entities[key].append(loaded)

    selected = []
    for x in rules_and_entities.values():
        if len(x) == 1:
            selected += x
        else:
            selected += random.sample(x, k=(int(np.sqrt(len(x))) + 1))

    with open(savepath, 'w+') as fout:
        for line in tqdm.tqdm(selected):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

def subsample_rules(filepath, savepath, seed=1):
    """
    Sub-sample based on (<rule>)
    Keep only np.sqrt(count)
    """
    random.seed(seed)
    rules = defaultdict(list)
    i = 0
    with open(filepath) as fin:
        for line in tqdm.tqdm(fin):
            # i+=1
            # if i > 10_000_000:
            #     continue
            loaded = json.loads(line)

            key = (
                loaded['query'],
            )

            rules[key].append(loaded)

    selected = []
    for x in rules.values():
        if len(x) == 1:
            selected += x
        else:
            selected += random.sample(x, k=(int(np.sqrt(len(x))) + 1))

    with open(savepath, 'w+') as fout:
        for line in tqdm.tqdm(selected):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

def subsample_entitytypes(filepath, savepath, seed=1):
    """
    Sub-sample based on (<entity1_type>, <entity2_type>)
    Keep only np.sqrt(count)
    """
    random.seed(seed)
    entities = defaultdict(list)
    i = 0
    with open(filepath) as fin:
        for line in tqdm.tqdm(fin):
            # i+=1
            # if i > 10_000_000:
            #     continue
            loaded = json.loads(line)

            key = (
                loaded['subj_type'], 
                loaded['obj_type']
            )

            entities[key].append(loaded)

    min_value = min([len(x) for x in entities.values()])
    selected = []
    for x in entities.values():
        if len(x) <= min_value:
            selected += x
        else:
            selected += random.sample(x, k=min(int(np.sqrt((len(x) - min_value) * min_value) + min_value), len(x)))

    with open(savepath, 'w+') as fout:
        for line in tqdm.tqdm(selected):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

if __name__ == "__main__":
    # Step 1: remove full duplicates
    remove_full_duplicates('/storage/rvacareanu/data/softrules/rules/random_231201/enhanced_syntax_all.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s1.jsonl')
    remove_full_duplicates('/storage/rvacareanu/data/softrules/rules/random_231201/surface_all.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s1.jsonl')
    # Step 2: 
    subsample_rulesandentities('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s1.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s2.jsonl')
    subsample_rulesandentities('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s1.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s2.jsonl')

    # Step 3: 
    subsample_rules('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s2.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s3.jsonl')
    subsample_rules('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s2.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s3.jsonl')

    # Step 4: 
    subsample_entitytypes('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s3.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s4.jsonl')
    subsample_entitytypes('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s3.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s4.jsonl')
    
    
    # remove_full_duplicates('/storage/rvacareanu/data/softrules/rules/random_231201/surface_all.jsonl', '/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s1.jsonl')
