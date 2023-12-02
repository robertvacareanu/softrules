"""
Select data that will be used to generate paraphrases
"""

import datasets
from collections import defaultdict
import tqdm
import random
import json
import numpy as np

data1 = datasets.load_dataset('json', data_files=['/storage/rvacareanu/data/softrules/rules/random_231201/enhanced_syntax_all.jsonl'])['train']

entities_counts = defaultdict(int)
for i, x in tqdm.tqdm(enumerate(data1)):
    key = (' '.join(x['token'][x['subj_start']:(x['subj_end']+1)]),' '.join(x['token'][x['obj_start']:(x['obj_end']+1)]), x['subj_type'], x['obj_type'])
    entities_counts[key] += 1


data_es = defaultdict(list)
counts_es = defaultdict(int)

for i, x in tqdm.tqdm(enumerate(data1)):
    key = (x['subj_type'], x['obj_type'])
    counts_es[key] += 1
    data_es[key].append(i)

# We downsample to avoid certain entity types dominating the rephrasings
entity_types_to_probability = {k:np.sqrt(v**(2/3))/sum({k:np.sqrt(v**(2/3)) for (k, v) in counts_es.items()}.values()) for (k, v) in counts_es.items()}
entity_types_to_counts = {k:int(1_000_000 * v) for (k, v) in entity_types_to_probability.items()}

data_es_selected = []
keys = sorted(list(data_es.keys()))
for k in keys:
    # We always seed to 1 before sampling
    random.seed(1)
    data_es_selected += random.sample(data_es[key],)

data2 = datasets.load_dataset('json', data_files=['/storage/rvacareanu/data/softrules/rules/random_231201/surface_all.jsonl'])['train']
data_s = defaultdict(list)

for i, x in tqdm.tqdm(enumerate(data2)):
    key = (x['subj_type'], x['obj_type'])
    data_es[key].append(i)

data_filter = [x for x in data_merged.items() if len(x[1][0]) > 0 and len(x[1][1]) > 0]
random.shuffle(data_filter)

data_selected = data_filter[:2_000_000]

data_selected = [{**{k:v for (k, v) in data1[indices_d1[0]].items() if k != 'query'}, 'query_es': data1[indices_d1[0]]['query'], 'query_s': data2[indices_d2[0]]['query']} for (key, (indices_d1, indices_d2)) in tqdm.tqdm(data_selected)]

with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/6_merge_datasets/data/data_selected.jsonl', 'w+') as fout:
    for line in data_selected:
        _=fout.write(json.dumps(line))
        _=fout.write('\n')


data_filter_all = [x for x in data_merged.items() if len(x[1][0]) > 0 and len(x[1][1]) > 0]
data_filter_all = [{**{k:v for (k, v) in data1[indices_d1[0]].items() if k != 'query'}, 'query_es': data1[indices_d1[0]]['query'], 'query_s': data2[indices_d2[0]]['query']} for (key, (indices_d1, indices_d2)) in tqdm.tqdm(data_filter_all)]

# with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/6_merge_datasets/data/data_merged_all.jsonl', 'w+') as fout:
#     for line in data_selected:
#         _=fout.write(json.dumps(line))
#         _=fout.write(json.dumps(line))

