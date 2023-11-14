"""
Re-create a few-shot eval file to be used to create episodes.
Advantage: only keep the ones without noise
Disadvantage: might reduce the data from 
"""

import json

import pandas as pd

import tqdm

from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.utils import line_to_hash

from collections import Counter, defaultdict

def find_sub_list(sl,l):
    """
    From https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
    """
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

if __name__ == "__main__":
    with open('src/isolated_experiments/2_NYT_with_manual_annotations/data/arnor_dataset-2.0.0/train.json') as fin:
        train = []
        for line in fin:
            train.append(json.loads(line))
    with open('src/isolated_experiments/2_NYT_with_manual_annotations/data/arnor_dataset-2.0.0/dev.json') as fin:
        dev = []
        for line in fin:
            dev.append(json.loads(line))
    with open('src/isolated_experiments/2_NYT_with_manual_annotations/data/arnor_dataset-2.0.0/test.json') as fin:
        test = []
        for line in fin:
            test.append(json.loads(line))
    with open('src/isolated_experiments/2_NYT_with_manual_annotations/data/arnor_dataset-2.0.0/test_noise.json') as fin:
        test_noise = []
        for line in fin:
            test_noise.append(json.loads(line))

    data = train + dev + test + test_noise

    data[0]

    data_unrolled = []
    for example in data:
        entity_to_label = {e['text']:e['label'] for e in example['entityMentions']}
        token = example['sentText'].split()
        for relationMention in example['relationMentions']:
            if 'is_noise' not in relationMention:
                continue
            subj_start, subj_end = find_sub_list(relationMention['em1Text'].split(), token)[0]
            subj_type = entity_to_label[relationMention['em1Text']]
            obj_start, obj_end   = find_sub_list(relationMention['em2Text'].split(), token)[0]
            obj_type = entity_to_label[relationMention['em2Text']]
            line = {
                'token'     : token,
                'subj_start': subj_start,
                'subj_end'  : subj_end,
                'obj_start' : obj_start,
                'obj_end'   : obj_end,
                'subj_type' : subj_type,
                'obj_type'  : obj_type,
                'is_noise'  : relationMention['is_noise'],
                'relation'  : relationMention['label'] if relationMention['label'].lower() != 'none' else 'no_relation',
            }
            data_unrolled.append({**line, 'line_to_hash': line_to_hash(line)})
            

    print(len(data_unrolled))
    print(len([x for x in data_unrolled if x['is_noise'] is False]))
    print(Counter([x['relation'] for x in data_unrolled if x['is_noise'] is False]))

    data_unrolled_unique = {x['line_to_hash']: x for x in data_unrolled}
    data_unrolled_unique = [x[1] for x in data_unrolled_unique.items()]
    print(len(data_unrolled))
    print(len([x for x in data_unrolled if x['is_noise'] is False]))
    print(Counter([x['relation'] for x in data_unrolled if x['is_noise'] is False]))
    exit()
    with open('/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/few_shot_data/_train_data.json') as fin:
        train_data = json.load(fin)

    data_unrolled = []
    for example in data:
        entity_to_label = {e['text']:e['label'] for e in example['entityMentions']}
        token = example['sentText'].split()
        for relationMention in example['relationMentions']:
            if 'is_noise' not in relationMention:
                continue
            subj_start, subj_end = find_sub_list(relationMention['em1Text'].split(), token)[0]
            subj_type = entity_to_label[relationMention['em1Text']]
            obj_start, obj_end   = find_sub_list(relationMention['em2Text'].split(), token)[0]
            obj_type = entity_to_label[relationMention['em2Text']]
            line = {
                'token'     : token,
                'subj_start': subj_start,
                'subj_end'  : subj_end,
                'obj_start' : obj_start,
                'obj_end'   : obj_end,
                'subj_type' : subj_type,
                'obj_type'  : obj_type,
                'is_noise'  : relationMention['is_noise'],
                'relation'  : relationMention['label'] if relationMention['label'].lower() != 'none' else 'no_relation',
            }
            data_unrolled.append({**line, 'line_to_hash': line_to_hash(line)})


    noise_free_data = defaultdict(list)
    for x in data_unrolled:
        if x['is_noise'] is False:
            relation = x['relation'] if x['relation'] != 'None' else 'no_relation'
            line = {**x, 'relation': relation}
            noise_free_data[relation].append(line)
    print(noise_free_data.keys())
    print(train_data.keys())
    print(sum([len(x[1]) for x in train_data.items()]))
    print(sum([len(x[1]) for x in train_data.items() if x[0] not in noise_free_data or x[0] == 'no_relation']))

    noise_free_data = dict(noise_free_data)
    print("############")
    new_train_data = defaultdict(list)
    for x in tqdm.tqdm(train_data.items()):
        if x[0] not in noise_free_data or x[0] == 'no_relation':
            for item in x[1]:
                if item not in data_unrolled:
                    new_train_data[x[0]].append(item)

    new_train_data = dict(new_train_data)
    # print("############")
    # with open('src/isolated_experiments/2_NYT_with_manual_annotations/resulting_data/_new_train_data.json', 'w+') as fout:
    #     json.dump(new_train_data, fout)

    # with open('src/isolated_experiments/2_NYT_with_manual_annotations/resulting_data/_new_eval_data.json', 'w+') as fout:
    #     json.dump(noise_free_data, fout)

