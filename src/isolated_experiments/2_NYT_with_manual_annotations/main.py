import json

import pandas as pd

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
    sentences_for_relation = defaultdict(list)
    for line in data_unrolled_unique:
        sentences_for_relation[line['relation']].append(line)

    data_unrolled_unique[0]
    for (relation, sentences_for_relation) in sentences_for_relation.items():
        if relation == 'no_relation':
            continue
        data = [{'id': i, 'tokens': preprocess_line(line, preprocessing_type='typed_entity_marker')} for (i, line) in enumerate(sentences_for_relation)]
        
        df = pd.DataFrame(data)
        df['MS'] = ''
        df['RV'] = ''
        df[['MS', 'RV', 'id', 'tokens']].to_csv(f"results/231031/relations/{relation.split('/')[-1].lower()}.csv", index=False)