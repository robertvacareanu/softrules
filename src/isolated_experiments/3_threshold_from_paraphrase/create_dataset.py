import json
import pickle
from typing import List
from collections import defaultdict, Counter

def starts_with_any(string: str, starts: List[str]):
    for start in starts:
        if string.startswith(start):
            return True

    return False

def find_sub_list(sl,l):
    """
    From src/isolated_experiments/2_NYT_with_manual_annotations
    """
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results



if __name__ == "__main__":
    with open('src/isolated_experiments/3_threshold_from_paraphrase/data/es_alltest_paraphrases.pickle', 'rb') as fin:
        data, work_data, result = pickle.load(fin)

    dataset = defaultdict(list)
    for idx, (example, paraphrases) in enumerate(zip(data, result)):
        # Parse result
        paraphrases_split = paraphrases.split("\n")
        paraphrases_split = [x.strip() for x in paraphrases_split]
        paraphrases_split = [x for x in paraphrases_split if x != '']
        paraphrases_split = [x[3:] if starts_with_any(x, ["1. ", "2. ", "3. ", "4. ", "5. ", ]) else x for x in paraphrases_split]
        paraphrases_split = [x[2:].strip() if starts_with_any(x, ["1:", "2:", "3:", "4:", "5:", ]) else x for x in paraphrases_split]

        # Relation name
        # if idx < -1:
            # relation_name = 'no_relation'
        if idx in [24, 45, 3, 5, 10, 13, 26, 28, 31, 43, 50, 54, 55, 58, 60, 67, 72, 86, 96, 20, 36, 41, 68, 73, 92, 23, 1,]:
            relation_name = 'no_relation'
        else:
            relation_name = f'relation{idx}'

        for paraphrase in paraphrases_split:
            tokens = paraphrase.split(' ')
            subj = find_sub_list(example['token'][example['subj_start']:(example['subj_end']+1)], tokens)
            obj  = find_sub_list(example['token'][example['obj_start']:(example['obj_end']+1)], tokens)
            if len(subj) != 1 or len(obj) != 1:
                print(idx)
                continue
            subj_start, subj_end = subj[0]
            obj_start, obj_end   = obj[0]

            dataset[relation_name].append({**example, 'relation': relation_name, 'token': tokens, 'subj_start': subj_start, 'subj_end': subj_end, 'obj_start': obj_start, 'obj_end': obj_end})
    
    
    dataset = dict(dataset)
    print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    dataset = {k:v for (k, v) in dataset.items() if len(v) > 2}
    print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    
    with open('src/isolated_experiments/3_threshold_from_paraphrase/data/_random_eval_data.json', 'w+') as fout:
        json.dump(dataset, fout)