import json
import tqdm
import pickle
import nltk
from typing import List
from collections import defaultdict, Counter

def starts_with_any(string: str, starts: List[str]):
    for start in starts:
        if string.startswith(start):
            return True

    return False

# def find_sub_list(sl,l):
#     """
#     From src/isolated_experiments/2_NYT_with_manual_annotations
#     """
#     results=[]
#     sll=len(sl)
#     for ind in (i for i,e in enumerate(l) if e==sl[0]):
#         if all(x.lower().strip() in y.lower().strip() or y.lower().strip() in x.lower().strip() for (x, y) in zip(l[ind:ind+sll], sl)):
#             results.append((ind,ind+sll-1))
#             print([x.lower().strip() in y.lower().strip() or y.lower().strip() in x.lower().strip() for (x, y) in zip(l[ind:ind+sll], sl)])
#     return results

def find_sub_list(sl,l):
    """
    From https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
    """
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if len(sl) == len(l[ind:ind+sll]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll-1))
            # elif all(x.lower().strip() in y.lower().strip() or y.lower().strip() in x.lower().strip() for (x, y) in zip(l[ind:ind+sll], sl)):
            #     results.append((ind,ind+sll-1))

    return results
    
def do_train():
    # exit()
    dataset = [] #defaultdict(list)
    for (start, end) in [
        # (0,       1000 ),
        (0,       10000 ), (10000,   20000 ),  (20000,   30000 ), (30000,  40000 ), (40000,   50000 ), (50000,   60000 ), (60000,   70000 ), (70000,   80000 ), (80000,   90000 ), (90000,  100000 ),
        (100000, 110000 ), (110000, 120000 ), (120000, 130000 ), (130000, 140000 ), (140000, 150000 ), (150000, 160000 ), (160000, 170000 ), (170000, 180000 ), (180000, 190000 ), (190000, 200000 ),
        (200000, 210000 ), (210000, 220000 ), (220000, 230000 ), (230000, 240000 ), (240000, 250000 ),# (250000, 260000 ), (260000, 270000 ), (270000, 280000 ), (280000, 290000 ), (290000, 300000 ),
        # (300000, 310000 ), (310000, 320000 ), (320000, 330000 ), (330000, 340000 ), (340000, 350000 ), (350000, 360000 ), (360000, 370000 ), (370000, 380000 ), (380000, 390000 ), (390000, 400000 ),
        # (400000, 410000 ), (410000, 420000 ), (420000, 430000 ), (430000, 440000 ), (440000, 450000 ), (450000, 460000 ), (460000, 470000 ), (470000, 480000 ), (480000, 490000 ), (490000, 500000 ),
    ]:

        with open(f'src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_1M_{start}_{end}.pickle', 'rb') as fin:
            data, work_data, result = pickle.load(fin)
        for idx, (example, paraphrases) in tqdm.tqdm(enumerate(zip(data, result))):
            # Parse result
            paraphrases_split = paraphrases.split("\n")
            paraphrases_split = [x.strip() for x in paraphrases_split]
            paraphrases_split = [x for x in paraphrases_split if x != '']
            paraphrases_split = [x[3:] if starts_with_any(x, ["1. ", "2. ", "3. ", "4. ", "5. ", ]) else x for x in paraphrases_split]
            paraphrases_split = [x[2:].strip() if starts_with_any(x, ["1:", "2:", "3:", "4:", "5:", ]) else x for x in paraphrases_split]

            # Relation name
            # if idx < -1:
                # relation_name = 'no_relation'
            # if idx in [24, 45, 3, 5, 10, 13, 26, 28, 31, 43, 50, 54, 55, 58, 60, 67, 72, 86, 96, 20, 36, 41, 68, 73, 92, 23, 1,]:
            #     relation_name = 'no_relation'
            # else:
            #     relation_name = f'relation{idx}'

            for paraphrase in paraphrases_split:
                tokens = nltk.word_tokenize(paraphrase)
                tokens = ' '.join(tokens)
                tokens = tokens.replace('http : ', 'http:').replace('http :', 'http:').replace('https : ', 'https:').replace('https :', 'https:').split(" ")
                subj = find_sub_list(example['token'][example['subj_start']:(example['subj_end']+1)], tokens)
                obj  = find_sub_list(example['token'][example['obj_start']:(example['obj_end']+1)], tokens)

                extra_details = {}
                # If there are multiple, keep the one closest to the original
                if len(subj) > 1:
                    subj_temp = [(x - example['subj_start'], y - example['subj_end']) for x, y in subj]
                    subj_temp = [(i, y-x) for (i, (x, y)) in enumerate(subj_temp)]
                    subj_temp = sorted(subj_temp, key=lambda x: x[1])
                    subj = [subj[subj_temp[0][0]]]
                    extra_details['heuristically_chosen_subj'] = True
                if len(obj) > 1:
                    obj_temp = [(x - example['obj_start'], y - example['obj_end']) for x, y in obj]
                    obj_temp = [(i, y-x) for (i, (x, y)) in enumerate(obj_temp)]
                    obj_temp = sorted(obj_temp, key=lambda x: x[1])
                    obj = [obj[obj_temp[0][0]]]
                    extra_details['heuristically_chosen_obj'] = True

                if len(subj) != 1 or len(obj) != 1:
                    # print(idx)
                    # print("-"*20)
                    # print(example)
                    # print(len(subj), len(obj))
                    # print(subj, obj)
                    # print(tokens)
                    # print(example['token'][example['subj_start']:(example['subj_end']+1)])
                    # print(example['token'][example['obj_start']:(example['obj_end']+1)])
                    # print("-"*20)
                    # print("\n\n")
                    # exit()
                    continue
                subj_start, subj_end = subj[0]
                obj_start, obj_end   = obj[0]

                # print(example)
                # print({**example, 'token': tokens, 'subj_start': subj_start, 'subj_end': subj_end, 'obj_start': obj_start, 'obj_end': obj_end, **extra_details, **{f'{k}_original': v for (k, v) in example.items()}})
                # exit()
                dataset.append({**example, 'token': tokens, 'subj_start': subj_start, 'subj_end': subj_end, 'obj_start': obj_start, 'obj_end': obj_end, **extra_details, **{f'{k}_original': v for (k, v) in example.items()}})
    
    # dataset = dict(dataset)
    # print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    # dataset = {k:v for (k, v) in dataset.items() if len(v) > 2}
    # print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    
    # with open('src/isolated_experiments/4_paraphrase_random_data/data/231125/enhanced_syntax_all_0_100.jsonl', 'w+') as fout:
        # for line in dataset:
            # _=fout.write(json.dumps(line))
            # _=fout.write('\n')
    return dataset

def episode_like_test():
    """
    Save the data in a format that can then be used to create episodes
    """
    with open('src/isolated_experiments/4_paraphrase_random_data/data/es_alltest_paraphrases.pickle', 'rb') as fin:
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
        # if idx in [24, 45, 3, 5, 10, 13, 26, 28, 31, 43, 50, 54, 55, 58, 60, 67, 72, 86, 96, 20, 36, 41, 68, 73, 92, 23, 1,]:
        #     relation_name = 'no_relation'
        # else:
        relation_name = f'relation{idx}'

        for paraphrase in paraphrases_split:
            tokens = paraphrase.split(' ')
            subj = find_sub_list(example['token'][example['subj_start']:(example['subj_end']+1)], tokens)
            obj  = find_sub_list(example['token'][example['obj_start']:(example['obj_end']+1)], tokens)
            if len(subj) != 1 or len(obj) != 1:
                # print(idx)
                continue
            subj_start, subj_end = subj[0]
            obj_start, obj_end   = obj[0]

            dataset[relation_name].append({**example, 'relation': relation_name, 'token': tokens, 'subj_start': subj_start, 'subj_end': subj_end, 'obj_start': obj_start, 'obj_end': obj_end})
    
    
    dataset = dict(dataset)
    # print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    # dataset = {k:v for (k, v) in dataset.items() if len(v) > 2}
    # print(Counter([y['relation'] for (a, b) in dataset.items() for y in b]))
    print(len([y for x in dataset.items() for y in x[1]]))
    
    # with open('src/isolated_experiments/4_paraphrase_random_data/data/_random_eval_data.json', 'w+') as fout:
        # json.dump(dataset, fout)

if __name__ == "__main__":
    dataset = do_train()
    # episode_like_test()

    with open('src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/enhanced_syntax_paraphrase_0_250k.jsonl', 'w+') as fout:
        for line in dataset:
            if ']+  [' in line['query_es']:
                print("It is")
            result_line = {**line, 'query': line['query_es'].replace(']+  [', ']+ [')}
            _=fout.write(json.dumps(result_line))
            _=fout.write('\n')

    with open('src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/surface_paraphrase_0_250k.jsonl', 'w+') as fout:
        for line in dataset:
            result_line = {**line, 'query': line['query_s'].replace(']+  [', ']+ [')}
            _=fout.write(json.dumps(result_line))
            _=fout.write('\n')
