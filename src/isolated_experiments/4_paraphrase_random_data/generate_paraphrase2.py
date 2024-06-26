"""
Use one of the Open AI models to generate paraphrasings


"""

import argparse

from string import Template

import os
import time
import json
import multiprocessing
import tqdm
import random
from collections import defaultdict, Counter
from langchain.chat_models import ChatOpenAI
import pickle

template = Template(
"""Please generate a number of $HOW_MANY paraphrases for the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generations: "$ENTITY1", "$ENTITY2". 
Please be concise.
```
$TEXT
```
1. """
)


llm = ChatOpenAI(temperature=0, request_timeout=60, max_retries=10)

def work_fn(example):
    # time.sleep(random.random())
    text = llm.call_as_llm(example)
    # time.sleep(random.random())
    text = "1. " + text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate paraphrases")
    parser.add_argument('--dataset_path', type=str, required=True, help="The dataset to generate paraphrasings on.")
    # parser.add_argument('--save_path', type=str, required=True, help="Where to save the resulting dataset")
    # parser.add_argument('--number_of_paraphrases', type=int, required=False, default=3, help="How many paraphrases to generate.")

    args = vars(parser.parse_args())

    random.seed(1)


    data = [] # defaultdict(list)
    with open(args['dataset_path']) as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded = json.loads(line)
            # Filter out if it is too long (saving money)
            if len(loaded['token']) > 60:
                continue
            # Filter out if it is too short (not many paraphrases when it is too short)
            if len(loaded['token']) < 8:
                continue
            data.append(loaded)

    # data = [random.choice(y) for (x, y) in data.items()]
    # data = sorted([x['query'] for x in data])
    # random.shuffle(data)

    # data = data[:1_000_000]
    # exit()

    result_max_length = []
    work_data = []
    for line in data:
        entity1 = ' '.join(line['token'][line['subj_start']:(line['subj_end']+1)])
        entity2 = ' '.join(line['token'][line['obj_start']:(line['obj_end']+1)])
        text    = ' '.join(line['token'])
        # Proxy for complexity; When the entities are close, generate fewer paraphrases
        # than when they are far apart
        if 0 <= abs(line['obj_start'] - line['subj_end']) < 3:
            how_many = 2
        elif 3 <= abs(line['obj_start'] - line['subj_end']) < 5:
            how_many = 3
        elif 5 <= abs(line['obj_start'] - line['subj_end']) < 7:
            how_many = 4
        elif 7 <= abs(line['obj_start'] - line['subj_end']):
            how_many = 5
        else:
            raise ValueError(f"Distance value not handles: {abs(line['obj_start'] - line['subj_end'])} ({line['obj_start']}, {line['subj_end']}).")
        template_data = {
            'HOW_MANY': how_many,
            'ENTITY1' : entity1,
            'ENTITY2' : entity2,
            'TEXT'    : text,
        }
        result_max_length.append(how_many)
        work_data.append(template.substitute(**template_data))

    print(len(work_data))
    print(sum(result_max_length))
    # print(len(work_data))
    # print(Counter([len(x['token']) for x in data]))
    # print(work_data[0])
    # exit()
    all_result = []
    for (start, end) in [
        (0,      10000 ),
        (10000,  20000 ),
        (20000,  30000 ),
        (30000,  40000 ),
        (40000,  50000 ),
        (50000,  60000 ),
        (60000,  70000 ),
        (70000,  80000 ),
        (80000,  90000 ),
        (90000, 100000 ),
        # (100000, 110000 ),
        # (110000, 120000 ),
        # (120000, 130000 ),
        # (130000, 140000 ),
        # (140000, 150000 ),
        # (150000, 160000 ),
        # (160000, 170000 ),
        # (170000, 180000 ),
        # (180000, 190000 ),
        # (190000, 200000 ),

        # (200000, 210000 ),
        # (210000, 220000 ),
        # (220000, 230000 ),
        # (230000, 240000 ),
        # (240000, 250000 ),
        # (250000, 260000 ),
        # (260000, 270000 ),
        # (270000, 280000 ),
        # (280000, 290000 ),
        # (290000, 300000 ),
        
    ]:
        print(start, end)
        current_work_data = work_data[start:end]
        with multiprocessing.Pool(60) as p:
            result = list(tqdm.tqdm(p.imap(work_fn, current_work_data), total=len(current_work_data)))
        # result = [work_fn(x) for x in tqdm.tqdm(current_work_data)]
        
        print(result[0])
        print(result[1])

        with open(f'src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_1M_{start}_{end}.pickle', 'wb+') as fout:
            pickle.dump([data[start:end], current_work_data, result], fout)