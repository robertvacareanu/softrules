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

# template = Template(
# """Please generate a paraphrase for the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generations: "$ENTITY1", "$ENTITY2". 
# Please be concise.
# ```
# $TEXT
# ```
# 1. """
# ) # 31/100
# template = Template(
# """Please generate a paraphrase for the following sentence. 
# Please ensure the meaning and the message stays the same.
# Additionally, please make sure that the following two entities are preserved as they are in your generations: "$ENTITY1" and "$ENTITY2". 
# Please be concise. You can find the text below.
# ```
# $TEXT
# ```
# 1. """
# ) # 28/100
# template = Template(
# """Please generate a paraphrase for the following sentence. Ensure the meaning and the message of the text stays the same in your output.
# A second extremely important condition is to ensure that the following two spans are preserved precisely as they are in your generation: "$ENTITY1" and "$ENTITY2". These two spans must appear as they are in your generation in order for it to be considered correct.
# Please be concise. You can find the text below.
# ```
# $TEXT
# ```
# 1. """
# ) # 4/100
# template = Template(
# """Please generate a paraphrase for the following sentence. Ensure the meaning and the message of the text stays the same in your output.
# A second extremely important condition is to ensure that the following two spans are preserved precisely as they are in your generation: "$ENTITY1" and "$ENTITY2". These two spans must appear as they are in your generation in order for it to be considered correct.
# Please be concise. You can find the text below.

# Text to be rephrased:
# ```
# $TEXT
# ```

# Paraphrase:
# ```
# """
# ) # 10/100
# template = Template(
# """Instruction: Please generate a paraphrase for the sentence that you will find below. Ensure the meaning and the message of the text stays the same in your output. A second extremely important condition is to ensure that the following two spans are preserved precisely as they are in your generation: "$ENTITY1" and "$ENTITY2". These two spans must appear as they are in your generation in order for it to be considered correct. Please be concise. You can find the text below.

# ```
# $TEXT
# ```
# """
# ) # 29/100
# template = Template(
# """Instruction: Please generate a paraphrase for the sentence that you will find below. Ensure the meaning and the message of the text stays the same in your output. A second extremely important condition is to ensure that the following two spans are preserved precisely as they are in your generation: "$ENTITY1" and "$ENTITY2". These two spans must appear as they are in your generation in order for it to be considered correct. Please be concise. You can find the text below. Very important to keep the two spans previously mentioned in your generation.

# ```
# $TEXT
# ```
# """
# ) # 29/100
template = Template(
"""Please generate a paraphrase for the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generation: "$ENTITY1", "$ENTITY2". 
Please be concise.
```
$TEXT
```
"""
) # 31/100


llm = ChatOpenAI(temperature=0)

def work_fn(example):
    # time.sleep(random.random())
    text = llm.call_as_llm(example)
    # time.sleep(random.random())
    text = "1. " + text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sentence-Pair Baseline")
    parser.add_argument('--dataset_path', type=str, required=True, help="The dataset to generate paraphrasings on.")
    parser.add_argument('--save_path', type=str, required=True, help="Where to save the resulting dataset")
    parser.add_argument('--number_of_paraphrases', type=int, required=False, default=3, help="How many paraphrases to generate.")
    parser.add_argument('--subset_start', type=int, default=0, required=False, help="Limit on how many to do (simple attempt to make sure we do not run an overwhelmingly expensive number of queries to OpenAI)")
    parser.add_argument('--subset_end', type=int, default=100000, required=False, help="Limit on how many to do (simple attempt to make sure we do not run an overwhelmingly expensive number of queries to OpenAI)")


    args = vars(parser.parse_args())

    random.seed(1)


    data = [] # defaultdict(list)
    with open(args['dataset_path']) as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded = json.loads(line)
            if len(loaded['token']) > 60:
                continue
            if len(loaded['token']) < 8:
                continue
            data.append(loaded)

    # data = [random.choice(y) for (x, y) in data.items()]
    # data = sorted([x['query'] for x in data])
    # random.shuffle(data)

    # data = data[:1_000_000]
    # exit()

    work_data = []
    for line in data:
        entity1 = ' '.join(line['token'][line['subj_start']:(line['subj_end']+1)])
        entity2 = ' '.join(line['token'][line['obj_start']:(line['obj_end']+1)])
        text    = ' '.join(line['token'])
        template_data = {
            # 'HOW_MANY': args['number_of_paraphrases'],
            'ENTITY1' : entity1,
            'ENTITY2' : entity2,
            'TEXT'    : text,
        }
        work_data.append(template.substitute(**template_data))

    print(len(work_data))
    # work_data = work_data[args['subset_start']:args['subset_end']]
    # print(len(work_data))
    # print(Counter([len(x['token']) for x in data]))
    # print(work_data[0])
    # exit()
    all_result = []
    for (start, end) in [
        # (0,     10000 ),
        (100000, 110000 ),
        (110000, 120000 ),
        (120000, 130000 ),
        (130000, 140000 ),
        (140000, 150000 ),
        (150000, 160000 ),
        (160000, 170000 ),
        (170000, 180000 ),
        (180000, 190000 ),
        (190000, 200000 ),

        (200000, 210000 ),
        (210000, 220000 ),
        (220000, 230000 ),
        (230000, 240000 ),
        (240000, 250000 ),
        (250000, 260000 ),
        (260000, 270000 ),
        (270000, 280000 ),
        (280000, 290000 ),
        (290000, 300000 ),
        
    ]:
        current_work_data = work_data[start:end]
        with multiprocessing.Pool(40) as p:
            result = list(tqdm.tqdm(p.imap(work_fn, current_work_data), total=len(current_work_data)))
        # result = [work_fn(x) for x in tqdm.tqdm(current_work_data)]
        
        print(result[0])
        print(result[1])

        with open(f'src/isolated_experiments/4_paraphrase_random_data/data/231125/merged_dataset/merged_1M_{start}_{end}.pickle', 'wb+') as fout:
            pickle.dump([data[start:end], current_work_data, result], fout)