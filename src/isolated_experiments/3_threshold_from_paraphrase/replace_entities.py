"""
Use one of the Open AI models to generate paraphrasings

python -m src.isolated_experiments.1_paraphrase.openai --dataset_path ../Few-Shot_Datasets/TACRED/few_shot_data/_train_data.json --save_path results/231027/paraphrashing/openai/TACRED/output.pickle --number_of_paraphrases 5

python -m src.isolated_experiments.1_paraphrase.openai --dataset_path ../Few-Shot_Datasets/WIKIDATA/few_shot_data/_dev_data.json --save_path results/231027/paraphrashing/openai/NYT29/output.pickle --number_of_paraphrases 5
"""

import argparse

from string import Template

import os
import time
import json
import multiprocessing
import tqdm
import random
from langchain.chat_models import ChatOpenAI
from src.baseline.entity_marker_cross_encoder import entity_marker


template = Template(
"""Please generate $HOW_MANY paraphrases for the following sentence. Please ensure the meaning and the message stays the same. Please replace these two entities with other entities that would be suitable: "$ENTITY1", "$ENTITY2". Please be concise.
```
$TEXT
```
1. """
)

llm = ChatOpenAI(temperature=0)

def work_fn(example):
    time.sleep(random.random())
    text = llm.call_as_llm(example)
    time.sleep(random.random())
    text = "1. " + text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sentence-Pair Baseline")
    parser.add_argument('--dataset_path', type=str, required=True, help="The dataset to generate paraphrasings on.")
    parser.add_argument('--save_path', type=str, required=True, help="Where to save the resulting dataset")
    parser.add_argument('--number_of_paraphrases', type=int, required=False, default=5, help="How many paraphrases to generate.")

    args = vars(parser.parse_args())

    random.seed(1)

    data = []
    with open(args['dataset_path']) as fin:
        for line in fin:
            data.append(json.loads(line))

    
    # entity_marker({'token': line.split(), ''}, subj_start_marker, subj_end_marker, obj_start_marker, obj_end_marker)


    work_data = []
    for line in data:
        entity1 = ' '.join(line['token'][line['subj_start']:(line['subj_end']+1)])
        entity2 = ' '.join(line['token'][line['obj_start']:(line['obj_end']+1)])
        text    = ' '.join(line['token'])
        template_data = {
            'HOW_MANY': args['number_of_paraphrases'],
            'ENTITY1' : entity1,
            'ENTITY2' : entity2,
            'TEXT'    : entity_marker(line, subj_start_marker = lambda x: '[E1]', subj_end_marker = lambda x: '[E1]', obj_start_marker = lambda x: '[E2]', obj_end_marker = lambda x: '[E2]'),
        }
        work_data.append(template.substitute(**template_data))
    
    with multiprocessing.Pool(20) as p:
        result = list(tqdm.tqdm(p.imap(work_fn, work_data), total=len(work_data)))
    # result = [work_fn(x) for x in tqdm.tqdm(work_data)]
    

    print(result[0])
    print(result[1])

    with open(args['save_path'], 'wb+') as fout:
        import pickle
        pickle.dump([data, work_data, result], fout)