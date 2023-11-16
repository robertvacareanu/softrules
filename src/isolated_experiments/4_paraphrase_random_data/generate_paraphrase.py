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
from langchain.chat_models import ChatOpenAI
from src.baseline.entity_marker_cross_encoder import entity_marker

template1 = Template(
"""Please rephrase the following sentence. Please ensure the meaning and the message stays the same and these two entities are preserved in your generation: "$ENTITY1", "$ENTITY2". Please be concise.
```
$TEXT
```
"""
)
"""
This template generates different text, but keeps the entities
"""

llm = ChatOpenAI(temperature=0)

def work_fn(example):
    time.sleep(random.random())
    text = llm.call_as_llm(example)
    time.sleep(random.random())
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Paraphrases1")
    parser.add_argument('--dataset_path', type=str, required=True, help="The dataset to generate paraphrasings on.")
    parser.add_argument('--save_path', type=str, required=True, help="Where to save the resulting dataset")
    parser.add_argument('--subset_start', type=int, default=0, required=False, help="Limit on how many to do (simple attempt to make sure we do not run an overwhelmingly expensive number of queries to OpenAI)")
    parser.add_argument('--subset_end', type=int, default=100000, required=False, help="Limit on how many to do (simple attempt to make sure we do not run an overwhelmingly expensive number of queries to OpenAI)")

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
            'ENTITY1' : entity1,
            'ENTITY2' : entity2,
            'TEXT'    : text,
        }
        work_data.append(template1.substitute(**template_data))
    

    work_data = work_data[args['subset_start']:args['subset_end']]
    print(len(work_data))
    print(work_data[0])

    with multiprocessing.Pool(500) as p:
        result = list(tqdm.tqdm(p.imap(work_fn, work_data), total=len(work_data)))
    # result = [work_fn(x) for x in tqdm.tqdm(work_data)]
    

    print(result[0])
    print(result[1])

    with open(args['save_path'], 'wb+') as fout:
        import pickle
        pickle.dump([data, work_data, result], fout)