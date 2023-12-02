"""
Create a train-test split
"""

import random
import json

if __name__ == "__main__":
    with open('/home/rvacareanu/projects_5_2210/rule_generation/random/enhanced_syntax_all.jsonl') as fin:
        data = []
        for line in fin:
                data.append(json.loads(line))


    random.seed(1)
    random.shuffle(data)
    train = data[100:]
    test  = data[:100]

    with open('/home/rvacareanu/projects_5_2210/rule_generation/random/enhanced_syntax_all_train.jsonl', 'w+') as fout:
        for line in train:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

    with open('/home/rvacareanu/projects_5_2210/rule_generation/random/enhanced_syntax_all_test.jsonl', 'w+') as fout:
        for line in test:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')

