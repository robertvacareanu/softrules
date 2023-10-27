from collections import defaultdict
import json
import random
import tqdm
import argparse
import os
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


from src.utils import tacred_score

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--seed', type=int, required=False, default=1, help="The random seed (default=1)")

    return parser

"""
Some sanity checks to ensure everything works smooth
Function with side-effects. If a check fails we raise a ValueError
"""
def sanity_checks_args(args):
    if not os.path.exists(args['dataset_path']):
        dataset_path = args['dataset_path']
        raise ValueError(f"The file at {dataset_path} does not exist. Is everything ok?")




# python -m src.baseline.unsupervised_entity_based_classification --dataset_path "/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160290.json"
if __name__ == "__main__":
    random.seed(1)

    args = vars(get_parser().parse_args())
    sanity_checks_args(args)

    with open(args['dataset_path']) as fin:
        data = json.load(fin)
    gold = []
    pred = []
    for episode, relations in tqdm.tqdm(zip(data[0], data[2]), total=len(data[0])):
        meta_train = episode['meta_train']
        meta_test  = episode['meta_test']
        for test_sentence, gr in zip(meta_test, relations[1]):
            rules             = []
            rules_relations   = []
            rules_types       = []

            if gr not in relations[0]: 
                gold_relation = 'no_relation'
            else:
                gold_relation = gr

            gold.append(gold_relation)

            entity_types = defaultdict(list)
            for support_sentences_per_relation, relation in zip(meta_train, relations[0]):
                for ss in support_sentences_per_relation:
                    entity_types[(ss['subj_type'], ss['obj_type'])].append(relation)
            if (test_sentence['subj_type'], test_sentence['obj_type']) in entity_types:
                # if test_sentence['subj_type'] == test_sentence['obj_type']:
                    # pred.append('no_relation')
                # else:
                pred.append(random.choice(entity_types[(test_sentence['subj_type'], test_sentence['obj_type'])]))
            else:
                pred.append('no_relation')
            

    print(tacred_score(gold, pred, verbose=True))
    print(f1_score(gold, pred, average='micro') * 100)
    print(f1_score(gold, pred, labels=list(set(gold).difference(['no_relation'])), average='micro') * 100)
    print("#####################")
    for rel in sorted(list(set(gold).difference(['no_relation']))):
        print(rel, *[x * 100 for x in precision_recall_fscore_support(gold, pred, labels=[rel], average='micro')[:3]])

    print("\n")
    print("---------------------")
    for rel in sorted(list(set(gold))):
        print(rel, *[x * 100 for x in precision_recall_fscore_support(gold, pred, labels=[rel], average='micro')[:3]])

    print('overall', *[x * 100 for x in precision_recall_fscore_support(gold, pred, labels=sorted(list(set(gold).difference(['no_relation']))), average='micro')[:3]])

    print("\n")
    print("---------------------")
    print(confusion_matrix(gold, pred, labels=sorted(list(set(gold)))))

    # print(tacred_score([x for x in gold if x!='no_relation'], [x for (x, y) in zip(pred, gold) if y!='no_relation'], verbose=True))
    # print(f1_score(gold, pred, average='macro'))

