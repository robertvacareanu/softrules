import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
from src.utils import line_to_hash
import argparse
import json
import tqdm
import torch
from src.baseline.entity_marker_cross_encoder import preprocess_line

from collections import defaultdict

from src.softrules.different_encoder.model_cliplike_multidata import SoftRulesEncoder, read_rules, get_valdata
from src.utils import tacred_score, compute_results_with_thresholds

from torch.utils.data import DataLoader
import datasets

import scipy as sp
import scipy.special as spp

def read_data(args):
    rules = read_rules(args['rules_path'])
    rules = dict(rules)

    with open(args['train_path']) as fin:
        data = json.load(fin)
        train_rules = [
            {
                'id': i,
                'rule': rules[line_to_hash(x, use_all_fields=True)],
                'relation': x['relation']
            }    
        for i, x in enumerate(data)]

    with open(args['test_path']) as fin:
        data = json.load(fin)
        test_sentences = [
            {
                'id': i,
                'sentence': preprocess_line(line, 'typed_entity_marker_punct'),
                'relation': line['relation']
            }
            for i, line in enumerate(data)
        ]

    return (train_rules, test_sentences)

def custom_rules():
    rules = {
        # 'org:alternate_names'                : '[entity=organization]+ <nsubj known >nmod_as [entity=organization]+',
        # 'org:city_of_headquarters'           : '[entity=organization]+ <nsubj located >nmod_in [entity=city]+',
        # 'org:country_of_headquarters'        : '[entity=organization]+ <nsubj located >nmod_in [entity=city]+',
        # 'org:dissolved'                      : '[entity=organization]+ <nsubj dissolved >nmod_on [entity=date]+',
        # 'org:founded'                        : '[entity=organization]+ <nsubj founded >nmod_on [entity=date]+',
        'org:founded_by'                     : '[entity=organization]+ <nsubj founded >nmod_by [entity=date]+',
        # 'org:member_of'                      : '[entity=organization]+ <nsubj part >nmod_of [entity=organization]+',
        # 'org:members'                        : '[entity=organization]+ <nsubj member >nmod_of [entity=organization, country]+',
        # 'org:number_of_employees/members'    : '[entity=organization]+ <nsubj employes >nmod_of [entity=number]+',
        # 'org:parents'                        : '[entity=organization]+ <nsubj parent >nmod_of [entity=organization]+',
        # 'org:political/religious_affiliation': '[entity=organization]+ <nsubj has >nmod [entity=ideology]+',
        # 'org:shareholders'                   : '[entity=person]+ <nsubj shareholder >nmod_of [entity=organization]+',
        # 'org:stateorprovince_of_headquarters': '',
        # 'org:subsidiaries'                   : '[entity=organization]+ <nsubj subsidiary >nmod_of [entity=organization]+',
        # 'org:top_members/employees'          : '',
        'org:website'                        : '[entity=organization]+ <appos [entity=url]+',
        'per:age'                            : '[entity=person]+ <nsubj is >dobj [entity=age]+',
        'per:alternate_names'                : '[entity=person]+ <nsubj known >nmod_as [entity=name]+',
        'per:cause_of_death'                 : '[entity=person]+ <nsubj died >nmod_of [entity=cause of death]+',
        'per:charges'                        : '[entity=person]+ <nsubj charged >nmod_of [entity=legal charge]+',
        'per:children'                       : '[entity=person]+ <nsubj child >nmod_of [entity=parent]+',
        'per:cities_of_residence'            : '[entity=person]+ <nsubj lives >nmod_in [entity=city]+',
        'per:city_of_birth'                  : '[entity=person]+ <nsubj born >nmod_in [entity=city]+',
        'per:city_of_death'                  : '[entity=person]+ <nsubj died >nmod_in [entity=city]+',
        'per:countries_of_residence'         : '[entity=person]+ <nsubj lives >nmod_in [entity=country]+',
        'per:country_of_birth'               : '[entity=person]+ <nsubj born >nmod_in [entity=country]+',
        'per:country_of_death'               : '[entity=person]+ <nsubj died >nmod_in [entity=country]+',
        'per:date_of_birth'                  : '[entity=person]+ <nsubj born >nmod_on [entity=date]+',
        'per:date_of_death'                  : '[entity=person]+ <nsubj died >nmod_on [entity=date]+',
        'per:employee_of'                    : '[entity=person]+ <nsubj works >nmod_for [entity=employer]+',
        # 'per:origin'                         : '',
        # 'per:other_family'                   : '',
        'per:parents'                        : '[entity=parent]+ >nmod_of [entity=child]+',
        'per:religion'                       : '[entity=person]+ <nsubj has >nmod [entity=religion]+',
        'per:schools_attended'               : '[entity=person]+ >nmod_attended [entity=school]+',
        'per:siblings'                       : '[entity=person]+ >sibling_of [entity=person]+',
        'per:spouse'                         : '[entity=person]+ <nsubj married >nmod_to [entity=person]+',
        'per:stateorprovince_of_birth'       : '[entity=person]+ <nsubj born >nmod_in [entity=state or province]+',
        'per:stateorprovince_of_death'       : '[entity=person]+ <nsubj died >nmod_in [entity=state or province]+',
        'per:stateorprovinces_of_residence'  : '[entity=person]+ <nsubj lives >nmod_in [entity=state or province]+',
        'per:title'                          : '[entity=title]+ <appos [entity=person]+',
    }

    return rules

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,) # /storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_113/checkpoints/epoch=0-step=85000.ckpt
    parser.add_argument("--how_many_rules_to_average", type=int, default=1)
    parser.add_argument("--rules_path", type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full/surface.jsonl", 
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full/enhanced_syntax.jsonl", 
    ])
    parser.add_argument("--train_path", type=str, default="/storage/rvacareanu/data/softrules/fsre_dataset/TACRED_full/train.json")
    parser.add_argument("--test_path",  type=str, default="/storage/rvacareanu/data/softrules/fsre_dataset/TACRED_full/dev.json")

    parser.add_argument('--use_rules_for_no_relation', action='store_true')
    parser.add_argument('--unique_rules', action='store_true')


    args = vars(parser.parse_args())
    print(args)
    pl.seed_everything(1)
    model = SoftRulesEncoder.load_from_checkpoint(args['checkpoint'])
    model.thresholds = np.linspace(0, 1, 101).tolist()
    model.hyperparameters['append_results_to_file'] = None
    model.hyperparameters['dev_path'] = args['test_path']
    model.hyperparameters['how_many_rules_to_average'] = args['how_many_rules_to_average']

    with open(args['train_path']) as fin:
        train = json.load(fin)

    rules = read_rules(args['rules_path'])
    rules = dict(rules)

    # Get the rules over the sentences in training; Decide whether to include no relation or not
    if args['use_rules_for_no_relation']:
        train_rules = [y for line in train for y in rules[line_to_hash(line, use_all_fields=True)]]
    else:
        train_rules = [y for line in train for y in rules[line_to_hash(line, use_all_fields=True)] if line['relation'] != 'no_relation']

    train_rules = [(x['relation'], x['query'].lower()) for x in train_rules]
    train_rules = sorted(train_rules)
    if args['unique_rules']:
        train_rules = sorted(list(set(train_rules)))

    # train_rules = list(custom_rules().items())
    # print(len(train_rules))
    
    # Store the relation associated with each rule
    rules_relations = [x[0] for x in train_rules]

    # Tokenize, then construct the dataset and the dataloader
    rules_tokenized = model.tokenize_rules([x[1].lower() for x in train_rules])
    rules_dataset = datasets.Dataset.from_dict(rules_tokenized)
    rules_dl = DataLoader(rules_dataset, batch_size=128, collate_fn=lambda rule_inputs: model.data_collator_rule(rule_inputs))
    
    # Finally, construct the encodings of the sentences
    rules_encodings = []
    for batch in tqdm.tqdm(rules_dl):
        rules_encodings.append(model.encode_rule({k:v.to(model.device) for (k, v) in batch.items()}).detach().cpu().numpy())

    with open(args['test_path']) as fin:
        test_sentences = json.load(fin)
    
    # Store the relation of each sentence
    sentences_gold = [x['relation'] for x in test_sentences]

    # Tokenize, then construct the dataset and the dataloader
    sentences_tokenized = model.tokenize_sentences([preprocess_line(x, 'typed_entity_marker_punct') for x in test_sentences])
    sentences_dataset = datasets.Dataset.from_dict(sentences_tokenized)
    sentences_dl = DataLoader(sentences_dataset, batch_size=128, collate_fn=lambda rule_inputs: model.data_collator_rule(rule_inputs))

    # Finally, construct the encodings of the sentences
    sentences_encodings = []
    for batch in tqdm.tqdm(sentences_dl):
        sentences_encodings.append(model.encode_sent({k:v.to(model.device) for (k, v) in batch.items()}).detach().cpu().numpy())

    # Stack every rule and every sentence
    x1 = np.vstack(rules_encodings)
    x2 = np.vstack(sentences_encodings)

    # Store the similarities between each rule and each sentence
    result = x1 @ x2.T
    preds_max  = result.max(axis=0)
    preds_ids  = result.argmax(axis=0)
    # preds_rels = np.array([rules_relations[x] for x in result.argmax(axis=0)])

    # Use the standard notation of `gold` to hold the gold predictions
    gold = sentences_gold

    # Iterate over some thresholds; Take the maximum similarity; If it is >= threshold, => predict the the associated relation
    # Otherwise, predict no relation
    # Lastly, compute the tacred score
    for threshold in np.linspace(0.0, 1, 11).tolist():
        preds_rels = np.array([rules_relations[x] for x in result.argmax(axis=0)])
        preds_rels[preds_max < threshold] = 'no_relation'
        pred = preds_rels.tolist()
        print(threshold, tacred_score(gold, pred, verbose=False))


    # In the evaluation above we do not use `logit_scale`, a parameter used during training to scale the predictions
    # And we do not use softmax
    # Below, it is an evaluation that attempts to fix these two potential issues (they might not be issues, actually)
    relations_to_rule_ids = defaultdict(list)
    for i, r in enumerate(rules_relations):
        relations_to_rule_ids[r].append(i)

    # A map from relation to the indices for the rules associated with this relation
    relations_to_rule_ids = {k:np.array(v) for (k, v) in sorted(relations_to_rule_ids.items(), key=lambda x: x[0])}

    # The value with which to scale the similarities
    logit_scale = torch.exp(model.logit_scale).detach().cpu().numpy()

    relations_to_sent_ids = defaultdict(list)
    for i, r in enumerate(gold):
        relations_to_sent_ids[r].append(i)

    # A map from relation to the indices for the sentences associated with this relation (gold)
    relations_to_sent_ids = {k:np.array(v) for (k, v) in sorted(relations_to_sent_ids.items(), key=lambda x: x[0])}

    # Construct a list of lists; each column (i.e. sentence) has a list with two elements: relation, and scores for that relation
    # so `all_pred_for_col[0]` gives a list associated with sentence 0. And `all_pred_for_col[0][0]` looks like so: (<relation>, <list with similarities between sentence 0 and rules with <relation> relation>)
    all_pred_for_col = []
    for col in tqdm.tqdm(range(result.shape[1])):
        pred_for_col = []
        for relation, ids in relations_to_rule_ids.items():
            pred_for_col.append([relation, sorted(result[:,col][ids], reverse=True)])
        all_pred_for_col.append(pred_for_col)

    print("\n\n")
    print("-"*20)
    print("\n\n")
    print("Average top X")



    # Iterate over how many rules for each relation to consider
    for how_many_to_average in [1, 3, 5, 7, 11, 23, 100, 100_000]:
        # how_many_to_average=1
        all_sents_text = []
        all_preds = []
        all_rels  = []
        for col in tqdm.tqdm(range(result.shape[1])):
            pred_for_col = [[x[0], np.mean(x[1][:how_many_to_average])] for x in all_pred_for_col[col]]
            all_sents_text.append(preprocess_line(test_sentences[col], 'typed_entity_marker_punct'))
            all_preds.append((spp.softmax([logit_scale*x[1] for x in pred_for_col])).tolist())
            all_rels.append([x[0] for x in pred_for_col])
        final_result = compute_results_with_thresholds(gold, all_preds, all_rels, np.linspace(0, 1, 101).tolist(), verbose=False)
        best = max(final_result, key=lambda x: x['f1_tacred'])
        print({'how_many_to_average': how_many_to_average, **best})
        print(compute_results_with_thresholds(gold, all_preds, all_rels, [best['threshold']], verbose=False))

    # for (a,b) in [(i, ast) for i, (ast, ap, ar, g) in enumerate(zip(all_sents_text, all_preds, all_rels, gold)) if g == 'per:parents'][:10]:
    #     print('-'*20)
    #     print(b)
    #     print(sorted(list(zip(all_preds[a], all_rels[a])), key=lambda x: x[0], reverse=True)[:3])
    #     print('-'*20)
    #     print("\n\n")

    # import pandas as pd
    # temp_result = []
    # for (idx, (rule, rule_relation)) in enumerate(train_rules):
    #     for r in relations_to_sent_ids.keys():
    #         for similarity in result[idx, relations_to_sent_ids[r]].tolist():
    #             temp_result.append({
    #                 'relation': r,
    #                 'similarity': similarity,
    #             })
    #     df = pd.DataFrame(temp_result)
    #     df['rule'] = rule
    #     df['rule_relation'] = rule_relation
    #     print("-"*10)
    #     print(rule_relation, rule)
    #     print(df.groupby(by=['relation']).agg({'similarity': ['mean', 'std', 'count']}).reset_index().sort_values(('similarity', 'mean')).reset_index())
    #     print("-"*10)
    # # df.to_csv('/storage/rvacareanu/code/projects_7_2309/softrules/results/231122/interesting_rules/rule3.csv', sep=',', index=False)

    print("\n\n")
    print("-"*20)
    print("\n\n")
    print("Average top X% instead of top X")

    # Iterate over how many rules for each relation to consider
    for how_many_to_average in [0.01, 0.1, 0.2, 0.5, 0.8]:
        all_preds = []
        all_rels  = []
        for col in tqdm.tqdm(range(result.shape[1])):
            pred_for_col = [[x[0], np.mean(x[1][:max(int(how_many_to_average * len(x[1])), 1)])] for x in all_pred_for_col[col]]
            all_preds.append((logit_scale*spp.softmax([x[1] for x in pred_for_col])).tolist())
            all_rels.append([x[0] for x in pred_for_col])
        final_result = compute_results_with_thresholds(gold, all_preds, all_rels, np.linspace(0, 1, 101).tolist(), verbose=False)
        best = max(final_result, key=lambda x: x['f1_tacred'])
        print({'how_many_to_average': how_many_to_average, **best})

        print(compute_results_with_thresholds(gold, all_preds, all_rels, [best['threshold']], verbose=False))

    # rules = read_rules(args['rules'])
    # rules = dict(rules)
    # val_data = get_valdata({**model.hyperparameters, **args}, model=model)

    # trainer = Trainer(
    #     accelerator="gpu", 
    #     precision='16-mixed',
    # )
    
    # for vd in val_data:
    #     print("-"*25)
    #     print(vd)
    #     trainer.validate(model=model,dataloaders=DataLoader(dataset=vd, collate_fn=model.collate_tokenized_fn, batch_size=64, num_workers=32))
    #     print("-"*25)
