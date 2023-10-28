"""

A sentence pair baseline, where prediction happens
(<support sentence1>, <test sentence>) -> \in [0, 1]
(<support sentence2>, <test sentence>) -> \in [0, 1]
..

"""

import json
import tqdm
import argparse
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from sentence_transformers import CrossEncoder
from scipy.special import expit
from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.utils import tacred_score
from sklearn.metrics import f1_score

def predict_with_model(model, all_episodes: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], **kwargs):
    gold           = []
    pred_scores    = []
    pred_relations = []
    for episode, gold_positions, ss_t_relations in tqdm.tqdm(zip(all_episodes[0], all_episodes[1], all_episodes[2]), total=len(all_episodes[0])):
        meta_train = episode['meta_train']
        meta_test  = episode['meta_test']
        
        to_be_predicted = []
        for test_sentence, gold_relation_position in zip(meta_test, gold_positions):
            relations            = []
            for support_sentences_for_relation, ss_relation in zip(meta_train, ss_t_relations[0]):
                for support_sentence in support_sentences_for_relation:
                    to_be_predicted.append([preprocess_line(support_sentence, preprocessing_type=kwargs['marker_type']), preprocess_line(test_sentence, preprocessing_type=kwargs['marker_type'])])
                relations.append(ss_relation)
            
            if gold_relation_position >= len(relations):
                gold.append('no_relation')
            else:
                gold.append(relations[gold_relation_position])

            pred_relations.append(relations)

        pred_scores += expit(model.predict(to_be_predicted).reshape(len(meta_test), len(meta_train), -1).mean(axis=2)).tolist()

    return gold, pred_scores, pred_relations


def train_model(model, **kwargs):
    """
    This function is reponsible for training the `model` over a file of the form `_train_data.json`
    This file looks like this:
    ```
    {
        'relation1': [<example1>, <example2>, ..],
        'relation2': [<example1>, <example2>, ..],
        'relation3': [<example1>, <example2>, ..],
        ..
    }
    ```
    Each `example<x>` looks the same like in a generic episode.
    """
    training_path = kwargs['training_path']
    with open(training_path) as fin:
        training_data = json.load(fin)
        training_data = [{**s, 'relation': relation} for (relation, sentences) in training_data.items() for s in sentences]

    # Sample 
    data = []
    for i in range(kwargs['finetuning_examples']):
        sent_a = random.choice(training_data)
        # It does not really make sense to add to the training data pairs 
        # with relations (<sentence with no_relation>, <sentence with no_relation>). We cannot train as if
        # they represent the same relation, since `no_relation` has at least
        # two interpretations: (1) purely no relation (debatable if this even
        # exists), (2) none of the relations considered
        # Therefore, it is possible to sample two sentences labeled as
        # `no_relation` that have a real (but not considered in this dataset) relation
        # different from each other
        # For example, 
        # S1: "John works in New York City." ((John, New York City) -> place of work)
        # S2: "John studied in New York City at NYU." ((John, New York City) -> place of college study)
        # If the dataset only labels 4 relations: (age, title, city of birth, city of death),
        # both S1 and S2 will be labeled with `no_relation`, which is correct. But it would be incorrect
        # to make a classifier to classify (S1, S2) as having the same relation.
        while sent_a['relation'] == 'no_relation':
            sent_a = random.choice(training_data)
        sent_b = random.choice(training_data)

        sent_a_tokens = preprocess_line(sent_a, preprocessing_type=kwargs['marker_type'])
        sent_b_tokens = preprocess_line(sent_b, preprocessing_type=kwargs['marker_type'])

        data.append(InputExample(texts=[sent_a_tokens, sent_b_tokens], label=1 if sent_a['relation'] == sent_b['relation'] else 0))

    dataloader = DataLoader(data, shuffle=True, batch_size=64)

    model.fit(train_dataloader=dataloader,
            epochs=1,
            warmup_steps=len(data) * 0.1)
    
    return model


def compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds, verbose):
    """
    Compute the results for each threshold and returns the results
    """
    results = []
    for threshold in thresholds:
        pred = []
        for ps, pr in zip(pred_scores, pred_relations):
            if np.max(ps) > threshold:
                pred.append(pr[np.argmax(ps)])
            else:
                pred.append('no_relation')
        scores = [s * 100 for s in tacred_score(gold, pred, verbose=verbose)] # Make the scores be 0-100

        results.append({
            'threshold'            : threshold,
            'p_tacred'             : scores[0],
            'r_tacred'             : scores[1],
            'f1_tacred'            : scores[2],
            'f1_macro'             : f1_score(gold, pred, average='macro'),
            'f1_micro'             : f1_score(gold, pred, average='micro'),
            'f1_micro_withoutnorel': f1_score(gold, pred, average='macro', labels=sorted(list(set(gold).difference(["no_relation"])))),
        })
    return results

def main(args):
    """
    The main entry point of this file
    This function is reponsible for:
        - creating (and training) the model
        - computing final results over potentially multiple files with a given threshold; if 
        there is no given threshold, find it on a second file
        - append the results to a file
    """
    model = CrossEncoder(args['model_name'], max_length=512)
    
    if args['do_train']:
        model = train_model(model, training_path=args['training_path'], finetuning_examples=args['finetuning_examples'], marker_type=args['marker_type'])
    
    if args['threshold']: # If threshold is passed, use it
        threshold = [args['threshold']]
    else: # If not, then we have to find it on `find_threshold_on_path`
        if args['find_threshold_on_path']:
            with open(args['find_threshold_on_path']) as fin:
                all_episodes = json.load(fin)
            
            # Compute predictions
            gold, pred_scores, pred_relations = predict_with_model(model, all_episodes, marker_type=args['marker_type'])
            # Select best threshold 
            best_threshold_results = compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds=np.linspace(0, 1, 101).tolist(), verbose=False)
            threshold = max(best_threshold_results, key=lambda x: x['f1_tacred'])['threshold']
            print("Best threshold: ", max(best_threshold_results, key=lambda x: x['f1_tacred'])['threshold'], "with score: ", max(best_threshold_results, key=lambda x: x['f1_tacred'])['f1_tacred'])
            print(best_threshold_results)
        else:
            raise ValueError("No threshold nor file to find it is passed. Is everything ok?")
    
    results = []
    for evaluation_path in args['evaluation_paths']:
        with open(evaluation_path) as fin:
            test_episodes = json.load(fin)

        gold, pred_scores, pred_relations = predict_with_model(model, test_episodes, marker_type=args['marker_type'])
        print("###############")
        print('Evaluation Path: ', evaluation_path)
        # [0] -> Results for only one thresholds; 
        # [1] -> Get the result portion (it is a Tuple with (1) -> Threshold and (2) -> Reults)
        scores = compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds=[threshold], verbose=True)[0]
        results.append({'evaluation_path': evaluation_path, **scores})
        print(scores)
        print("###############")

    print("Final results")
    df = pd.DataFrame(results)
    print("P:  ", str(df['p_tacred'].mean())                                 + " +- " + str(df['p_tacred'].std()))
    print("R:  ", str(df['r_tacred'].mean())                                 + " +- " + str(df['r_tacred'].std()))
    print("F1: ", str(df['f1_tacred'].mean())                                + " +- " + str(df['f1_tacred'].std()))
    print("F1: (macro) ", str(df['f1_macro'].mean())                         + " +- " + str(df['f1_macro'].std()))
    print("F1: (micro) ", str(df['f1_micro'].mean())                         + " +- " + str(df['f1_micro'].std()))
    print("F1: (micro) (wo norel) ", str(df['f1_micro_withoutnorel'].mean()) + " +- " + str(df['f1_micro_withoutnorel'].std()))

    if args['append_results_to_file']:
        with open(args['append_results_to_file'], 'a+') as fout:
            for line in results:
                _=fout.write(json.dumps({**line, 'args': args}))
                _=fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sentence-Pair Baseline")
    parser.add_argument('--marker_type',            type=str, default='entity_mask', help="How to mark the entities", choices=['entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct'])
    parser.add_argument('--model_name',             type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2', help="What model to use")
    parser.add_argument('--finetuning_examples',    type=int, default=50_000, help="How many examples to fine-tune on")
    parser.add_argument('--do_train', action='store_true', required=False, help="Whether to fine-tune the model or not")
    parser.add_argument('--training_path',          type=str, help="What to train on.")
    parser.add_argument('--evaluation_paths',       type=str, nargs='*', help="What to evaluate on")
    parser.add_argument('--threshold',              type=float, default=None, help="The classification threshold")
    parser.add_argument('--seed',                   type=int, default=1, help="The random seed to use")
    parser.add_argument('--find_threshold_on_path', type=str, help="Finds the best threshold by evaluating on this file")
    parser.add_argument('--append_results_to_file', type=str, help="Appends results to this file")
    args = vars(parser.parse_args())
    print(args)

    seed_everything(args['seed'])

    main(args)

