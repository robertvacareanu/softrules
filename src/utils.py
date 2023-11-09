from collections import Counter, defaultdict
import math

from collections import defaultdict
from sklearn.metrics import confusion_matrix, f1_score

import numpy as np
import random
import multiprocessing

import sys
import scipy as sp
from typing import Dict, List, Any

import hashlib
import json

NO_RELATION = "no_relation"

def tacred_score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
         
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print( "Precision (micro): {:.2%}".format(prec_micro) ) 
        print( "   Recall (micro): {:.2%}".format(recall_micro) )
        print( "       F1 (micro): {:.2%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro


def line_to_hash(line: Dict[str, Any], use_all_fields: bool = False):
    if use_all_fields:
        name_variables = [
            str(' '.join(line['token'])),
            str(line['subj_start']),
            str(line['subj_end']),
            str(line['obj_start']),
            str(line['obj_end']),
            str(line['subj_type']),
            str(line['obj_type']),
            str(line['relation']),
        ]
    else:
        name_variables = [
            str(' '.join(line['token'])),
        ]

    return hashlib.md5('-'.join(name_variables).encode('utf-8')).hexdigest().lower()

def read_rules(path: str):
    result = defaultdict(list)
    with open(path) as fin:
        for line in fin:
            result.append(json.loads(line))

    return result

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
            'f1_macro'             : f1_score(gold, pred, average='macro') * 100,
            'f1_micro'             : f1_score(gold, pred, average='micro') * 100,
            'f1_micro_withoutnorel': f1_score(gold, pred, average='macro', labels=sorted(list(set(gold).difference(["no_relation"])))) * 100,
        })
    return results