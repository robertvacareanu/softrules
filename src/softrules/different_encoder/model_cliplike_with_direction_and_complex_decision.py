"""
This file contains the following logic:
- Model
- Tokenization according to what the model expects
- Collator according to how the model expects

(1) Training: CLIP style
have a list of (rule, sentence) tuples
[
    (rule1, sentence1),
    (rule2, sentence2),
    (rule3, sentence3),
    (rule4, sentence4),
    (rule5, sentence5),
    (rule6, sentence6),
    (rule7, sentence7),
    (rule8, sentence8),
]
Train such that encoder1(rule_i) ~= encoder2(sentence_i) and encoder1(rule_i) !~= encoder2(rule_j) (i != j)

Intuitively, in CLIP this worked because of the very large space of possible texts/images. Here we have a much
smaller space. But this file follows CLIP implementation. Changes might come after a robust evaluation


TODO
- Projection head? (https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1)
"""

import json
import argparse
import tqdm
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from src.utils import line_to_hash
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import CombinedLoader

from datetime import datetime

from torch.utils.data import DataLoader

from typing import Dict, List, Literal, Any
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.softrules.entity_marker_with_reg import typed_entity_marker_punct, typed_entity_marker_punct_v2, replace_rule_entity_types
from src.utils import compute_results_with_thresholds
from src.softrules.projection_head import ProjectionHead

import numpy as np
import scipy as sp
import scipy.special as spp

import datasets

from collections import defaultdict, Counter

import itertools
import hashlib

import torch.optim as optim

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class SoftRulesEncoder(pl.LightningModule):
    """
    Encode the concatenation of rules and the sentences with the same encoder
    Training is done in a CLIP style
    Classification is done as follows:
    In an N-way K-shot episode, you have N*K support sentences.
    (1) Encode all of the corresponding rules, resulting in an [N*K, D] vector. Reshape it
    to [N, K, D], then perform a mean pooling over dimension=1, resulting in
    a vector of shape [N, 1, D] => [N, D]
    (2) Encode the sentence corresponding to the query => [1, D]
    (3) Compute the similarity, like in training step (i.e. `logit_scale * input1_embs @ input2_embs.t()`)
    (4) Take the one with the highest similarity; If it is `>= threshold`, predict its corresponding relation
    Otherwise predict `no_relation`.

    To save up time during validation, the process works as follows:
    - Encode all rules (unique)
    - Encode all sentences (unique)
    - Then, in a method like `validation_end`, collect everything in the right order.
    
    """
    def __init__(
        self, 
        # model_name_or_checkpoint_rule: str, 
        # model_name_or_checkpoint_sentence: str,
        # marker_type: str, 
        # correct_expected_score : float = 1.0,
        # same_rel_expected_score: float = 0.5,
        # diff_rel_expected_score: float = 0.0,
        # thresholds = np.linspace(0,1,101).tolist()
        hyperparameters = {}
    ):
        super().__init__()
        self.hyperparameters    = hyperparameters

        if self.hyperparameters['train_rule_model_from_scratch']:
            self.model_rule         = AutoModel.from_config(AutoConfig.from_pretrained(self.hyperparameters['model_name_or_checkpoint_rule']))
        else:
            self.model_rule         = AutoModel.from_pretrained(self.hyperparameters['model_name_or_checkpoint_rule'])
        self.tokenizer_rule     = AutoTokenizer.from_pretrained(self.hyperparameters['model_name_or_checkpoint_rule'])
        self.data_collator_rule = DataCollatorWithPadding(tokenizer=self.tokenizer_rule, return_tensors="pt")

        if self.hyperparameters['train_sentence_model_from_scratch']:
            self.model_sentence         = AutoModel.from_config(AutoConfig.from_pretrained(self.hyperparameters['model_name_or_checkpoint_sentence']))
        else:
            self.model_sentence         = AutoModel.from_pretrained(self.hyperparameters['model_name_or_checkpoint_sentence'])
        self.tokenizer_sentence     = AutoTokenizer.from_pretrained(self.hyperparameters['model_name_or_checkpoint_sentence'])
        self.data_collator_sentence = DataCollatorWithPadding(tokenizer=self.tokenizer_sentence, return_tensors="pt")

        self.temperature = 1.0

        # self.marker_type = marker_type

        # self.correct_expected_score  = correct_expected_score
        # self.same_rel_expected_score = same_rel_expected_score
        # self.diff_rel_expected_score = diff_rel_expected_score

        self.thresholds = self.hyperparameters['thresholds']
        self.rule_target_token_idx = self.hyperparameters.get('rule_target_token_idx', 0)
        self.sent_target_token_idx = self.hyperparameters.get('sent_target_token_idx', 0)

        self.projection_dims = self.hyperparameters.get('projection_dims', 256)
        
        self.dropout = self.hyperparameters.get('dropout', 0.0)


        self.rule_projection = ProjectionHead(embedding_dim=self.model_rule.config.hidden_size, projection_dim=self.projection_dims, dropout=self.dropout)
        self.sent_projection = ProjectionHead(embedding_dim=self.model_sentence.config.hidden_size, projection_dim=self.projection_dims, dropout=self.dropout)



        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))

        self.save_hyperparameters(self.hyperparameters)

        self.val_step_outputs = defaultdict(list)

        # Cache the rule/sentence during evaluation to speed-up the process
        # The few-shot datasets have the same rule/sentence appearing multiple times
        self.val_rule_encodings = defaultdict(dict)
        self.val_sentence_encodings = defaultdict(dict)


        if self.hyperparameters.get('gradient_checkpointing_enable', False):
            self.model_rule.gradient_checkpointing_enable()
            self.model_sentence.gradient_checkpointing_enable()


    def predict(self, batch):

        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]
        
        input1_embs = self.encode_rule(input1)
        input2_embs = self.encode_sent(input2)
        
        return input1_embs @ input2_embs.t()

    def encode_rule(self, inputs):
        result = self.model_rule.forward(**inputs).last_hidden_state[:, self.rule_target_token_idx, :]
        result = self.rule_projection.forward(result)
        return F.normalize(result, dim=-1)

    def encode_sent(self, inputs):
        result = self.model_sentence.forward(**inputs).last_hidden_state[:, self.sent_target_token_idx, :]
        result = self.sent_projection.forward(result)
        return F.normalize(result, dim=-1)
    
    def training_step_one_batch(self, batch, batch_dataloader_id=0, return_logits=False, return_embs=False, return_individual_losses=False, *args, **kwargs):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size
        """
        rule_id     = torch.tensor(batch['rule_id'])
        sentence_id = torch.tensor(batch['sentence_id'])

        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]
        input1_embs = self.encode_rule(input1) # [B, D]
        input2_embs = self.encode_sent(input2) # [B, D]
        
        logit_scale     = self.logit_scale.exp()
        # logits_per_rule = logit_scale * input1_embs @ input2_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        logits_per_rule = logit_scale * input1_embs @ input2_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        logits_per_sent = logit_scale * input2_embs @ input1_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        # logits_per_sent = logits_per_rule.t() # [B, B]; [i, j] => similarity of sentence i with rule j

        # # Construct ground truth
        # true_rels = batch['relations']
        # ground_truth = []
        # for (i, r_i) in enumerate(true_rels):
        #     for (j, r_j) in enumerate(true_rels):
        #         if i == j:
        #             ground_truth.append(self.correct_expected_score)
        #         else:
        #             if r_i == r_j:
        #                 ground_truth.append(self.same_rel_expected_score)
        #             else:
        #                 ground_truth.append(self.diff_rel_expected_score)

        # ground_truth = torch.tensor(ground_truth).to(logits_per_rule.device)
        # loss_rule = torch.nn.functional.binary_cross_entropy_with_logits(logits_per_rule, ground_truth)
        # loss_sent = torch.nn.functional.binary_cross_entropy_with_logits(logits_per_sent, ground_truth)

        # ground_truth = torch.arange(input1_embs.size(0)).to(logits_per_rule.device)

        # The same underlying rule should match the same underlying sentences
        # The same underlying sentences should be matched by the same underlying rules
        # The or allows for cases when same sentence appears with different rules and when same rule appears with different sentences
        ground_truth = torch.logical_or(rule_id == rule_id[:, None], (sentence_id == sentence_id[:, None])).float().to(logits_per_rule.device)


        loss_rule = torch.nn.functional.cross_entropy(logits_per_rule, ground_truth)
        loss_sent = torch.nn.functional.cross_entropy(logits_per_sent, ground_truth)
        loss = (loss_rule + loss_sent) / 2

        # If there are duplicates the loss can vary a lot (e.g. `[[1,0,0], [0,1,0], [0,0,1]]` vs `[[1,0,1], [0,1,0], [1,0,1]]`; in the 
        # beginning, the model outputs mostly uniform probability; So `[0.33, 0.33, 0.33]`; If there are no duplicates, the loss is mostly
        # from from 0.33 -> 1; so when there are multiple 1's, there will be multiple 0.33 -> 1, increasing the loss)
        # So this parameter calculates the loss, then divides by number of ones, then multiply by batch size
        # 
        # >>> target1 = torch.tensor([[1,0,0], [0,1,0], [0,0,1]]).float()
        # >>> target2 = torch.tensor([[1,0,1], [0,1,0], [1,0,1]]).float()
        # >>> x1 = torch.tensor([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
        # >>> x2 = torch.tensor([[0.5, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
        # >>> torch.nn.functional.cross_entropy(x1, target1.float())         # tensor(1.0986)
        # >>> torch.nn.functional.cross_entropy(x1, target2.float())         # tensor(1.8310)
        # >>> torch.nn.functional.cross_entropy(x1, target2.float()) / 5 * 3 # tensor(1.0986)
        # 
        # >>> torch.nn.functional.cross_entropy(x2, target1.float())         # tensor(1.0619)
        # >>> torch.nn.functional.cross_entropy(x2, target2.float())         # tensor(1.8143)
        # >>> torch.nn.functional.cross_entropy(x2, target2.float()) / 5 * 3 # tensor(1.0886)
        if self.hyperparameters.get('normalize_loss_scale_inbatch_duplicates', False):
            loss = loss / torch.sum(ground_truth)
            loss = loss * self.hyperparameters['train_batch_size']

        if self.hyperparameters.get('scale_loss_by_batch_size', False):
            loss = loss / self.hyperparameters['train_batch_size']


        self.log(f"loss_{batch_dataloader_id}", loss, prog_bar=True)

        output = {
            'loss': loss
        }

        if return_individual_losses:
            output['loss_rule'] = loss_rule
            output['loss_sent'] = loss_sent
        if return_logits:
            output['input1_embs'] = input1_embs
            output['input2_embs'] = input2_embs
        if return_embs:
            output['logits_per_rule'] = logits_per_rule
            output['logits_per_sent'] = logits_per_sent

        return loss

    def training_step(self, batch, return_logits=False, return_embs=False, return_individual_losses=False, *args, **kwargs):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size
        """
        if isinstance(batch, list):
            loss = sum([scale_factor * self.training_step_one_batch(x, batch_dataloader_id=i) for i, (x, scale_factor) in enumerate(zip(batch, self.hyperparameters['loss_scalers']))]) / sum(self.hyperparameters['loss_scalers'])
        else:
            loss = self.training_step_one_batch(batch)

        self.log("loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0, *args, **kwargs):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size

        Here, B should be
        """
        input1_ids = batch['rule_id']
        input2_ids = batch['sentence_id']

        # Non-cached indices
        input1_ids_not_cached = [(i, x) for i, x in enumerate(input1_ids) if x not in self.val_rule_encodings[dataloader_idx]]
        input2_ids_not_cached = [(i, x) for i, x in enumerate(input2_ids) if x not in self.val_sentence_encodings[dataloader_idx]]

        # Calculate embeddings for non-cached, then cache
        if len(input1_ids_not_cached) > 0:
            # Slice to keep only the not cached ones, then run through the encoder
            input1_notcached_embs = self.encode_rule({k:v[[i for (i, x) in input1_ids_not_cached]] for (k, v) in batch['input1'].items()}).detach().cpu().numpy()
            # Store the ones that were not stored
            for ((_, x), embedding) in zip(input1_ids_not_cached, input1_notcached_embs):
                self.val_rule_encodings[dataloader_idx][x] = embedding

        if len(input2_ids_not_cached) > 0:
            # Slice to keep only the not cached ones, then run through the encoder
            input2_notcached_embs = self.encode_sent({k:v[[i for (i, x) in input2_ids_not_cached]] for (k, v) in batch['input2'].items()}).detach().cpu().numpy()
            # Store the ones that were not stored
            for ((_, x), embedding) in zip(input2_ids_not_cached, input2_notcached_embs):
                self.val_sentence_encodings[dataloader_idx][x] = embedding


        input1_embs = np.vstack([self.val_rule_encodings[dataloader_idx][x] for x in input1_ids])
        input2_embs = np.vstack([self.val_sentence_encodings[dataloader_idx][x] for x in input2_ids])
    
        o = {
            'input1_embs': input1_embs,
            'input2_embs': input2_embs,
            **{k:v for (k, v) in batch.items() if k not in ['input1', 'input2']}, # Add everything else
        }
        self.val_step_outputs[dataloader_idx].append(o)
        return o

    def validation_step123(self, batch, batch_idx: int, dataloader_idx: int = 0, *args, **kwargs):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size

        Here, B should be
        """
        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]

        input1_embs = self.encode_rule(input1) # [B, D]
        input2_embs = self.encode_sent(input2) # [B, D]

        o = {
            'input1_embs': input1_embs.detach().cpu().numpy(),
            'input2_embs': input2_embs.detach().cpu().numpy(),
            **{k:v for (k, v) in batch.items() if k not in ['input1', 'input2']}, # Add everything else
        }
        self.val_step_outputs[dataloader_idx].append(o)
        return o


    def on_validation_epoch_end(self, *args, **kwargs):
        all_results = {}
        for key in self.val_step_outputs.keys():
            outputs = self.val_step_outputs[key]

            print("dev_default_predictions_path", self.hyperparameters['dev_default_predictions_path'], key)
            if self.hyperparameters['dev_default_predictions_path'] != [] and self.hyperparameters['dev_default_predictions_path'] is not None and len(self.hyperparameters['dev_default_predictions_path']) > key:
                dev_default_predictions_path = self.hyperparameters['dev_default_predictions_path'][key]
            else:
                dev_default_predictions_path = {}


            rules     = np.vstack([x['input1_embs'] for x in outputs])
            sents     = np.vstack([x['input2_embs'] for x in outputs])
            ids       = [y for x in outputs for y in x['id']]
            rels      = [y for x in outputs for y in x['ss_relation']]
            gold_rels = [y for x in outputs for y in x['ts_relation']]
            
            rule_types = [y for x in outputs for y in x['rule_types']]
            potential_rule_types = sorted(list(set(rule_types)))

            rule_direction     = [y for x in outputs for y in x['rule_direction']]
            sentence_direction = [y for x in outputs for y in x['sentence_direction']]

            rule_entity_ids     = [y for x in outputs for y in x['rule_entity_ids']]
            sentence_entity_ids = [y for x in outputs for y in x['sentence_entity_ids']]

            rule_lexical_entities_ids     = [y for x in outputs for y in x['rule_lexical_entities']]
            sentence_lexical_entities_ids = [y for x in outputs for y in x['sentence_lexical_entities']]

            

            logit_scale = self.logit_scale.exp().detach().cpu().numpy()
            # rules = logit_scale * rules

            # This will hold a map from episode_id (i.e. `0` is for the first episode, etc) to indices in `rules` and `sents`
            # for data for that episode
            ids_to_position = defaultdict(list)
            # Map from episode id to goldrel of that test sentence
            id_to_goldrel = defaultdict(list)
            for i, x in enumerate(ids):
                ids_to_position[x].append(i)
                id_to_goldrel[x].append(gold_rels[i])
            
            for (episode_id, gold_rels_for_episode) in id_to_goldrel.items():
                assert(len(set(gold_rels_for_episode)) == 1)
            
            id_to_goldrel = {episode_id:gold_rels_for_episode[0] for (episode_id, gold_rels_for_episode) in id_to_goldrel.items()}
            

            ids_to_position = ids_to_position.items()
            ids_to_position = sorted(ids_to_position, key = lambda x: x[0])

            all_pred_rels = {}
            final_gold = None
            for rt in potential_rule_types:
                gold        = []
                pred_scores = []
                relations   = []
                for episode_id, positions in ids_to_position:
                    if self.hyperparameters.get('enforce_rule_direction', False):
                        positions = [x for x in positions if rule_direction[x] == sentence_direction[x]]
                    if self.hyperparameters.get('enforce_entity_types', False):
                        positions = [x for x in positions if rule_entity_ids[x] == sentence_entity_ids[x]]
                    positions = [x for x in positions if rule_types[x] == rt]
                    # print(positions)
                    rules_for_episode = rules[positions]
                    sents_for_episode = sents[positions]
                    similarities = rules_for_episode @ sents_for_episode.T
                    # print(similarities)
                    # print("\n")
                    similarities = similarities.diagonal()

                    # Aggregate similarities corresponding to the same underlying relation
                    from_rel_to_sim = defaultdict(list)
                    for i, pos in enumerate(positions):
                        if self.hyperparameters.get('boost_when_same_lexicalized_entities', False) and rule_lexical_entities_ids[pos] == sentence_lexical_entities_ids[pos]:
                            from_rel_to_sim[rels[pos]].append(1.0)
                        else:
                            from_rel_to_sim[rels[pos]].append(similarities[i])
                    from_rel_to_sim = dict(from_rel_to_sim)
                    from_rel_to_sim = {k:sorted(v, reverse=True) for (k, v) in from_rel_to_sim.items()}

                    from_rel_to_sim_keys = sorted(list(from_rel_to_sim.keys()))

                    # If there is no rule to be matched we will assume it is `no_relation` with max similarity
                    if len(from_rel_to_sim_keys) == 0:
                        from_rel_to_sim_keys = ['no_relation']
                        similarities = [1.0]
                    else:
                        similarities = [np.mean(from_rel_to_sim[rel][:self.hyperparameters.get('how_many_rules_to_average', 1)]) for rel in from_rel_to_sim_keys]

                    pred_scores.append(similarities)
                    relations.append(from_rel_to_sim_keys)
                    gold.append(id_to_goldrel[episode_id])

                results = compute_results_with_thresholds(gold=gold, pred_scores=pred_scores, pred_relations=relations, thresholds=self.thresholds, verbose=False, overwrite_results=dev_default_predictions_path)
                best = max(results, key=lambda x: x['f1_tacred'])
                final_output = compute_results_with_thresholds(gold=gold, pred_scores=pred_scores, pred_relations=relations, thresholds=[best['threshold']], verbose=True, overwrite_results=dev_default_predictions_path, return_gold=True, return_pred=True)

                if final_gold is None:
                    final_gold = gold

                all_pred_rels[rt] = final_output['pred']
                from src.utils import tacred_score
                result = [s * 100 for s in tacred_score(gold, pred, verbose=verbose)]
                print(result)
                exit()
                print("\n")
                print("-"*20)
                print("VAL", key, self.hyperparameters['dev_path'][key])
                print("Prediction score")
                print("Logit scale", self.logit_scale)
                print("Rule Type", rt)
                # print(relations)
                print("#"*5)
                print(best)
                print("-"*20)
                print("\n")
            
            gold = final_gold
            pred = []
            for line in pd.DataFrame(all_pred_rels).to_dict('records'):
                counter = Counter(line.values())
                max_value = max(counter.items(), key=lambda x: x[1])[1]
                only_rels_with_max_value = [x[0] for x in counter.items() if x[1] == max_value]
                if len(only_rels_with_max_value) == 1:
                    pred.append(only_rels_with_max_value[0])
                else:
                    pred.append(random.choice(only_rels_with_max_value))

            print(0, np.round(pred_scores[0], decimals=2),  relations[0],  gold[0])
            print(1, np.round(pred_scores[1], decimals=2),  relations[1],  gold[1])
            print(2, np.round(pred_scores[2], decimals=2),  relations[2],  gold[2])
            print(3, np.round(pred_scores[3], decimals=2),  relations[3],  gold[3])
            print(4, np.round(pred_scores[4], decimals=2),  relations[4],  gold[4])
            print(5, np.round(pred_scores[5], decimals=2),  relations[5],  gold[5])
            print(6, np.round(pred_scores[6], decimals=2),  relations[6],  gold[6])
            print(7, np.round(pred_scores[7], decimals=2),  relations[7],  gold[7])
            print(8, np.round(pred_scores[8], decimals=2),  relations[8],  gold[8])
            print(9, np.round(pred_scores[9], decimals=2),  relations[9],  gold[9])
            
            print(10, np.round(pred_scores[10], decimals=2), relations[10], gold[10])
            print(11, np.round(pred_scores[11], decimals=2), relations[11], gold[11])
            print(12, np.round(pred_scores[12], decimals=2), relations[12], gold[12])
            print(13, np.round(pred_scores[13], decimals=2), relations[13], gold[13])
            print(14, np.round(pred_scores[14], decimals=2), relations[14], gold[14])
            print(15, np.round(pred_scores[15], decimals=2), relations[15], gold[15])
            print(16, np.round(pred_scores[16], decimals=2), relations[16], gold[16])
            print(17, np.round(pred_scores[17], decimals=2), relations[17], gold[17])
            print(18, np.round(pred_scores[18], decimals=2), relations[18], gold[18])
            print(19, np.round(pred_scores[19], decimals=2), relations[19], gold[19])
            
            print(20, np.round(pred_scores[20], decimals=2), relations[20], gold[20])
            print(21, np.round(pred_scores[21], decimals=2), relations[21], gold[21])
            print(22, np.round(pred_scores[22], decimals=2), relations[22], gold[22])
            print(23, np.round(pred_scores[23], decimals=2), relations[23], gold[23])
            print(24, np.round(pred_scores[24], decimals=2), relations[24], gold[24])
            print(25, np.round(pred_scores[25], decimals=2), relations[25], gold[25])
            print(26, np.round(pred_scores[26], decimals=2), relations[26], gold[26])
            print(27, np.round(pred_scores[27], decimals=2), relations[27], gold[27])
            print(28, np.round(pred_scores[28], decimals=2), relations[28], gold[28])
            print(29, np.round(pred_scores[29], decimals=2), relations[29], gold[29])


            all_results[key] = results

            if self.hyperparameters.get("append_results_to_file", None):
                with open(self.hyperparameters["append_results_to_file"], 'a+') as fout:
                    dump = {
                        'dataloader_idx': key,
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        **best,
                        'dataloader_path': self.hyperparameters['dev_path'][key],
                        'logdir': self.trainer.log_dir,
                        'datetime': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                        'all_results': results,
                        'args': self.hyperparameters
                    }
                    _=fout.write(json.dumps(dump))
                    _=fout.write('\n')

        results = all_results[0]
        best = max(results, key=lambda x: x['f1_tacred'])
        self.log('threshold', best['threshold'])
        self.log('p_tacred',  best['p_tacred'])
        self.log('r_tacred',  best['r_tacred'])
        self.log('f1_tacred', best['f1_tacred'])
        
        print(best)
        # Reset val step outputs
        self.val_step_outputs = defaultdict(list)

        if self.hyperparameters.get('quit_training_prematurely_flag', False):
            if self.global_step > self.hyperparameters.get('quit_training_prematurely_steps', 1e9):
                exit("Reached max number of steps and `quit_training_prematurely_flag` is set to `True`.")

        self.val_rule_encodings = defaultdict(dict)
        self.val_sentence_encodings = defaultdict(dict)


        return {
            'best_score': best,
            '': '',
        }

    def configure_optimizers(self):
        parameters = [
            {"params": self.model_rule.parameters(), "lr": self.hyperparameters['model_rule_lr']},
            {"params": self.model_sentence.parameters(), "lr": self.hyperparameters['model_sentence_lr']},
            {
                "params": itertools.chain(
                    self.rule_projection.parameters(),
                    self.sent_projection.parameters(),
                ),
                "lr": self.hyperparameters['head_lr'],
                "weight_decay": self.hyperparameters['weight_decay'],
            },
            {"params": self.logit_scale, "lr": self.hyperparameters['logit_scale_lr']},
        ]
        optimizer = optim.AdamW(parameters, weight_decay=self.hyperparameters['weight_decay'], betas=(0.9,0.98), eps=1e-6)
        print("Esitmated steps:", self.trainer.estimated_stepping_batches)
        total_steps = self.trainer.estimated_stepping_batches
        # lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps)
        # lr_scheduler = optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     # mode="max",
        #     # patience=self.hyperparameters['lr_scheduler_patience'],
        #     # factor=self.hyperparameters['lr_scheduler_factor'],
        # )
        # return ([optimizer], [lr_scheduler])
        return {
            "optimizer": optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step', # or 'epoch'
                'frequency': 1,
            }
            # "lr_scheduler": lr_scheduler,
        }

    def on_before_zero_grad(self, *args, **kwargs):
        # clamp the weights here
        # See 
        # https://github.com/KeremTurgutlu/self_supervised/blob/2e7a7dc418891edccbf01efc0ab03d5e48586c8d/self_supervised/multimodal/clip.py#L346
        # https://lightning.ai/forums/t/where-to-clamp-weights/433
        # See https://github.com/openai/CLIP/issues/48 for why to clip and why to use logit scale
        # print("Yeaah Buddyyyy", self.logit_scale.data)
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        # print("Yeaah Buddyyyy", self.logit_scale.data)


    def tokenize(self, examples, max_length=384, padding=False):
        """
        Tokenize a list of examples the way this model expects
        Padding is set, by default, to False because we pad to the
        longest in the batch (it is more time efficient during training)
        """
        rules     = []
        sentences = []
        for i in range(len(examples['rule'])):
            if examples['rule'][i]:
                rules.append(examples['rule'][i])
        for i in range(len(examples['sentence'])):
            if examples['sentence'][i]:
                sentences.append(examples['sentence'][i])

        rules_tokenized     = self.tokenizer_rule(rules, max_length=max_length, padding=padding, truncation=True)
        sentences_tokenized = self.tokenizer_sentence(sentences, max_length=max_length, padding=padding, truncation=True)
        return {
            **{f'{k}_rule':     v for (k, v) in rules_tokenized.items()},
            **{f'{k}_sentence': v for (k, v) in sentences_tokenized.items()},
            # 'input1_relation': examples['input1_relation'],
            # 'input2_relation': examples['input2_relation'],
            # 'rule_id'        : examples['rule_id'],
            # 'sentence_id'    : examples['sentence_id'],
        }

    def tokenize_rules(self, rules, max_length=128, padding=False):
        rules_tokenized = self.tokenizer_rule(rules, max_length=max_length, padding=padding, truncation=True)
        return rules_tokenized

    def tokenize_sentences(self, sentences, max_length=128, padding=False):
        sentences_tokenized = self.tokenizer_rule(sentences, max_length=max_length, padding=padding, truncation=True)
        return sentences_tokenized

    def process_and_tokenize(self, examples, max_length=384, padding=False):
        """
        Process and Tokenize a list of examples the way this model expects
        Process -> (1) augment rule (if flag set)
                   (2) entity markers (+ augment, if flag set))
        Padding is set, by default, to False because we pad to the
        longest in the batch (it is more time efficient during training)
        """
        rules     = []
        sentences = []

        rules_ids     = []
        sentences_ids = []
        for i in range(len(examples['rule'])):
            if examples['rule'][i]:
                if self.hyperparameters.get('augment_rules', True):
                    rules.append(replace_rule_entity_types(examples['rule'][i].lower()))
                else:
                    rules.append(examples['rule'][i].lower())
                rules_ids.append(examples['rule_id'][i])
        for i in range(len(examples['sentence'])):
            if examples['sentence'][i]:
                if self.hyperparameters.get('augment_sentences', True):
                    if self.hyperparameters.get('preprocessing_type', 'typed_entity_marker_punct') == "typed_entity_marker_punct":
                        sentences.append(typed_entity_marker_punct(examples['sentence'][i]))
                    else:
                        sentences.append(typed_entity_marker_punct_v2(examples['sentence'][i]))
                else:
                    sentences.append(preprocess_line(examples['sentence'][i], self.hyperparameters['preprocessing_type']))
                sentences_ids.append(examples['sentence_id'][i])

        rules_tokenized     = self.tokenizer_rule(rules, max_length=max_length, padding=padding, truncation=True)
        sentences_tokenized = self.tokenizer_sentence(sentences, max_length=max_length, padding=padding, truncation=True)
        return {
            **{f'{k}_rule'    : v for (k, v) in rules_tokenized.items()},
            **{f'{k}_sentence': v for (k, v) in sentences_tokenized.items()},
            'rule_id'         : rules_ids, 
            'sentence_id'     : sentences_ids, 
            # 'input1_relation': examples['input1_relation'],
            # 'input2_relation': examples['input2_relation'],
            # 'rule_id'        : examples['rule_id'],
            # 'sentence_id'    : examples['sentence_id'],
        }

    def collate_tokenized_fn(self, batch: List[Dict[str, Any]], keep_columns: List[str] = ['id', 'relation', 'ss_relation', 'ts_relation', 'rule_id', 'sentence_id', 'rule_direction', 'sentence_direction', 'rule_entity_ids', 'sentence_entity_ids', 'rule_lexical_entities', 'sentence_lexical_entities']) -> Dict[str, Any]:
        """
        Define how this model expects the data to be batched
        """
        # Replace our suffixes because HF Collator's expect strict input names
        rule_inputs     = [{k[:-5]: v for (k ,v) in x.items() if k.endswith('_rule')} for x in batch]
        sentence_inputs = [{k[:-9]: v for (k ,v) in x.items() if k.endswith('_sentence')} for x in batch]

        # A primitive way of keeping ids around (not `input_id` though, since that is not one of our added ids (comes from tokenizer))
        ids = defaultdict(list)
        for el in batch:
            for (k, v) in el.items():
                if k in keep_columns:
                    ids[k].append(v)

        return {
            'input1'         : self.data_collator_rule(rule_inputs),
            'input2'         : self.data_collator_sentence(sentence_inputs),
            **ids,
            # 'input1_relation': [x['input1_relation'] for x in batch],
            # 'input2_relation': [x['input2_relation'] for x in batch],
            # 'rule_id'        : [x['rule_id'] for x in batch],
            # 'sentence_id'    : [x['sentence_id'] for x in batch],
        }

    def collate_untokenized_fn(self, batch: List[Dict[str, Any]], keep_columns: List[str] = ['id', 'relation', 'ss_relation', 'ts_relation', 'rule_id', 'sentence_id']) -> Dict[str, Any]:
        """
        Define how this model expects the data to be batched
        """
        rule_inputs     = self.tokenizer_rule([x['rule'] for x in batch], max_length=self.hyperparameters.get('max_length', 72), padding='longest', truncation=True)
        sentence_inputs = self.tokenizer_sentence([x['sentence'] for x in batch], max_length=self.hyperparameters.get('max_length', 72), padding='longest', truncation=True)

        # A primitive way of keeping ids around (not `input_id` though, since that is not one of our added ids (comes from tokenizer))
        ids = defaultdict(list)
        for el in batch:
            for (k, v) in el.items():
                if k in keep_columns:
                    ids[k].append(v)

        return {
            'input1'         : self.data_collator_rule(rule_inputs),
            'input2'         : self.data_collator_sentence(sentence_inputs),
            **ids,
            # 'input1_relation': [x['input1_relation'] for x in batch],
            # 'input2_relation': [x['input2_relation'] for x in batch],
            # 'rule_id'        : [x['rule_id'] for x in batch],
            # 'sentence_id'    : [x['sentence_id'] for x in batch],
        }

    def get_model_tokenization_signature(self):
        return hashlib.sha256(self.hyperparameters['model_name_or_checkpoint_rule'].encode()).hexdigest()[:5] + "_" + hashlib.sha256(self.hyperparameters['model_name_or_checkpoint_sentence'].encode()).hexdigest()[:5]


def read_rules(paths: List[str]):
    from collections import defaultdict
    result = defaultdict(list)
    for path in paths:
        with open(path) as fin:
            for line in fin:
                loaded_line = json.loads(line)
                line = {**loaded_line,
                        'query': loaded_line['query'].replace('[entity=B-', '[entity=').replace('[entity=I-', '[entity=').replace(']+  [', ']+ [').lower()
                }
                result[line['line_to_hash']].append(line)
    # return dict(result)
    return result

def read_valdata(dev_path: str, preprocessing_type: str, rules: Dict[str, dict]):
    """
    Val data in the form of few-shot episodes
    """

    print(f"Read valdata {dev_path}")
    with open(dev_path) as fin:
        val_data = json.load(fin)
    
    rule_to_id         = {}
    sentence_to_id     = {}
    entity_types_to_id = {}
    entity_to_id       = {}
    val = []
    idx = -1
    rule_idx = -1
    sent_idx = -1
    ent_type_idx  = -1
    ent_idx  = -1
    for episode, selections, relations in zip(val_data[0], val_data[1], val_data[2]):
        for ts in episode['meta_test']:
            sent_idx += 1
            idx += 1
            episode_ss = [y for x in episode['meta_train'] for y in x]
            episode_ss_rules = [y for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True) if 'line_to_hash' not in s else s['line_to_hash']]]
            relations = [s['relation'] for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True) if 'line_to_hash' not in s else s['line_to_hash']]]

            processed_sentence = preprocess_line(ts, preprocessing_type)
            if processed_sentence not in sentence_to_id:
                sentence_to_id[processed_sentence] = sent_idx
                
            for rule, ss_relation in zip(episode_ss_rules, relations):
                ent_type_idx  += 1
                rule_idx += 1
                rule_query = rule['query'].lower().replace(']+  [', ']+ [').replace('=b-', '=').replace('=i-', '=')
                if rule_query not in rule_to_id:
                    rule_to_id[rule_query] = rule_idx
                    
                if ts['obj_start'] > ts['subj_end']:
                    fe = ' '.join(ts['token'][ts['subj_start']:(ts['subj_end']+1)])
                    se = ' '.join(ts['token'][ts['obj_start']:(ts['obj_end']+1)])
                    if (ts['subj_type'], ts['obj_type']) not in entity_types_to_id:
                        entity_types_to_id[(ts['subj_type'], ts['obj_type'])] = ent_type_idx
                        ent_type_idx += 1
                    sentence_entity_ids = entity_types_to_id[(ts['subj_type'], ts['obj_type'])]
                else:
                    fe = ' '.join(ts['token'][ts['obj_start']:(ts['obj_end']+1)])
                    se = ' '.join(ts['token'][ts['subj_start']:(ts['subj_end']+1)])
                    if (ts['obj_type'], ts['subj_type']) not in entity_types_to_id:
                        entity_types_to_id[(ts['obj_type'], ts['subj_type'])] = ent_type_idx
                        ent_type_idx += 1
                    sentence_entity_ids = entity_types_to_id[(ts['obj_type'], ts['subj_type'])]

                if (fe, se) not in entity_to_id:
                    entity_to_id[(fe, se)] = ent_idx
                    ent_idx += 1

                if (rule['first_entity_type'], rule['second_entity_type']) not in entity_types_to_id:
                    entity_types_to_id[(rule['first_entity_type'], rule['second_entity_type'])] = ent_type_idx

                r_fe = ' '.join(rule['sentence_tokenized'][rule['first_entity_start']:rule['first_entity_end']]) # NOTE: no more `+1` here as we already add one when we created the rule file
                r_se = ' '.join(rule['sentence_tokenized'][rule['second_entity_start']:rule['second_entity_end']]) # NOTE: no more `+1` here as we already add one when we created the rule file

                if (r_fe, r_se) not in entity_types_to_id:
                    entity_types_to_id[(r_fe, r_se)] = ent_type_idx
                    ent_type_idx += 1

                val.append({'id': idx, 'rule': rule_query, 'sentence': processed_sentence, 'ss_relation': ss_relation, 'ts_relation': ts['relation'] if ts['relation'] in relations else 'no_relation', 'rule_id': rule_to_id[rule_query], 'sentence_id': sentence_to_id[processed_sentence], 'rule_direction': int(rule['subj_then_obj_order']), 'sentence_direction': int(ts['obj_start'] > ts['subj_end']), 'rule_entity_ids': entity_types_to_id[(rule['first_entity_type'], rule['second_entity_type'])], 'sentence_entity_ids': sentence_entity_ids, 'rule_lexical_entities': entity_types_to_id[(r_fe, r_se)], 'sentence_lexical_entities': entity_to_id[(fe, se)]})
    print(len(val))
    return val

def get_valdata(args, model, max_length=384, padding=False, load_from_cache_file=True):
    """
    Use `read_valdata` + tokenization
    """ 

    rules = read_rules(args['rules_path'])
    rules = dict(rules)

    result = []
    keep_columns = ['id', 'relation', 'ss_relation', 'ts_relation', 'rule_id', 'sentence_id', 'rule_direction', 'sentence_direction', 'rule_entity_ids', 'sentence_entity_ids', 'rule_lexical_entities', 'sentence_lexical_entities']
    for dev_path in args['dev_path']:
        val_data = datasets.Dataset.from_list(read_valdata(dev_path, args['preprocessing_type'], rules=rules))
        # val_data_tok = val_data.map(lambda x: model.tokenize(x, padding=padding, max_length=max_length), batched=True, cache_file_name=model.get_model_tokenization_signature() + "_" + str(max_length) + "_" + str(int(padding)), load_from_cache_file=True, remove_columns=[x for x in val_data.column_names if x not in keep_columns])
        val_data_tok = val_data.map(lambda x: model.tokenize(x, padding=padding, max_length=max_length), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in val_data.column_names if x not in keep_columns])
        result.append(val_data_tok)

    return result

def read_valdata_random(dev_path: str, preprocessing_type: str):
    """
    Val data in the form of few-shot episodes coming from random rules
    Important distinction, because this means that it contains the rules as well (see `ss['query']`)
    """
    with open(dev_path) as fin:
        val_data = json.load(fin)
    val = []
    idx = -1
    for episode, selections, relations in zip(val_data[0][:5000], val_data[1], val_data[2]):
        for ts in episode['meta_test']:
            idx += 1
            episode_ss = [y for x in episode['meta_train'] for y in x]
            relations = [s['relation'] for s in episode_ss]
            for ss, ss_relation in zip(episode_ss, relations):
                val.append({'id': idx, 'rule': ss['query'].lower(), 'sentence': preprocess_line(ts, preprocessing_type), 'ss_relation': ss_relation, 'ts_relation': ts['relation'] if ts['relation'] in relations else 'no_relation'})
    print(len(val))
    return val

def get_valdata_random(args, model):
    """
    Use `read_valdata_random` + tokenization
    """
    result = []
    keep_columns = ['id', 'ss_relation', 'ts_relation']
    for dev_path in args['dev_path']:
        val_data = datasets.Dataset.from_list(read_valdata_random(dev_path, args['preprocessing_type']))
        val_data_tok = val_data.map(lambda x: model.tokenize(x, padding=False, max_length=384), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in val_data.column_names if x not in keep_columns])
        result.append(val_data_tok)

    return result

def prepare_data_step4(args):
    """
    Store the data (to avoid memory-issues)
    Reading can then be done with `datasets.load_dataset('json', data_files=['<>'])`
    """
    query_to_id  = {}
    tokens_to_id = {}

    idx = -1
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/es/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s4.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/s/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s4.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/esp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/enhanced_syntax_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/sp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/surface_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id (but we store the new one as well)
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/query_to_id.json', 'w+') as fout:
        json.dump(query_to_id, fout)

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/tokens_to_id.json', 'w+') as fout:
        json.dump(tokens_to_id, fout)


    # Save to keep keys (some Huggingface Datasets issue where Dataset.from_list does not respect column order for inner dicts)
    sentence_keys = ['query', 'token', 'matched_tokens', 'matched_by_rule', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type', 'rule_type'] # Has to match order from the above two datasets
    data_esp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)
            
            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_esp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})
    
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/esp/data2.jsonl', 'w+') as fout:
        for line in data_esp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


    data_sp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)

            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_sp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/sp/data2.jsonl', 'w+') as fout:
        for line in data_sp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


def prepare_data_step3(args):
    """
    Store the data (to avoid memory-issues)
    Reading can then be done with `datasets.load_dataset('json', data_files=['<>'])`
    """
    query_to_id  = {}
    tokens_to_id = {}

    idx = -1
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/es/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s3.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/s/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s3.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/esp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/enhanced_syntax_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/sp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/surface_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id (but we store the new one as well)
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/query_to_id.json', 'w+') as fout:
        json.dump(query_to_id, fout)

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/tokens_to_id.json', 'w+') as fout:
        json.dump(tokens_to_id, fout)


    # Save to keep keys (some Huggingface Datasets issue where Dataset.from_list does not respect column order for inner dicts)
    sentence_keys = ['query', 'token', 'matched_tokens', 'matched_by_rule', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type', 'rule_type'] # Has to match order from the above two datasets
    data_esp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)
            
            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_esp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})
    
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/esp/data2.jsonl', 'w+') as fout:
        for line in data_esp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


    data_sp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)

            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_sp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step3/sp/data2.jsonl', 'w+') as fout:
        for line in data_sp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


def prepare_data_step2(args):
    """
    Store the data (to avoid memory-issues)
    Reading can then be done with `datasets.load_dataset('json', data_files=['<>'])`
    """
    query_to_id  = {}
    tokens_to_id = {}

    idx = -1
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/es/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s2.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/s/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s2.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/esp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/enhanced_syntax_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/sp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/surface_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id (but we store the new one as well)
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/query_to_id.json', 'w+') as fout:
        json.dump(query_to_id, fout)

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/tokens_to_id.json', 'w+') as fout:
        json.dump(tokens_to_id, fout)


    # Save to keep keys (some Huggingface Datasets issue where Dataset.from_list does not respect column order for inner dicts)
    sentence_keys = ['query', 'token', 'matched_tokens', 'matched_by_rule', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type', 'rule_type'] # Has to match order from the above two datasets
    data_esp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)
            
            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_esp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})
    
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/esp/data2.jsonl', 'w+') as fout:
        for line in data_esp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


    data_sp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)

            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_sp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step2/sp/data2.jsonl', 'w+') as fout:
        for line in data_sp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


def prepare_data_step1(args):
    """
    Store the data (to avoid memory-issues)
    Reading can then be done with `datasets.load_dataset('json', data_files=['<>'])`
    """
    query_to_id  = {}
    tokens_to_id = {}

    idx = -1
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/es/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/enhanced_syntax_all_s1.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/s/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/surface_all_s1.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')


    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/esp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/enhanced_syntax_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/sp/data.jsonl', 'w+') as fout:
        with open('/storage/rvacareanu/code/projects_7_2309/softrules/src/isolated_experiments/4_paraphrase_random_data/data/231202/merged_dataset/surface_paraphrase_0_250k.jsonl') as fin:
            for i, line in tqdm.tqdm(enumerate(fin)):
                idx += 1
                loaded_line = json.loads(line)
                query = loaded_line['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx
                if len(preprocessed_line.split()) > 66:
                    continue
                if len(query.split()) > 11:
                    continue

                idx + 1
                # For paraphrases we use the original line for the id (but we store the new one as well)
                preprocessed_line_original = preprocess_line({k.replace('_original', ''):v for (k, v) in loaded_line.items() if '_original' in k}, args['preprocessing_type'])
                if preprocessed_line_original not in tokens_to_id:
                    tokens_to_id[preprocessed_line_original] = idx
                
                line = {
                    'id': idx,
                    'rule': query, # replace a double space if there is one
                    'sentence': loaded_line,
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line_original],
                }
                _=fout.write(json.dumps(line))
                _=fout.write('\n')

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/query_to_id.json', 'w+') as fout:
        json.dump(query_to_id, fout)

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/tokens_to_id.json', 'w+') as fout:
        json.dump(tokens_to_id, fout)


    # Save to keep keys (some Huggingface Datasets issue where Dataset.from_list does not respect column order for inner dicts)
    sentence_keys = ['query', 'token', 'matched_tokens', 'matched_by_rule', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type', 'rule_type'] # Has to match order from the above two datasets
    data_esp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)
            
            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_esp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})
    
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/esp/data2.jsonl', 'w+') as fout:
        for line in data_esp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


    data_sp = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/esp/data.jsonl') as fin:
        for i, line in tqdm.tqdm(enumerate(fin)):
            loaded_line = json.loads(line)

            sentence = {k:loaded_line['sentence'][k] for k in sentence_keys}
            data_sp.append({'id': loaded_line['id'], 'rule': loaded_line['rule'], 'sentence': sentence, 'rule_id': loaded_line['rule_id'], 'sentence_id': loaded_line['sentence_id']})

    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step1/sp/data2.jsonl', 'w+') as fout:
        for line in data_sp:
            _=fout.write(json.dumps(line))
            _=fout.write('\n')




def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint_rule",     type=str,            default='bert-base-cased')
    parser.add_argument("--model_name_or_checkpoint_sentence", type=str,            default='bert-base-cased')
    parser.add_argument("--preprocessing_type",                type=str,            default='typed_entity_marker_punct')
    parser.add_argument("--rules_path",                        type=str, nargs='+', default=["/home/rvacareanu/projects_5_2210/rule_generation/fsre_dataset_rules/TACRED/enhanced_syntax.jsonl"])
    parser.add_argument("--train_path",                        type=str,            default="/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/few_shot_data/_train_data.json")
    parser.add_argument("--dev_path",                          type=str, nargs='+', default=["/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"])
    parser.add_argument("--dev_default_predictions_path",      type=str, nargs='+', default=[])
    parser.add_argument("--test_path",                         type=str,            default="/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json")
    parser.add_argument("--dropout",                           type=float,          default=0.0)
    parser.add_argument("--projection_dims",                   type=int,            default=256)
    parser.add_argument("--model_rule_lr",                     type=float,          default=1e-5)
    parser.add_argument("--model_sentence_lr",                 type=float,          default=1e-6)
    parser.add_argument("--head_lr",                           type=float,          default=1e-4)
    parser.add_argument("--weight_decay",                      type=float,          default=0.2)
    parser.add_argument("--lr_scheduler_patience",             type=float,          default=1.0)
    parser.add_argument("--lr_scheduler_factor",               type=float,          default=0.8)
    parser.add_argument("--logit_scale_lr",                    type=float,          default=1e-5)

    parser.add_argument("--max_epochs",                        type=int,   default=3)
    parser.add_argument("--max_steps",                         type=int,   default=10000)
    parser.add_argument("--val_check_interval_float",          type=float, default=0.5)
    parser.add_argument("--val_check_interval_int",            type=int,   default=5000)
    parser.add_argument("--gradient_clip_val",                 type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches",           type=int,   default=1)
    parser.add_argument("--train_batch_size",                  type=int,   default=64)
    parser.add_argument("--val_batch_size",                    type=int,   default=64)

    parser.add_argument("--how_many_rules_to_average",         type=int,   default=3,    help='There can be multiple rules associated with the same underlying relation; This parameter controls how many to average (mean topK style)')

    parser.add_argument("--es_duplicate",                     type=int,   default=1,    help='If to duplicate the `es` data when concatenatid (default=1)')
    parser.add_argument("--s_duplicate",                      type=int,   default=1,    help='If to duplicate the `s` data when concatenatid (default=1)')
    parser.add_argument("--esp_duplicate",                    type=int,   default=1,    help='If to duplicate the `esp` data when concatenatid (default=1)')
    parser.add_argument("--sp_duplicate",                     type=int,   default=1,    help='If to duplicate the `sp` data when concatenatid (default=1)')
    parser.add_argument("--es_take_max",                      type=int,   default=None, help='How much data to take (max); When `None`, take all')
    parser.add_argument("--s_take_max",                       type=int,   default=None, help='How much data to take (max); When `None`, take all')
    parser.add_argument("--esp_take_max",                     type=int,   default=None, help='How much data to take (max); When `None`, take all')
    parser.add_argument("--sp_take_max",                      type=int,   default=None, help='How much data to take (max); When `None`, take all')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing_enable',     action='store_true')
    parser.add_argument('--train_rule_model_from_scratch',     action='store_true')
    parser.add_argument('--train_sentence_model_from_scratch', action='store_true')

    parser.add_argument('--augment_rules',     action='store_true')
    parser.add_argument('--augment_sentences', action='store_true')

    parser.add_argument('--normalize_loss_scale_inbatch_duplicates',      action='store_true', help="The loss can vary a lot based on the in-batch duplicates; If this flag is set, we divide by the number of 1's, then multiply by batch size")
    parser.add_argument('--scale_loss_by_batch_size',                     action='store_true', help="If set, the loss will be divided by `train_batch_size`")

    parser.add_argument('--show_progress', action='store_true')

    parser.add_argument("--append_results_to_file",            type=str, required=False, help = "Where to append the results; If passed, will print some metrics in a JSON-lines format.")
    parser.add_argument("--run_description",                   type=str, required=False, help = "Some description of the run")

    parser.add_argument("--quit_training_prematurely_steps",    type=int,   default=25000, help='Only when `quit_training_prematurely_flag` is set to True; Will stop training after the first evaluation loop where global_step > this')
    parser.add_argument("--quit_training_prematurely_flag",     action='store_true', help='If set, will quit training prematurely based on `quit_training_prematurely_steps`. The idea of this is to enable fast experimentation, but to keep stuff that depends on the number of steps (e.g. LR scheduler) constant. We might not want a full training cycle.')

    parser.add_argument('--mix_train_data_with_random_data',   action='store_true', help="If set, we will mix the training data from `train_path` with the randomly generated data")
    parser.add_argument("--supervised_data_batch_size",        type=int,   default=64, help="Only used if `mix_train_data_with_random_data` is set to true.")
    parser.add_argument('--separate_paraphrase_from_exact_during_training',   action='store_true', help="If set, we will give the paraphrases in a different dataloader")

    parser.add_argument("--random_data_which_preprocessing_step_to_use", type=int, default=4, choices=[1,2,3,4], help="The data after which pre-processing step to load")

    parser.add_argument("--loss_scalers", type=int, nargs="+", default=[1,1,1], help="The factor with which to scale the loss (only used when we pass multiple dataloaders)")

    return parser

if __name__ == "__main__":
    # m = SoftRulesEncoder.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=0-step=50000.ckpt")
    # exit()
    args = vars(get_argparser().parse_args())
    print(args)
    pl.seed_everything(1)
    model = SoftRulesEncoder(
        hyperparameters = {
            **args,
            # 'model_name_or_checkpoint_rule'     : 'roberta-large',
            # 'model_name_or_checkpoint_sentence' : 'roberta-large',
            # 'model_name_or_checkpoint_rule'     : args['model_name_or_checkpoint_rule'],
            # 'model_name_or_checkpoint_sentence' : args['model_name_or_checkpoint_sentence'],
            'thresholds'                        : np.linspace(0.2, 1, 81).tolist(),
            # 'thresholds'                        : [0.52],
            # 'model_rule_lr'                     : args.get('model_rule_lr', 1e-5),
            # 'model_sentence_lr'                 : args.get('model_sentence_lr', 1e-6),
            # 'head_lr'                           : args.get('head_lr', 1e-4),
            # 'weight_decay'                      : args.get('weight_decay', 0.0),
            # 'logit_scale_lr'                    : args.get('logit_scale_lr', 1e-5),
        }
        # marker_type                       = args['preprocessing_type'],
        # correct_expected_score            = 1.0,
        # same_rel_expected_score           = 0.5,
        # diff_rel_expected_score           = 0.0,
    )
    
    with open(args['train_path']) as fin:
        train_data = json.load(fin)
    
    rules = read_rules(args['rules_path'])
    
    query_to_id  = {}
    tokens_to_id = {}

    data_tacred = []
    idx = -1
    for (relation, sentences) in train_data.items():
        for sentence in sentences:
            for rule in rules[line_to_hash(sentence, use_all_fields=True)]:
                idx += 1
                query = rule['query'].lower().replace(']+  [', ']+ [')
                if query not in query_to_id:
                    query_to_id[query] = idx
                preprocessed_line = preprocess_line(sentence, args['preprocessing_type'])
                if preprocessed_line not in tokens_to_id:
                    tokens_to_id[preprocessed_line] = idx

                data_tacred.append({
                    'id': idx,
                    'rule': replace_rule_entity_types(query),
                    'sentence': {k:v for (k, v) in sentence.items() if k in ['token', 'subj_start', 'subj_end', 'obj_start', 'obj_end', 'subj_type', 'obj_type']},
                    'rule_id': query_to_id[query],
                    'sentence_id': tokens_to_id[preprocessed_line],
                })
                
    data_tacred = datasets.Dataset.from_list(data_tacred)#.select(range(250_000))
    data_tacred.set_transform(lambda x: model.process_and_tokenize(x, padding=False, max_length=84))

    which_step = args['random_data_which_preprocessing_step_to_use']

    data_es = datasets.load_dataset('json', data_files=[f'/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step{which_step}/es/data.jsonl'], keep_in_memory=True)['train'].shuffle(seed=1)#.select(range(1_000_000))
    if args['es_take_max']:
        data_es = data_es.select(range(min(len(data_es), args['es_take_max'])))

    data_s = datasets.load_dataset('json', data_files=[f'/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step{which_step}/s/data.jsonl'], keep_in_memory=True)['train'].shuffle(seed=2)#.select(range(1_000_000))
    if args['s_take_max']:
        data_s = data_es.select(range(min(len(data_es), args['es_take_max'])))

    data_esp = datasets.load_dataset('json', data_files=[f'/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step{which_step}/esp/data2.jsonl'], keep_in_memory=True)['train'].shuffle(seed=3)
    if args['esp_take_max']:
        data_esp = data_es.select(range(min(len(data_es), args['es_take_max'])))

    data_sp = datasets.load_dataset('json', data_files=[f'/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step{which_step}/sp/data2.jsonl'], keep_in_memory=True)['train'].shuffle(seed=4)
    if args['sp_take_max']:
        data_sp = data_es.select(range(min(len(data_es), args['es_take_max'])))

    if args['separate_paraphrase_from_exact_during_training']:
        data_exact = datasets.concatenate_datasets([data_es] * args.get('es_duplicate', 1) +  [data_s] * args.get('s_duplicate', 1))
        data_paraphrase = datasets.concatenate_datasets([data_esp] * args.get('esp_duplicate', 1) +  [data_sp] * args.get('sp_duplicate', 1))
        data_exact.set_transform(lambda x: model.process_and_tokenize(x, padding=False, max_length=84))
        data_paraphrase.set_transform(lambda x: model.process_and_tokenize(x, padding=False, max_length=84))
        print(data_exact)
        print(data_exact[0])
        print(data_exact)
        print(data_exact[0])
        print(data_exact[0])
        print(data_exact[0])
        print(len(data_exact))
    else:
        data = datasets.concatenate_datasets([data_es] * args.get('es_duplicate', 1) +  [data_s] * args.get('s_duplicate', 1) + [data_esp] * args.get('esp_duplicate', 1) +  [data_sp] * args.get('sp_duplicate', 1))
        data.set_transform(lambda x: model.process_and_tokenize(x, padding=False, max_length=84))
        print(data)
        print(data[0])
        print(data)
        print(data[0])
        print(data[0])
        print(data[0])
        print(len(data))


    val_data = get_valdata(args, model=model)



    # model_checkpoint = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}-p={p_tacred:.2f}-r={r_tacred:.2f}-f1={f1_tacred:.2f}-thr={threshold:.3f}', monitor='f1_tacred', mode='max')
    model_checkpoint = ModelCheckpoint(every_n_train_steps=args['val_check_interval_int'], save_top_k=-1, save_on_train_epoch_end=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # es = EarlyStopping(monitor="f1_tacred", mode="max", patience=3)

    trainer = Trainer(
        accelerator="gpu", 
        precision='16-mixed' if args['fp16'] else 32,
        max_steps=args['max_steps'],
        val_check_interval=args['val_check_interval_int'] * args['accumulate_grad_batches'],
        # val_check_interval=5.0,#1000 * args['accumulate_grad_batches'],
        num_sanity_val_steps=0,
        # check_val_every_n_epoch=5,
        # max_epochs=args['max_epochs'],
        # val_check_interval=args['val_check_interval'],
        # check_val_every_n_epoch=1,
        gradient_clip_val=args['gradient_clip_val'],
        accumulate_grad_batches=args['accumulate_grad_batches'],
        log_every_n_steps=10,
        enable_progress_bar=args['show_progress'],
        callbacks=[model_checkpoint, lr_monitor]
    )

    # TODO perhaps refactor nested if statements
    if args['separate_paraphrase_from_exact_during_training']:
        if args['mix_train_data_with_random_data']:
            print("A")
            train_dataloaders = CombinedLoader([
                DataLoader(dataset=data_exact,       collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'],           shuffle=True, num_workers=0),
                DataLoader(dataset=data_paraphrase,  collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'],           shuffle=True, num_workers=0),
                DataLoader(dataset=data_tacred,      collate_fn=model.collate_tokenized_fn, batch_size=args['supervised_data_batch_size'], shuffle=True, num_workers=0),
            ], mode='max_size_cycle')
        else:
            print("B")
            train_dataloaders = CombinedLoader([
                DataLoader(dataset=data_exact,       collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'], shuffle=True, num_workers=0),
                DataLoader(dataset=data_paraphrase,  collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'], shuffle=True, num_workers=0),
            ], mode='max_size_cycle')
    else:
        if args['mix_train_data_with_random_data']:
            print("C")
            train_dataloaders = CombinedLoader([
                DataLoader(dataset=data,        collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'],           shuffle=True, num_workers=0),
                DataLoader(dataset=data_tacred, collate_fn=model.collate_tokenized_fn, batch_size=args['supervised_data_batch_size'], shuffle=True, num_workers=0),
            ], mode='max_size_cycle')
        else:
            print("D")
            train_dataloaders = DataLoader(dataset=data,        collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'], shuffle=True, num_workers=0)

    trainer.fit(
        model, 
        # train_dataloaders = DataLoader(dataset=data,  collate_fn=model.collate_tokenized_fn, batch_size=args['train_batch_size'], shuffle=True, num_workers=0),
        train_dataloaders = train_dataloaders,
        val_dataloaders   = [DataLoader(dataset=val_data_tok, collate_fn=model.collate_tokenized_fn, batch_size=args['val_batch_size'], num_workers=32) for val_data_tok in val_data],
    )

    # batch = model.collate_tokenized_fn([data_tok[0], data_tok[1], data_tok[2], data_tok[3]])
    # print(batch)
    # o = model.training_step(batch, return_individual_losses=True, return_embs=True)
    # print(o)

