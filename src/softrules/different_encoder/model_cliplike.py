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
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from src.utils import line_to_hash
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from typing import Dict, List, Literal, Any
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.utils import compute_results_with_thresholds
from src.softrules.projection_head import ProjectionHead

import numpy as np
import scipy as sp
import scipy.special as spp

import datasets

from collections import defaultdict

import itertools

import torch.optim as optim

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
        self.model_rule         = AutoModel.from_pretrained(self.hyperparameters['model_name_or_checkpoint_rule'])
        self.tokenizer_rule     = AutoTokenizer.from_pretrained(self.hyperparameters['model_name_or_checkpoint_rule'])
        self.data_collator_rule = DataCollatorWithPadding(tokenizer=self.tokenizer_rule, return_tensors="pt")

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

    def training_step(self, batch, return_logits=False, return_embs=False, return_individual_losses=False, *args, **kwargs):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size
        """
        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]
        
        input1_embs = self.encode_rule(input1) # [B, D]
        input2_embs = self.encode_sent(input2) # [B, D]

        input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        
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

        ground_truth = torch.arange(input1_embs.size(0)).to(logits_per_rule.device)
        loss_rule = torch.nn.functional.cross_entropy(logits_per_rule, ground_truth)
        loss_sent = torch.nn.functional.cross_entropy(logits_per_sent, ground_truth)
        loss = (loss_rule + loss_sent) / 2

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

        return output

    def predict_episode(self, rules, sentences, n: int, k: int):
        """
        rules     -> [B * N * K]
        sentences -> [B * N * K] (duplicated N*K times)
        """
        input1_embs = self.model_rule.forward(**rules).pooler_output # [B * N * K, D]
        input2_embs = self.model_sentence.forward(**sentences).pooler_output # [B * N * K, D]


        batch_size = input2_embs.size(0) // (n * k)

        # print(input1_embs.shape)
        # print(input2_embs.shape)
        # print(batch_size, len(sentences), n, k)

        input1_embs = input1_embs.reshape(batch_size, n, k, -1).mean(axis=2) # [B, N, D]
        input2_embs = input2_embs.reshape(batch_size, n * k, -1) # [B, N * K, D]
        # print(input1_embs.shape)
        input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B, N, D]
        input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, N * K, D]
        # print(input1_embs.shape)
        # print(input2_embs.shape)


        # logit_scale = self.logit_scale.exp()
        # input1_embs = logit_scale * input1_embs
        input1_embs = input1_embs

        logits_per_rule = input1_embs @ input2_embs.transpose(1,2) # [B, N, N]
        # print("-"*10)
        # print(logits_per_rule)
        # print("-"*10)
        logits_per_rule = logits_per_rule[:, :, 0] # [B, N] (we keep only `0` because sentences are duplicated to have (rule, sent) pairs)
        # predictions = torch.matmul(input1_embs, input2_embs.unsqueeze(dim=2)).squeeze(dim=1) # [B, N]
        # print(logit_scale)
        # print(logits_per_rule.shape)
        # print(logits_per_rule.squeeze())
        # exit()

        return logits_per_rule.detach().numpy()


    # def predict_episode(self, rules, sentences, n: int, k: int):
    #     """
    #     rules     -> [B * N * K]
    #     sentences -> [B]
    #     """
    #     input1_embs = self.model_rule.forward(**rules).pooler_output # [B * N * K, D]
    #     input2_embs = self.model_sentence.forward(**sentences).pooler_output # [B, D]

    #     print(input1_embs.shape)
    #     print(input2_embs.shape)
    #     exit()

    #     batch_size = len(sentences)

    #     input1_embs = input1_embs.reshape(batch_size, n, k, -1).mean(axis=2) # [B, N, D]

    #     input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B * N * K, D]
    #     input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, D]


    #     predictions = torch.matmul(input1_embs, input2_embs.unsqueeze(dim=2)).squeeze(dim=1) # [B, N]

    #     return predictions.numpy()
        
    # def predict_all(self, all_episodes, kwargs):
    #     for episode, gold_positions, ss_t_relations in zip(all_episodes[0], all_episodes[1], all_episodes[2]):
    #         meta_train = episode['meta_train']
    #         meta_test  = episode['meta_test']
            
    #         to_be_predicted = []
    #         for test_sentence, gold_relation_position in zip(meta_test, gold_positions):
    #             relations            = []
    #             for support_sentences_for_relation, ss_relation in zip(meta_train, ss_t_relations[0]):
    #                 for support_sentence in support_sentences_for_relation:
    #                     to_be_predicted.append([preprocess_line(support_sentence, preprocessing_type=kwargs['marker_type']), preprocess_line(test_sentence, preprocessing_type=kwargs['marker_type'])])
    #                 relations.append(ss_relation)
                
    #             if gold_relation_position >= len(relations):
    #                 gold.append('no_relation')
    #             else:
    #                 gold.append(relations[gold_relation_position])

    #             pred_relations.append(relations)

    #         pred_scores += expit(model.predict(to_be_predicted).reshape(len(meta_test), len(meta_train), -1).mean(axis=2)).tolist()


    def validation_step(self, batch, *args, **kwargs):
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

        logit_scale     = self.logit_scale.exp()
        logits_per_rule = logit_scale * input1_embs @ input2_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        logits_per_sent = logit_scale * input2_embs @ input1_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        
        return {
            'input1_embs': input1_embs.detach().cpu().numpy(),
            'input2_embs': input2_embs.detach().cpu().numpy(),
            **{k:v for (k, v) in batch.items() if k not in ['input1', 'input2']}, # Add everything else
        }

    def validation_epoch_end(self, outputs: List):
        rules     = np.vstack([x['input1_embs'] for x in outputs])
        sents     = np.vstack([x['input2_embs'] for x in outputs])
        ids       = [y for x in outputs for y in x['id']]
        rels      = [y for x in outputs for y in x['ss_relation']]
        gold_rels = [y for x in outputs for y in x['ts_relation']]

        # logit_scale = self.logit_scale.exp().detach().cpu().numpy()
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

        gold        = []
        pred_scores = []
        relations   = []
        for episode_id, positions in ids_to_position:
            rules_for_episode = rules[positions]
            sents_for_episode = sents[positions]
            similarities = rules_for_episode @ sents_for_episode.T
            similarities = similarities[:, 0]
            similarities = spp.expit(similarities).tolist()
            pred_scores.append(similarities)
            relations.append([rels[i] for i in positions])
            gold.append(id_to_goldrel[episode_id])

        print("-"*20)
        print("VAL")
        print("Prediction score")
        print("Logit scale", self.logit_scale)
        print(pred_scores[0])
        print(pred_scores[1])
        print(pred_scores[2])
        print(pred_scores[3])
        print("-"*20)
        # print(relations)
        results = compute_results_with_thresholds(gold=gold, pred_scores=pred_scores, pred_relations=relations, thresholds=self.thresholds, verbose=False)
        best = max(results, key=lambda x: x['f1_tacred'])

        self.log('val/p_tacred',  best['p_tacred'])
        self.log('val/r_tacred',  best['r_tacred'])
        self.log('val/f1_tacred', best['f1_tacred'])
        
        print(best)

        return best

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
        optimizer = optim.Adam(parameters, weight_decay=self.hyperparameters['weight_decay'])
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.hyperparameters['lr_scheduler_patience'],
            factor=self.hyperparameters['lr_scheduler_factor'],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/f1_tacred",
        }

    def on_before_zero_grad(self, *args, **kwargs):
        # clamp the weights here
        # See 
        # https://github.com/KeremTurgutlu/self_supervised/blob/2e7a7dc418891edccbf01efc0ab03d5e48586c8d/self_supervised/multimodal/clip.py#L346
        # https://lightning.ai/forums/t/where-to-clamp-weights/433
        # See https://github.com/openai/CLIP/issues/48 for why to clip and why to use logit scale
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)


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
            **{f'{k}_rule': v     for (k, v) in rules_tokenized.items()},
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

    def collate_fn(self, batch: List[Dict[str, Any]], keep_columns: List[str] = ['id', 'relation', 'ss_relation', 'ts_relation', ]) -> Dict[str, Any]:
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


def read_rules(path: str):
    from collections import defaultdict
    result = defaultdict(list)
    with open(path) as fin:
        for line in fin:
            line = json.loads(line)
            result[line['line_to_hash']].append(line)
    return result

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint_rule",     type=str,   default='bert-base-cased')
    parser.add_argument("--model_name_or_checkpoint_sentence", type=str,   default='bert-base-cased')
    parser.add_argument("--preprocessing_type",                type=str,   default='typed_entity_marker_punct')
    parser.add_argument("--rules_path",                        type=str,   default="/home/rvacareanu/projects_5_2210/rule_generation/fsre_dataset_rules/TACRED/enhanced_syntax.jsonl")
    parser.add_argument("--train_path",                        type=str,   default="/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/few_shot_data/_train_data.json")
    parser.add_argument("--dev_path",                          type=str,   default="/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json")
    parser.add_argument("--test_path",                         type=str,   default="/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json")
    parser.add_argument("--model_rule_lr",                     type=float, default=1e-5)
    parser.add_argument("--model_sentence_lr",                 type=float, default=1e-6)
    parser.add_argument("--head_lr",                           type=float, default=1e-4)
    parser.add_argument("--weight_decay",                      type=float, default=0.0)
    parser.add_argument("--lr_scheduler_patience",             type=float, default=1.0)
    parser.add_argument("--lr_scheduler_factor",               type=float, default=0.8)
    parser.add_argument("--logit_scale_lr",                    type=float, default=1e-5)

    parser.add_argument("--max_epochs",                        type=int,   default=3)
    parser.add_argument("--val_check_interval",                type=float, default=0.5)
    parser.add_argument("--accumulate_grad_batches",           type=int,   default=1)
    parser.add_argument("--train_batch_size",                  type=int,   default=64)
    parser.add_argument("--val_batch_size",                    type=int,   default=64)

    parser.add_argument('--show_progress', action='store_true')

    return parser

if __name__ == "__main__":
    args = vars(get_argparser().parse_args())
    print(args)
    model = SoftRulesEncoder(
        hyperparameters = {
            # 'model_name_or_checkpoint_rule'     : 'roberta-large',
            # 'model_name_or_checkpoint_sentence' : 'roberta-large',
            'model_name_or_checkpoint_rule'     : args['model_name_or_checkpoint_rule'],
            'model_name_or_checkpoint_sentence' : args['model_name_or_checkpoint_sentence'],
            'thresholds'                        : np.linspace(0, 1, 101).tolist(),
            # 'thresholds'                        : [0.52],
            'model_rule_lr'                     : args.get('model_rule_lr', 1e-5),
            'model_sentence_lr'                 : args.get('model_sentence_lr', 1e-6),
            'head_lr'                           : args.get('head_lr', 1e-4),
            'weight_decay'                      : args.get('weight_decay', 0.0),
            'lr_scheduler_patience'             : args.get('lr_scheduler_patience', 1.0),
            'lr_scheduler_factor'               : args.get('lr_scheduler_factor', 0.8),
            'logit_scale_lr'                    : args.get('logit_scale_lr', 1e-5),
        }
        # marker_type                       = args['preprocessing_type'],
        # correct_expected_score            = 1.0,
        # same_rel_expected_score           = 0.5,
        # diff_rel_expected_score           = 0.0,
    )
    rule_path = args['rules_path']
    rules = read_rules(rule_path)

    with open(args['train_path']) as fin:
        train_data = json.load(fin)
    
    data = []
    idx = -1
    for (relation, sentences) in train_data.items():
        for sentence in sentences:
            for rule in rules[line_to_hash(sentence, use_all_fields=True)]:
                idx += 1
                data.append({
                    'id': idx,
                    'rule': rule['query'].lower(),
                    'sentence': preprocess_line(sentence, args['preprocessing_type']),
                })
    
    data = datasets.Dataset.from_list(data)
    print(data)
    print(data[0])
    data_tok = data\
        .map(lambda x: model.tokenize(x, padding=False, max_length=96), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in data.column_names if x not in ['id', 'relation']]) \
        .filter(lambda x: sum(x['attention_mask_rule']) < 96 and sum(x['attention_mask_sentence']) < 96)
    print(data_tok)

    with open(args['dev_path']) as fin:
        val_data = json.load(fin)

    val = []
    idx = -1
    for episode, selections, relations in zip(val_data[0], val_data[1], val_data[2]):
        for ts in episode['meta_test']:
            idx += 1
            episode_ss = [y for x in episode['meta_train'] for y in x]
            episode_ss_rules = [y for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True)]]
            relations = [s['relation'] for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True)]]
            for rule, ss_relation in zip(episode_ss_rules, relations):
                val.append({'id': idx, 'rule': rule['query'].lower(), 'sentence': preprocess_line(ts, args['preprocessing_type']), 'ss_relation': ss_relation, 'ts_relation': ts['relation'] if ts['relation'] in relations else 'no_relation'})
    print(len(val_data))
    keep_columns = ['id', 'ss_relation', 'ts_relation']
    val_data = datasets.Dataset.from_list(val)
    val_data_tok = val_data.map(lambda x: model.tokenize(x, padding=False, max_length=384), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in val_data.column_names if x not in keep_columns])


    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    model_checkpoint = ModelCheckpoint()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        accelerator="gpu", 
        # max_steps=5000,
        # val_check_interval=2500,
        max_epochs=args['max_epochs'],
        val_check_interval=args['val_check_interval'],
        # check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args['accumulate_grad_batches'],
        log_every_n_steps=10,
        enable_progress_bar=args['show_progress'],
        callbacks=[model_checkpoint, lr_monitor]
    )
    trainer.fit(
        model, 
        train_dataloaders=DataLoader(dataset=data_tok, collate_fn=model.collate_fn, batch_size=args['train_batch_size'], shuffle=True),
        val_dataloaders=DataLoader(dataset=val_data_tok, collate_fn=model.collate_fn, batch_size=args['val_batch_size']),
    )

    # batch = model.collate_fn([data_tok[0], data_tok[1], data_tok[2], data_tok[3]])
    # print(batch)
    # o = model.training_step(batch, return_individual_losses=True, return_embs=True)
    # print(o)

def x():
    # train()
    exit()
    model = SoftRulesEncoder(
        hyperparameters = {
            'model_name_or_checkpoint_rule'     : 'bert-base-cased',
            'model_name_or_checkpoint_sentence' : 'bert-base-cased',
            'thresholds'                        : np.linspace(0, 1, 101).tolist(),
            'model_rule_lr'                     : 1e-5,
            'model_sentence_lr'                 : 1e-5,
            'head_lr'                           : 1e-4,
            'weight_decay'                      : 0.0,
            'lr_scheduler_patience'             : 1.0,
            'lr_scheduler_factor'               : 0.8,
        }
        # marker_type                       = args['preprocessing_type'],
        # correct_expected_score            = 1.0,
        # same_rel_expected_score           = 0.5,
        # diff_rel_expected_score           = 0.0,
    )
    
    examples = [
        {'id': 1, 'rule': 'rule1',      'sentence': 'sentence1', 'relation': 'r1'},
        {'id': 2, 'rule': 'rule2rule2', 'sentence': 'sentence2sentence2', 'relation': 'r2'},
        {'id': 3, 'rule': 'rule3rule3rule3rule3', 'sentence': 'sentence3sentence3sentence3', 'relation': 'r3'},
        {'id': 4, 'rule': 'rule4rule4rule4rule4rule4rule4', 'sentence': 'sentence4sentence4', 'relation': 'r4'},
        {'id': 5, 'rule': 'rule5', 'sentence': 'sentence5', 'relation': 'r5'},
    ]
    data = datasets.Dataset.from_list(examples)
    data_tok = data.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in data.column_names if x not in ['id', 'relation']])
    batch = model.collate_fn([data_tok[0], data_tok[1], data_tok[2], data_tok[3]])
    print(batch)
    o = model.training_step(batch, return_individual_losses=True)
    print(o)


    rule_path = "/home/rvacareanu/projects_5_2210/rule_generation/fsre_dataset_rules/TACRED/enhanced_syntax.jsonl"
    rules = read_rules(rule_path)

    with open("/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json") as fin:
        data = json.load(fin)

    val_data = []
    idx = -1
    for episode, selections, relations in zip(data[0], data[1], data[2]):
        for ts in episode['meta_test']:
            idx += 1
            episode_ss = [y for x in episode['meta_train'] for y in x]
            episode_ss_rules = [y for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True)]]
            relations = [s['relation'] for s in episode_ss for y in rules[line_to_hash(s, use_all_fields=True)]]
            for rule, ss_relation in zip(episode_ss_rules, relations):
                val_data.append({'id': idx, 'rule': rule['query'], 'sentence': preprocess_line(ts, args['preprocessing_type']), 'ss_relation': ss_relation, 'ts_relation': ts['relation'] if ts['relation'] in relations else 'no_relation'})
    print(len(val_data))
    keep_columns = ['id', 'ss_relation', 'ts_relation']
    val_dataset = datasets.Dataset.from_list(val_data)
    val_dataset_tok = val_dataset.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in val_dataset.column_names if x not in keep_columns])
    batch01 = model.collate_fn([val_dataset_tok[1], val_dataset_tok[2]], keep_columns=keep_columns)
    batch23 = model.collate_fn([val_dataset_tok[2], val_dataset_tok[3]], keep_columns=keep_columns)
    b01 = model.validation_step(batch01)
    b23 = model.validation_step(batch23)
    print(batch01)
    print(batch23)
    x = model.validation_epoch_end([b01, b23])
    print(x)
    exit()

    episode1 = data[0][0]
    episode1_ss = [y for x in episode1['meta_train'] for y in x]
    episode1_ts = [x for x in episode1['meta_test']]
    
    episode1_ss_rules = [rules[line_to_hash(s, use_all_fields=True)][0] for s in episode1_ss]
    examples_val = [{'id': i, 'rule': episode1_ss_rules[i]['query'], 'sentence': preprocess_line(episode1_ts[0], args['preprocessing_type'])} for i in range(len(episode1_ss_rules))]

    examples_val_rules = [{'id': i, 'rule': episode1_ss_rules[i]['query']} for i in range(len(episode1_ss_rules))]
    examples_val_sents = [{'id': i, 'sentence': preprocess_line(episode1_ts[0], args['preprocessing_type'])} for i in range(len(episode1_ss_rules))]

    examples_val_data = datasets.Dataset.from_list(examples_val)
    examples_val_data_tok = examples_val_data.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in examples_val_data.column_names if x != 'id'])
    
    # examples_val_data_rules = datasets.Dataset.from_list(examples_val_rules)
    # examples_val_data_rules_tok = examples_val_data_rules.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in examples_val_data.column_names if x != 'id'])
    # examples_val_data_sents = datasets.Dataset.from_list(examples_val_sents)
    # examples_val_data_sents_tok = examples_val_data_sents.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in examples_val_data.column_names if x != 'id'])
    batch = model.collate_fn(examples_val_data_tok)
    print(batch)
    o = model.training_step(batch, return_individual_losses=True)
    print(o)
    o = model.predict_episode(batch['input1'], batch['input2'], n=5, k=1)
    print(o)
