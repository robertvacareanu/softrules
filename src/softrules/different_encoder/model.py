from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import torch
from torch import nn
import pytorch_lightning as pl

from typing import Dict, List, Literal, Any
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np

class SoftRulesEncoder(nn.Module):
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
        model_name_or_checkpoint_rule: str, 
        model_name_or_checkpoint_sentence: str, 
        correct_expected_score : float = 1.0,
        same_rel_expected_score: float = 0.5,
        diff_rel_expected_score: float = 0.0,
    ):
        super().__init__()
        self.model_rule         = AutoModel.from_pretrained(model_name_or_checkpoint_rule)
        self.tokenizer_rule     = AutoTokenizer.from_pretrained(model_name_or_checkpoint_rule)
        self.data_collator_rule = DataCollatorWithPadding(tokenizer=self.tokenizer_rule, return_tensors="pt")

        self.model_sentence         = AutoModel.from_pretrained(model_name_or_checkpoint_sentence)
        self.tokenizer_sentence     = AutoTokenizer.from_pretrained(model_name_or_checkpoint_sentence)
        self.data_collator_sentence = DataCollatorWithPadding(tokenizer=self.tokenizer_sentence, return_tensors="pt")

        self.correct_expected_score  = correct_expected_score
        self.same_rel_expected_score = same_rel_expected_score
        self.diff_rel_expected_score = diff_rel_expected_score

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def predict(self, batch):
        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]
        
        input1_embs = self.model_rule.forward(**input1).pooler_output
        input2_embs = self.model_sentence.forward(**input2).pooler_output
        
        return input1_embs @ input2_embs.t()

    def training_step(self, batch, return_logits=False, return_embs=False):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size
        """
        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]
        
        input1_embs = self.model_rule.forward(**input1).pooler_output # [B, D]
        input2_embs = self.model_sentence.forward(**input2).pooler_output # [B, D]

        input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        
        logit_scale     = self.logit_scale.exp()
        logits_per_rule = logit_scale * input1_embs @ input2_embs.t() # [B, B]; [i, j] => similarity of rule i with sentence j
        logits_per_sent = logits_per_rule.t() # [B, B]; [i, j] => similarity of sentence i with rule j

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

        if return_logits:
            output['input1_embs'] = input1_embs
            output['input2_embs'] = input2_embs
        if return_embs:
            output['logits_per_rule'] = logits_per_rule
            output['logits_per_sent'] = logits_per_sent

        return output

    def predict_one(self, rules, sentences, n: int, k: int):
        """
        rules     -> [B * N * K]
        sentences -> [B]
        """
        input1_embs = self.model_rule.forward(**rules).pooler_output # [B * N * K, D]
        input2_embs = self.model_sentence.forward(**sentences).pooler_output # [B, D]

        batch_size = len(sentences)

        input1_embs = input1_embs.reshape(batch_size, n, k, -1).mean(axis=2) # [B, N, D]

        input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B * N * K, D]
        input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, D]


        predictions = torch.matmul(input1_embs, input2_embs.unsqueeze(dim=2)).squeeze(dim=1) # [B, N]

        return predictions.numpy()
        
    def predict_all(self, all_episodes):
        for episode, gold_positions, ss_t_relations in zip(all_episodes[0], all_episodes[1], all_episodes[2]):
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


    def validation_step(self, batch):
        """
        B -> batch size
        L -> sequence length
        D -> hidden size

        Here, B should be
        """
        input1 = batch['input1'] # Dict[str, torch.tensor]
        input2 = batch['input2'] # Dict[str, torch.tensor]

        input1_embs = self.model_rule.forward(**input1).pooler_output # [B, D]
        input2_embs = self.model_sentence.forward(**input2).pooler_output # [B, D]

        input1_embs = input1_embs / input1_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        input2_embs = input2_embs / input2_embs.norm(dim=1, keepdim=True) # [B, D] ; Normalize to compute cos sim
        
        return {
            'input1_embs': input1_embs,
            'input2_embs': input2_embs,
            'input1_relation': [x['input1_relation'] for x in batch],
            'input2_relation': [x['input2_relation'] for x in batch],
            'rule_id'        : [x['rule_id'] for x in batch],
            'sentence_id'    : [x['sentence_id'] for x in batch],
        }
        

    def tokenize(self, examples, max_length=384, padding=False):
        rules     = []
        sentences = []
        for i in range(len(examples['rule'])):
            if examples['rule'][i] and examples['sentence'][i]:
                rules.append(examples['rule'][i])
                sentences.append(examples['sentence'][i])

        rules_tokenized     = self.tokenizer_rule(rules, max_length=max_length, padding=padding, truncation=True)
        sentences_tokenized = self.tokenizer_sentence(sentences, max_length=max_length, padding=padding, truncation=True)
        return {
            **{f'{k}_rule': v     for (k, v) in rules_tokenized.items()},
            **{f'{k}_sentence': v for (k, v) in sentences_tokenized.items()},
            'input1_relation': examples['input1_relation'],
            'input2_relation': examples['input2_relation'],
            'rule_id'        : examples['rule_id'],
            'sentence_id'    : examples['sentence_id'],
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Define how this model expects the training data to be batched
        """
        # Replace our suffixes because HF Collator's expect strict input names
        rule_inputs     = [{k[:-5]: v for (k ,v) in x.items() if k.endswith('_rule')} for x in batch]
        sentence_inputs = [{k[:-9]: v for (k ,v) in x.items() if k.endswith('_sentence')} for x in batch]

        return {
            'input1'         : self.data_collator_rule(rule_inputs),
            'input2'         : self.data_collator_sentence(sentence_inputs),
            'input1_relation': [x['input1_relation'] for x in batch],
            'input2_relation': [x['input2_relation'] for x in batch],
            'rule_id'        : [x['rule_id'] for x in batch],
            'sentence_id'    : [x['sentence_id'] for x in batch],
        }



if __name__ == "__main__":
    model = SoftRulesEncoder(
        model_name_or_checkpoint_rule     = 'bert-base-cased',
        model_name_or_checkpoint_sentence = 'bert-base-cased',
        correct_expected_score            = 1.0,
        same_rel_expected_score           = 0.5,
        diff_rel_expected_score           = 0.0,
    )
    import datasets
    examples = [
        {'id': 1, 'rule': 'rule1',      'sentence': 'sentence1', 'relation': 'r1'},
        {'id': 2, 'rule': 'rule2rule2', 'sentence': 'sentence2sentence2', 'relation': 'r2'},
        {'id': 3, 'rule': 'rule3rule3rule3rule3', 'sentence': 'sentence3sentence3sentence3', 'relation': 'r3'},
        {'id': 4, 'rule': 'rule4rule4rule4rule4rule4rule4', 'sentence': 'sentence4sentence4', 'relation': 'r4'},
        {'id': 5, 'rule': 'rule5', 'sentence': 'sentence5', 'relation': 'r5'},
    ]
    data = datasets.Dataset.from_list(examples)
    data_tok = data.map(lambda x: model.tokenize(x, padding=False), batched=True, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in data.column_names if x != 'id'])
    batch = model.collate_fn([data_tok[0], data_tok[1], data_tok[2], data_tok[3]])
    print(batch)
    o = model.training_step(batch)
    print(o)


