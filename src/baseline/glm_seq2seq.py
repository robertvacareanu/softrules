"""

A baseline that leverages GLM to do the classification

"""
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, set_seed
import datasets
import json
import tqdm
import argparse
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any, List
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from scipy.special import expit
from src.utils import tacred_score
from src.baseline.entity_marker_cross_encoder import preprocess_line
from transformers.trainer_utils import PredictionOutput

def preprocess_function(examples, tokenizer, max_input_length=384, max_target_length=3, padding='max_length'):
    inputs, targets = [], []
    for i in range(len(examples['input'])):
        if examples['input'][i] and examples['output'][i]:
            inputs.append(examples['input'][i])
            targets.append(examples['output'][i])

    model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def sents_to_glm_data_input(sent_a, sent_b):
    return f'Is the same relation represented in the following two examples?\nSentence 1: {sent_a}\nSentence 2: {sent_b}'

def construct_training_dataset(kwargs):
    training_path = kwargs['training_path']
    with open(training_path) as fin:
        training_data = json.load(fin)
        training_data = [{**s, 'relation': relation} for (relation, sentences) in training_data.items() for s in sentences]


    # Sample 
    data = []
    for i in range(kwargs['finetuning_examples'] // 4):
        sent_a = random.choice(training_data)

        while sent_a['relation'] == 'no_relation':
            sent_a = random.choice(training_data)
        sent_a_tokens = preprocess_line(sent_a, preprocessing_type=kwargs['marker_type'])

        # 1/4 same
        sent_b = random.choice(training_data)
        while sent_b['relation'] == sent_a['relation']:
            sent_b = random.choice(training_data)
        sent_b_tokens = preprocess_line(sent_b, preprocessing_type=kwargs['marker_type'])

        assert(sent_a['relation'] != sent_b['relation'])
        data.append({
            'input' : sents_to_glm_data_input(sent_a_tokens, sent_b_tokens),
            'output': 'No'
        })

        # 3/4 different
        for _ in range(3):
            sent_c = random.choice(training_data)
            while sent_c['relation'] != sent_a['relation']:
                sent_c = random.choice(training_data)
            sent_c_tokens = preprocess_line(sent_c, preprocessing_type=kwargs['marker_type'])
            assert(sent_a['relation'] == sent_c['relation'])

            data.append({
                'input' : sents_to_glm_data_input(sent_a_tokens, sent_c_tokens),
                'output': 'Yes'
            })

    data = datasets.Dataset.from_list(data)
    print(data)
    return data

def construct_eval_dataset_from_episodes(kwargs):
    all_episodes = kwargs['all_episodes']

    data = []
    test_sentence_id = -1
    for episode_id, (episode, gold_positions, ss_t_relations) in tqdm.tqdm(enumerate(zip(all_episodes[0], all_episodes[1], all_episodes[2])), total=len(all_episodes[0])):
        meta_train = episode['meta_train']
        meta_test  = episode['meta_test']
        
        for test_sentence, gold_relation_position in zip(meta_test, gold_positions):
            test_sentence_id += 1

            for support_sentence_id, (support_sentences_for_relation, ss_relation) in enumerate(zip(meta_train, ss_t_relations[0])):
                for support_sentence_id_for_rel, support_sentence in enumerate(support_sentences_for_relation):
                    sent_a_tokens = preprocess_line(support_sentence, preprocessing_type=kwargs['marker_type'])
                    sent_b_tokens = preprocess_line(test_sentence, preprocessing_type=kwargs['marker_type'])

                    data.append({
                        'input' : sents_to_glm_data_input(sent_a_tokens, sent_b_tokens),
                        # Not ideal, but `preprocess_function` expects an `output`. And to not give gold output (i.e. if `support_sentence` has same relation as `test_sentence`),
                        # we give `''`
                        'output': 'Dummy', 
                        'episode_id': episode_id,
                        'test_sentence_id': test_sentence_id,
                        'support_sentence_id': support_sentence_id,
                        'support_sentence_id_for_rel': support_sentence_id_for_rel,
                        'support_sentence_relation': ss_relation,
                        'gold_relation': 'no_relation' if gold_relation_position >= len(ss_t_relations[0]) else ss_t_relations[0][gold_relation_position],
                    })
            
    return data

def compute_gold_pred(preds: PredictionOutput, dataset: List[Dict[str, Any]], tokenizer):
    decoded = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    assert(len(decoded) == len(dataset))
    number_of_episodes = max(dataset, key=lambda x: x['test_sentence_id'])['test_sentence_id']
    print(dataset[0])
    print(dataset[-1])
    all_preds = [[] for _ in range(number_of_episodes + 1)]
    all_rels  = [[] for _ in range(number_of_episodes + 1)]
    gold      = [[] for _ in range(number_of_episodes + 1)]

    # Unwrap the predictions using the extra fields stored
    for decoded_answer, line in zip(decoded, dataset):
        if decoded_answer == 'Yes':
            all_preds[line['test_sentence_id']].append(decoded_answer)
            all_rels[line['test_sentence_id']].append(line['support_sentence_relation'])
        gold[line['test_sentence_id']].append(line['gold_relation'])

    # Sanity checks
    for g in gold:
        if len(set(g)) != 1:
            print(g)
            exit()
        assert(len(set(g)) == 1)
    
    gold = [g[0] for g in gold]

    pred = []
    for (p, r) in zip(all_preds, all_rels):
        if len(p) == 0:
            pred.append('no_relation')
        elif len(p) == 1:
            pred.append(r[0])
        else:
            pred.append(random.choice(r))

    # score = tacred_score(gold, pred, verbose=True)
    
    return gold, pred


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    model     = AutoModelForSeq2SeqLM.from_pretrained(args['model_name'])

    train_data = construct_training_dataset(args).map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['input', 'output'], load_from_cache_file=False, cache_file_name=None)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args['output_dir'],
        fp16                        = args['fp16'],
        # fp16_backend                = "amp",
        per_device_train_batch_size = args['per_device_train_batch_size'],
        per_device_eval_batch_size  = args['per_device_eval_batch_size'],
        # eval_accumulation_steps     = 16,
        # evaluation_strategy         = "steps",
        # eval_steps                  = 5000,      #logging_steps,
        save_steps                  = args['save_steps'],
        logging_steps               = args['logging_steps'],
        save_total_limit            = args['save_total_limit'],
        max_steps                   = args['max_steps'],
        gradient_accumulation_steps = args['gradient_accumulation_steps'],
        # report_to                   = "wandb",
        remove_unused_columns       = False,
        # weight_decay                = 0.001,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = 'linear',
        dataloader_num_workers      = 16,
        learning_rate               = args['learning_rate'],
        predict_with_generate       = True,
    )


    collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        label_pad_token_id = -100,
        pad_to_multiple_of = 8,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_data,
        # eval_dataset    = dataset['validation'],
        # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
        tokenizer       = tokenizer,
        data_collator   = collator,
        # compute_metrics = {'en_ner': lambda x: compute_metrics(x[0], x[1], 'en_ner'), 'fr_ner': lambda x: compute_metrics(x[0], x[1], 'fr_ner'), }
    )

    # Train, if necessary
    if args['do_train']:
        trainer.train()

    # Evaluate
    for evaluation_path in args['evaluation_paths']:
        with open(evaluation_path) as fin:
            test_episodes = json.load(fin)

        eval_dataset = construct_eval_dataset_from_episodes({**args, 'all_episodes': test_episodes})#[:50000]

        eval_dataset_hf = datasets.Dataset.from_list(eval_dataset)
        preds = trainer.predict(eval_dataset_hf.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=eval_dataset_hf.column_names, load_from_cache_file=False, cache_file_name=None))
        
        gold, pred = compute_gold_pred(preds, eval_dataset, tokenizer)
        
        print("##########")
        print(evaluation_path)
        try:
            score = tacred_score(gold, pred, verbose=True)
        except Exception as e:
            score = ''
            print(e)
        print("##########")

    # return score, gold, pred, preds

def get_huggingface_args(parent_parser):
    """
    Some parameters specific to huggingface ecosystem
    """
    subparser = parent_parser.add_argument_group("huggingface")
    subparser.add_argument('--fp16',                        action='store_true', help='Whether to train with half precision or not')
    subparser.add_argument('--output_dir',                  type=str,   default='logs/seq2seq/',  help='Where to save the model')
    subparser.add_argument('--per_device_train_batch_size', type=int,   default=64,       help='How many batches per device (train)')
    subparser.add_argument('--per_device_eval_batch_size',  type=int,   default=64,       help='How many batches per device (eval)')
    subparser.add_argument('--save_steps',                  type=int,   default=1000,    help='After how many steps to save')
    subparser.add_argument('--logging_steps',               type=int,   default=100,      help='After how many steps to log')
    subparser.add_argument('--save_total_limit',            type=int,   default=1,        help='How many to save in total')
    subparser.add_argument('--max_steps',                   type=int,   default=1000,    help='Max steps for training')
    subparser.add_argument('--gradient_accumulation_steps', type=int,   default=1,        help='The number of batches to accumulate the gradient for before calling `optimize.step()`')
    subparser.add_argument('--learning_rate',               type=float, default=3e-4,     help='The learning rate')

    return parent_parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--marker_type',            type=str, default='entity_mask', help="How to mark the entities", choices=['entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct'])
    parser.add_argument('--model_name',             type=str, default='google/t5-v1_1-small', help="What model to use")
    parser.add_argument('--finetuning_examples',    type=int, default=50_000, help="How many examples to fine-tune on")
    parser.add_argument('--training_path',          type=str, help="What to train on.")
    parser.add_argument('--evaluation_paths',       type=str, nargs='*', help="What to evaluate on")
    parser.add_argument('--seed',                   type=int, default=1, help="The random seed to use")
    parser.add_argument('--append_results_to_file', type=str, help="Appends results to this file")
    parser.add_argument('--do_train', action='store_true', required=False, help="Whether to fine-tune the model or not")
    get_huggingface_args(parser)
    args = vars(parser.parse_args())
    print(args)

    seed_everything(args['seed'])
    # eval_dataset = main(args)
    # score, gold, pred, preds = main(args)
    main(args)

    # print(score)


    