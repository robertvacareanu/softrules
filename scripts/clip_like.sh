#!/bin/bash


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.model_cliplike --model_name_or_checkpoint_rule 'bert-base-cased' --model_name_or_checkpoint_sentence 'bert-base-cased' --model_rule_lr 1e-5 --model_sentence_lr 1e-6 --head_lr 1e-4 --logit_scale_lr 1e-5 --max_epochs 5 --train_batch_size 64
