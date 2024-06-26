#!/bin/bash

# Note: This is slightly different from checkpoint_501 because we are: (1) Using rules with lexicalized partial, and (2) use surface with words only and lexicalized partial
# In the original we used no lexicalized partial and for surface it was surface normal

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.model_cliplike --model_name_or_checkpoint_rule 'roberta-large' --model_name_or_checkpoint_sentence 'roberta-large' --how_many_rules_to_average 1\
    --model_rule_lr 3e-5 --model_sentence_lr 1e-5 --head_lr 1e-4 --logit_scale_lr 3e-4 --max_epochs 3 --train_batch_size 512 --val_batch_size 64 --gradient_clip_val 5.0 --dropout 0.1 --projection_dims 384 --weight_decay 0.001 \
    --run_description "Param search" \
    --random_data_which_preprocessing_step_to_use 4 \
    --gradient_checkpointing_enable \
    --augment_rules --augment_sentences \
    --rules_path \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized_partial.jsonl" \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_words_only_with_lexicalized_partial.jsonl" \
    --train_path "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/_train_data.json" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
    --dev_default_predictions_path \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_90.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
    --quit_training_prematurely_flag --quit_training_prematurely_steps 19000 \
    --preprocessing_type typed_entity_marker_punct_v2 \
    --val_check_interval_int 2000 --accumulate_grad_batches 1 --show_progress --max_steps 100000 \
    --append_results_to_file /storage/rvacareanu/code/projects_7_2309/softrules/results/240320/final_run/logs_settings_ps_large_v1.jsonl --fp16

