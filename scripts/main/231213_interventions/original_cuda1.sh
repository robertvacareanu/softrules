#!/bin/bash


# TACRED

# DEV

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval_interventions \
    --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
    --rules_path \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
    --default_preds_path \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_90.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_91.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_92.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_93.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_94.jsonl" \
    --append_results_to_file \
        "results/231213/interventions/original/tacred_dev_1.jsonl" > results/231213/interventions/original/tacred_dev_1.txt

# CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval_interventions \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_94.jsonl" \
#     --append_results_to_file \
#         "results/231213/interventions/original/tacred_dev_5.jsonl" > results/231213/interventions/original/tacred_dev_5.txt

