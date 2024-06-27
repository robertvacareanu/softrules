#!/bin/bash


# TACRED

# DEV

# No Augment rules/sentences
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval_interventions \
    --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
    --rules_path \
        "intervention_data/annotator2/enhanced_syntax.jsonl" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
    --default_preds_path \
        "intervention_data/annotator2/default_predictions/dev_k1_s90_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k1_s91_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k1_s92_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k1_s93_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k1_s94_rt_es_t0.json" \
    --append_results_to_file \
        "results/231213/interventions/intervention2_strictthreshold/tacred_dev_1.jsonl" > results/231213/interventions/intervention2_strictthreshold/tacred_dev_1.txt

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval_interventions \
    --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
    --rules_path \
        "intervention_data/annotator2/enhanced_syntax.jsonl" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
    --default_preds_path \
        "intervention_data/annotator2/default_predictions/dev_k5_s90_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k5_s91_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k5_s92_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k5_s93_rt_es_t0.json" \
        "intervention_data/annotator2/default_predictions/dev_k5_s94_rt_es_t0.json" \
    --append_results_to_file \
        "results/231213/interventions/intervention2_strictthreshold/tacred_dev_5.jsonl" > results/231213/interventions/intervention2_strictthreshold/tacred_dev_5.txt

