#!/bin/bash


# TACRED

# TEST
# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval/tacred_test_1.jsonl" > results/240312/eval/tacred_test_1.txt

# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval/tacred_test_5.jsonl" > results/240312/eval/tacred_test_5.txt


        
# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval2/tacred_test_1.jsonl" > results/240312/eval2/tacred_test_1.txt

# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval2/tacred_test_5.jsonl" > results/240312/eval2/tacred_test_5.txt


# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval3/tacred_test_1.jsonl" > results/240312/eval3/tacred_test_1.txt

# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval3/tacred_test_5.jsonl" > results/240312/eval3/tacred_test_5.txt


CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
    --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
    --rules_path \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized_partial.jsonl" \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized_partial.jsonl" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
    --default_preds_path \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_91.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_92.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_93.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_94.jsonl" \
    --append_results_to_file \
        "results/240312/eval4/tacred_test_1.jsonl" > results/240312/eval4/tacred_test_1.txt

CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
    --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
    --rules_path \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized_partial.jsonl" \
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized_partial.jsonl" \
    --dev_path \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
    --default_preds_path \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_91.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_92.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_93.jsonl" \
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_94.jsonl" \
    --append_results_to_file \
        "results/240312/eval4/tacred_test_5.jsonl" > results/240312/eval4/tacred_test_5.txt


# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized_partial2.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized_partial2.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval5/tacred_test_1.jsonl" > results/240312/eval5/tacred_test_1.txt

# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval \
#     --checkpoint "/storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt" \
#     --rules_path \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized_partial2.jsonl" \
#         "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized_partial2.jsonl" \
#     --dev_path \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json" \
#         "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json" \
#     --default_preds_path \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_91.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_92.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_93.jsonl" \
#         "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_94.jsonl" \
#     --append_results_to_file \
#         "results/240312/eval5/tacred_test_5.jsonl" > results/240312/eval5/tacred_test_5.txt





