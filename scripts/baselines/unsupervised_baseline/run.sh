#!/bin/bash

# TACRED
for K in 1 5
do
    for SEED in 0 1 2 3 4
    do
        python -m src.baseline.unsupervised_entity_based_classification \
            --dataset_path "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/TACRED/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_16029${SEED}.json" \
            >> results/231027/baselines/unsupervised/TACRED/logs_dev_k${K}_seed${SEED}.txt &
    done
done
wait

# NYT29
for K in 1 5
do
    for SEED in 0 1 2 3 4
    do
        python -m src.baseline.unsupervised_entity_based_classification \
            --dataset_path "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_16029${SEED}.json" \
            >> results/231027/baselines/unsupervised/NYT29/logs_dev_k${K}_seed${SEED}.txt &
    done
done
wait

# WIKIDATA
for K in 1 5
do
    for SEED in 0 1 2 3 4
    do
        python -m src.baseline.unsupervised_entity_based_classification \
            --dataset_path "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/WIKIDATA_episodes_231006//dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_16029${SEED}.json" \
            >> results/231027/baselines/unsupervised/WIKIDATA/logs_dev_k${K}_seed${SEED}.txt &
    done
done
wait
