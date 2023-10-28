#!/bin/bash

for MARKER in 'entity_mask' #'entity_marker' 'entity_marker_punct' 'typed_entity_marker' 'typed_entity_marker_punct'
do
    for K in 1 5
    do
        python -m src.baseline.sentence_pair \
            --marker_type $MARKER \
            --do_train \
            --training_path \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/few_shot_data/_train_data.json" \
            --evaluation_paths \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160290.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160291.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160292.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160293.json" \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160294.json" \
            --find_threshold_on_path \
                "/data/nlp/corpora/fs-re-dataset-paper/Few-Shot_Datasets/NYT29/episodes/dev_episodes/5_way_${K}_shots_10K_episodes_3q_seed_160290.json" \
            --append_results_to_file "results/231027/baselines/sentence_pair/nyt29_1shot.jsonl" \
            >> results/231027/baselines/sentence_pair/NYT29/${MARKER}_seedsall_k${1}.txt
    done
done
