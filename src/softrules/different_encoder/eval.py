import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
from torch.utils.data import DataLoader
import argparse
import json

from src.softrules.different_encoder.model_cliplike_multidata import SoftRulesEncoder, read_rules, get_valdata

# CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true python -m src.softrules.different_encoder.eval --checkpoint /storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_220/checkpoints/epoch=0-step=65000.ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,)
    parser.add_argument("--how_many_rules_to_average", type=int, default=1)
    parser.add_argument("--dev_default_predictions_path",           type=str,   default=None, help='Predictions of the hard rules (if exists);')
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument("--rules_path", type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl", 
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29_clean/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29_cleanclean/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax.jsonl"
    ])
    parser.add_argument("--dev_path",   type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json",
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_clean/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_v2/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_norelx100/dev_episodesq/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_norelx1000/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/WIKIDATA/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/WIKIDATA/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
    ])
    parser.add_argument("--default_preds_path",   type=str, nargs='+', default=[
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K1_90.jsonl",
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K1_90.jsonl",
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_dev_K5_90.jsonl",
        "/storage/rvacareanu/code/projects_7_2309/softrules/default_predictions/TACRED/predictions_test_K5_90.jsonl",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_clean/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_v2/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_norelx100/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/NYT29_cleanclean_norelx1000/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/WIKIDATA/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/WIKIDATA/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
    ])

    args = vars(parser.parse_args())
    print(args)
    pl.seed_everything(1)
    model = SoftRulesEncoder.load_from_checkpoint(args['checkpoint'])
    model.thresholds = np.linspace(0, 1, 101).tolist()
    model.hyperparameters['append_results_to_file'] = None
    model.hyperparameters['dev_path'] = args['dev_path']
    model.hyperparameters['how_many_rules_to_average'] = args['how_many_rules_to_average']
    # rules = read_rules(args['rules'])
    # rules = dict(rules)
    val_data = get_valdata({**model.hyperparameters, **args}, model=model)

    trainer = Trainer(
        accelerator="gpu", 
        precision='16-mixed',
        enable_progress_bar=args['show_progress'],
    )
    
    print("-" * 100)
    print("Without default predictions")
    for vd, default_preds, vd_path in zip(val_data, args['default_preds_path'], args['dev_path']):
        model.hyperparameters['dev_default_predictions_path'] = None
        print("\n\n\n")
        print("-"*25)
        print(vd_path)
        print(vd)
        if '_1_shot' in vd_path.lower():
            model.hyperparameters['how_many_rules_to_average'] = 1
        elif '_5_shot' in vd_path.lower():
            model.hyperparameters['how_many_rules_to_average'] = 3
        else:
            raise ValueError("Unknown K")
        trainer.validate(model=model,dataloaders=DataLoader(dataset=vd, collate_fn=model.collate_tokenized_fn, batch_size=256, num_workers=64))
        print("-"*25)
        print("\n\n\n")

    print("-" * 100)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print("-" * 100)

    print("With default predictions")
    for vd, default_preds, vd_path in zip(val_data, args['default_preds_path'], args['dev_path']):
        with open(default_preds) as fin:
            dp = json.load(fin)
        print(f"With default preds: {dp}")
        model.hyperparameters['dev_default_predictions_path'] = default_preds
        print(model.hyperparameters['dev_default_predictions_path'])
        print("\n\n\n")
        print("-"*25)
        print(vd_path)
        print(vd)
        if '_1_shot' in vd_path.lower():
            model.hyperparameters['how_many_rules_to_average'] = 1
        elif '_5_shot' in vd_path.lower():
            model.hyperparameters['how_many_rules_to_average'] = 3
        else:
            raise ValueError("Unknown K")
        trainer.validate(model=model,dataloaders=DataLoader(dataset=vd, collate_fn=model.collate_tokenized_fn, batch_size=256, num_workers=64))
        print("-"*25)
        print("\n\n\n")
    print("-" * 100)

