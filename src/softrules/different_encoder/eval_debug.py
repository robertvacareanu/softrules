"""

Peek into model's predictions

"""
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
from torch.utils.data import DataLoader
import argparse
import tqdm
import pandas as pd
import json
from src.baseline.entity_marker_cross_encoder import preprocess_line
import torch

from collections import defaultdict

import scipy as sp
import scipy.special as spp
from src.utils import line_to_hash

from src.softrules.different_encoder.model_cliplike import SoftRulesEncoder, read_rules, get_valdata


# CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true python -i -m src.softrules.different_encoder.eval_debug --checkpoint /storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_501/checkpoints/epoch=0-step=4000.ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,)
    parser.add_argument("--rules_path", type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29_clean/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29_cleanclean/enhanced_syntax.jsonl", 
        # "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax.jsonl"
    ])
    parser.add_argument("--dev_path",   type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
        # "/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json",
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
    model = SoftRulesEncoder.load_from_checkpoint(args['checkpoint']).eval()
    model.thresholds = np.linspace(0, 1, 101).tolist()
    model.hyperparameters['append_results_to_file'] = None
    ts = {'token': 'John and Mary are siblings'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 2, 'obj_end': 2, 'obj_type': 'person'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=person]+ sibling of [entity=person]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John and Mary are married'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 2, 'obj_end': 2, 'obj_type': 'person'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=person]+ sibling of [entity=person]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John and Mary are brother and sister'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 2, 'obj_end': 2, 'obj_type': 'person'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=sibling]+ and [entity=sibling]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John and Mary are married'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 2, 'obj_end': 2, 'obj_type': 'person'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=sibling]+ and [entity=sibling]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'organization'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=person]+ graduated from [entity=college]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'subject person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'object organization'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=subject person]+ graduated from [entity=object organization]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'object person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'subject organization'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=subject person]+ graduated from [entity=object person]+']).items()})
    print(r_embedding @ s_embedding.T)

    ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'object', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'subject'}
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=subject]+ graduated from [entity=object]+']).items()})
    print(r_embedding @ s_embedding.T)

    print("\n\n\n\n\n")

    sentence1 = "Sofia Coppola , daughter of Francis Ford Coppola , is one of the few to succeed in doing so : her film'' Lost in Translation'' won her a screenplay Oscar"
    ts1       = {'token': sentence1.split(), 'subj_start': 0, 'subj_end': 1, 'subj_type': 'person', 'obj_start': 5, 'obj_end': 7, 'obj_type': 'person', }
    rule1     = "[entity=person]+ >appos son >nmod_of [entity=person]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts1, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule1.lower()]).items()})
    print("-"*20)
    print(1)
    print(preprocess_line(ts1, 'typed_entity_marker_punct_v2'))
    print(rule1)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence2 = 'John got his degree from Oxford .'
    ts2       = {'token': sentence2.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'organization'}
    rule2     = "[entity=person]+ graduated from [entity=organization]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts2, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule2.lower()]).items()})
    print("-"*20)
    print(2)
    print(preprocess_line(ts2, 'typed_entity_marker_punct_v2'))
    print(rule2)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence3 = 'John moved to Athens , Greece .'
    ts3       = {'token': sentence3.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    rule3     = "[entity=location]+ <appos [entity=location]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts3, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule3.lower()]).items()})
    print("-"*20)
    print(3)
    print(preprocess_line(ts3, 'typed_entity_marker_punct_v2'))
    print(rule3)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence4 = 'John moved to SoHo , Manhattan .'
    ts4       = {'token': sentence4.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    rule4     = "[entity=Wynwood]+ <appos [entity=Miami]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts4, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule4.lower()]).items()})
    print("-"*20)
    print(4)
    print(preprocess_line(ts4, 'typed_entity_marker_punct_v2'))
    print(rule4)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence5 = 'John moved to Athens , Greece .'
    ts5       = {'token': sentence5.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    rule5     = "[entity=Wynwood]+ <appos [entity=Miami]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts5, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule5.lower()]).items()})
    print("-"*20)
    print(5)
    print(preprocess_line(ts5, 'typed_entity_marker_punct_v2'))
    print(rule5)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence6 = 'John moved to SoHo , Manhattan .'
    ts6       = {'token': sentence6.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    rule6     = "[entity=Berlin]+ <appos [entity=Germany]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts6, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule6.lower()]).items()})
    print("-"*20)
    print(6)
    print(preprocess_line(ts6, 'typed_entity_marker_punct_v2'))
    print(rule6)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)

    sentence7 = 'John moved to Athens , Greece .'
    ts7       = {'token': sentence7.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    rule7     = "[entity=Berlin]+ <appos [entity=Germany]+"
    s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts7, 'typed_entity_marker_punct_v2')]).items()}) 
    r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules([rule7.lower()]).items()})
    print("-"*20)
    print(7)
    print(preprocess_line(ts7, 'typed_entity_marker_punct_v2'))
    print(rule7)
    print(np.round((r_embedding @ s_embedding.T).detach().cpu().numpy().item(), 2))
    print("-"*20)



    # with open("/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_4q_seed_160290.json") as fin:
    #     data = json.load(fin)

    # rules = read_rules(["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl"])
    # rules = dict(rules)

    # ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'organization'}
    # s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    # r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=person]+ graduated from [entity=college]+']).items()})
    # print(r_embedding @ s_embedding.T)

    # ts = {'token': 'John got his degree from Oxford.'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'person', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'organization'}
    # s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    # r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=person]+ graduated from [entity=college]+']).items()})
    # print(r_embedding @ s_embedding.T)

    # ts = {'token': 'SoHo is part of Manhattan'.split(), 'subj_start': 0, 'subj_end': 0, 'subj_type': 'location', 'obj_start': 4, 'obj_end': 4, 'obj_type': 'location'}
    # s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    # r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=location]+ part of [entity=location]+']).items()})
    # print(r_embedding @ s_embedding.T)

    # ts = {'token': 'John moved to SoHo , Manhattan'.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    # s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    # r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=damascus]+ <appos [entity=syria]+']).items()})
    # print(r_embedding @ s_embedding.T)

    # ts = {'token': 'John moved to Berlin , Germany.'.split(), 'subj_start': 3, 'subj_end': 3, 'subj_type': 'location', 'obj_start': 5, 'obj_end': 5, 'obj_type': 'location'}
    # s_embedding = model.encode_sent({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_sentences([preprocess_line(ts, 'typed_entity_marker_punct_v2')]).items()}) 
    # r_embedding = model.encode_rule({k:torch.tensor(v).to(model.device) for (k, v) in model.tokenize_rules(['[entity=damascus]+ <appos [entity=syria]+']).items()})
    # print(r_embedding @ s_embedding.T)

    # intermediary_output = []
    # for ep, indices in zip(data[0], data[1]):
    #     for ts, ts_index in zip(ep['meta_test'], indices):
    #         rule_rels = {
    #             'gold_relation': ts['relation'],
    #             'gold_in_support': ts_index != 5,
    #             'rules': [],
    #             'rule_relations': [],
    #             'sentence': preprocess_line(ts, 'typed_entity_marker_punct_v2'),
    #         }
    #         for ss_for_rel in ep['meta_train']:
    #             for ss in ss_for_rel:
    #                 rule_rels['rules'] += [x['query'] for x in rules[line_to_hash(ss, use_all_fields=True)]]
    #                 rule_rels['rule_relations'] += [ss['relation'] for _ in rules[line_to_hash(ss, use_all_fields=True)]]
    #         intermediary_output.append(rule_rels)
    
    # df = pd.DataFrame(intermediary_output)
    # df_true = df[df['gold_in_support'] == True]
    # df_true_cad = df_true[df_true['gold_relation'] == '/location/country/administrative_divisions']
    # df_true_adc = df_true[df_true['gold_relation'] == '/location/administrative_division/country']
    
    # print([i for i in range(len(df_true_cad)) if len(set([x.replace('object ', '').replace('subject ', '') for x in df_true_cad['rules'].iloc[i]])) < 5][:10])

