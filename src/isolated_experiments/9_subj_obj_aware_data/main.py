"""
Slightly change the rules to:
- include subject/object on the entities
    - always
    - only when both entities have the same entity types
- include the lexicalized original entities
"""

import json

from src.utils import line_to_hash

from typing import Dict, Any

def replace_entity_types_with_lexicalized_entity(rule: str, first_entity: str, second_entity: str, first_entity_lexicalized: str, second_entity_lexicalzied: str):
    fe            = f'[entity={first_entity}]+'
    se            = f'[entity={second_entity}]+'
    everything_but_entities = rule[len(fe):(len(rule)-len(se))][1:-1]

    fe            = f'[entity={first_entity_lexicalized}]+'
    se            = f'[entity={second_entity_lexicalzied}]+'

    if everything_but_entities == '':
        result = fe + ' ' + se
    else:
        result = fe + ' ' + everything_but_entities + ' ' + se
    
    return result

def include_subj_obj_in_rule(rule: str, first_entity: str, second_entity: str, subj_then_obj_order: bool):
    fe            = f'[entity={first_entity}]+'
    se            = f'[entity={second_entity}]+'
    everything_but_entities = rule[len(fe):(len(rule)-len(se))][1:-1]

    if subj_then_obj_order:
        fe            = f'[entity=SUBJECT {first_entity}]+'
        se            = f'[entity=OBJECT {second_entity}]+'
    else:
        fe            = f'[entity=OBJECT {first_entity}]+'
        se            = f'[entity=SUBJECT {second_entity}]+'

    if everything_but_entities == '':
        result = fe + ' ' + se
    else:
        result = fe + ' ' + everything_but_entities + ' ' + se
    
    return result

def include_subj_obj_in_line(line: Dict[str, Any]):
    return {
        **line,
        'line_to_hash': line_to_hash(line, use_all_fields=True), 
        'subj_type': f"SUBJECT {line['subj_type']}",
        'obj_type' : f"OBJECT {line['obj_type']}",
    }

def do_v1(episode_paths, episode_paths_output, rules_path, rules_path_outputs):
    for rp_in, rp_out in zip(rules_path, rules_path_outputs):
        with open(rp_out, 'w+') as fout:
            with open(rp_in) as fin:
                for line in fin:
                    loaded_line = json.loads(line)
                    loaded_line['query'] = include_subj_obj_in_rule(loaded_line['query'], loaded_line['first_entity_type'], loaded_line['second_entity_type'], subj_then_obj_order=loaded_line['subj_then_obj_order'])
                    _=fout.write(json.dumps(loaded_line))
                    _=fout.write("\n")


    for ep_in, ep_out in zip(episode_paths, episode_paths_output):
        with open(ep_in) as fin:
            episodes, outputs, rels = json.load(fin)

        episodes_done = [
            {
                'meta_train': [[include_subj_obj_in_line(ss) for ss in all_ss] for all_ss in e['meta_train']],
                'meta_test' : [include_subj_obj_in_line(ts) for ts in e['meta_test']],
            }
            for e in episodes
        ]

        with open(ep_out, 'w+') as fout:
            json.dump([episodes_done, outputs, rels], fout)
    

def do_v2(episode_paths, episode_paths_output, rules_path, rules_path_outputs):
    for rp_in, rp_out in zip(rules_path, rules_path_outputs):
        with open(rp_out, 'w+') as fout:
            with open(rp_in) as fin:
                for line in fin:
                    loaded_line = json.loads(line)
                    if loaded_line['first_entity_type'] == loaded_line['second_entity_type']:
                        loaded_line['query'] = include_subj_obj_in_rule(loaded_line['query'], loaded_line['first_entity_type'], loaded_line['second_entity_type'], subj_then_obj_order=loaded_line['subj_then_obj_order'])
                    _=fout.write(json.dumps(loaded_line))
                    _=fout.write("\n")


    for ep_in, ep_out in zip(episode_paths, episode_paths_output):
        with open(ep_in) as fin:
            episodes, outputs, rels = json.load(fin)

        episodes_done = [
            {
                'meta_train': [[include_subj_obj_in_line(ss) if ss['subj_type'] == ss['obj_type'] else ss for ss in all_ss] for all_ss in e['meta_train']],
                'meta_test' : [include_subj_obj_in_line(ts) if ts['subj_type'] == ts['obj_type'] else ts for ts in e['meta_test']],
            }
            for e in episodes
        ]

        with open(ep_out, 'w+') as fout:
            json.dump([episodes_done, outputs, rels], fout)
    

# python -m src.isolated_experiments.9_subj_obj_aware_data.main
if __name__ == "__main__":
    # episode_paths        = ["/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"]
    # episode_paths_output = ["/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290_with_subjobj_v1.json"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface_with_lexicalized.jsonl"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/enhanced_syntax_with_lexicalized_partial.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/surface_with_lexicalized_partial.jsonl"]

    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface_with_lexicalized.jsonl"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_words_only.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_words_only_with_lexicalized_partial.jsonl"]
    # rules_path           = [
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/TACRED/surface.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/TACRED/enhanced_syntax.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/NYT29/surface.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/NYT29/enhanced_syntax.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/WIKIDATA/surface.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/WIKIDATA/enhanced_syntax.jsonl",
    # ]
    # rules_path_outputs   = [
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/TACRED/surface_with_lexicalized_partial.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/TACRED/enhanced_syntax_with_lexicalized_partial.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/NYT29/surface_with_lexicalized_partial.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/NYT29/enhanced_syntax_with_lexicalized_partial.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/WIKIDATA/surface_with_lexicalized_partial.jsonl",
    #     "/storage/rvacareanu/data/softrules/rules/fsre_dataset_smallscale_exp/WIKIDATA/enhanced_syntax_with_lexicalized_partial.jsonl",

    # ]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface_words_only.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface_words_only_with_lexicalized_partial.jsonl"]
    rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface_words_only.jsonl"]
    rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface_words_only_with_lexicalized_partial.jsonl"]

    
    # do_v1(episode_paths, episode_paths_output, rules_path, rules_path_outputs)


    # episode_paths        = ["/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"]
    # episode_paths_output = ["/storage/rvacareanu/data/softrules/fsre_dataset/NYT29/test_episodes/5_way_1_shots_10K_episodes_3q_seed_160290_with_subjobj_v2.json"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax_with_subjobj_v2.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface_with_subjobj_v2.jsonl"]
    # do_v2(episode_paths, episode_paths_output, rules_path, rules_path_outputs)

    for rp_in, rp_out in zip(rules_path, rules_path_outputs):
        with open(rp_out, 'w+') as fout:
            with open(rp_in) as fin:
                for line in fin:
                    loaded_line = json.loads(line)
                    # _=fout.write(json.dumps(loaded_line))
                    # _=fout.write("\n")
                    r_fe = ' '.join(loaded_line['sentence_tokenized'][loaded_line['first_entity_start']:loaded_line['first_entity_end']]) # NOTE: no more `+1` here as we already add one when we created the rule file
                    r_se = ' '.join(loaded_line['sentence_tokenized'][loaded_line['second_entity_start']:loaded_line['second_entity_end']]) # NOTE: no more `+1` here as we already add one when we created the rule file
                    if loaded_line['first_entity_type'] == loaded_line['second_entity_type']:
                        loaded_line['query'] = replace_entity_types_with_lexicalized_entity(rule=loaded_line['query'], first_entity=loaded_line['first_entity_type'], second_entity=loaded_line['second_entity_type'], first_entity_lexicalized=r_fe, second_entity_lexicalzied=r_se)
                    _=fout.write(json.dumps(loaded_line))
                    _=fout.write("\n")


    pass