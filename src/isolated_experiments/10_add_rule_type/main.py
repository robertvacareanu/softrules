import json
import tqdm

if __name__ == "__main__":
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/NYT29/surface_with_lexicalized.jsonl"]
    # rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface.jsonl"]
    # rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED/surface_with_lexicalized.jsonl"]
    rules_path           = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface_with_lexicalized.jsonl"]
    rules_path_outputs   = ["/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/enhanced_syntax_with_lexicalized.jsonl", "/storage/rvacareanu/data/softrules/rules/fsre_dataset/WIKIDATA/surface_with_lexicalized.jsonl"]

    rule_types = [1, 2, 3, 4]

    for rp_in, rp_out, rule_type in zip(rules_path, rules_path_outputs, rule_types):
        data = []
        with open(rp_in) as fin:
            for line in tqdm.tqdm(fin):
                loaded_line = json.loads(line)
                data.append({**loaded_line, 'rule_type': rule_type})

        with open(rp_out, 'w+') as fout:
            for line in tqdm.tqdm(data):
                _=fout.write(json.dumps(line))
                _=fout.write('\n')
