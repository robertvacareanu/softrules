import datasets
import json
import tqdm
import pytorch_lightning as pl

from src.baseline.entity_marker_cross_encoder import preprocess_line
from src.softrules.entity_marker_with_reg import typed_entity_marker_punct, replace_rule_entity_types


if __name__ == "__main__":
    pl.seed_everything(1)
    data = []
    idx = -1
    args = {
        'preprocessing_type': 'typed_entity_marker_punct'
    }
    with open('/storage/rvacareanu/data/softrules/rules/random/random_train_data_paraphrases.jsonl') as fin:
        for line in fin:
            idx += 1
            # if idx > 500_000:
                # continue
            loaded_line = json.loads(line)
            preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
            if len(preprocessed_line.split()) > 72:
                continue
            if len(loaded_line['query'].lower().split()) > 11:
                continue
            for i in range(5):
                data.append({
                    'id': idx,
                    # 'rule': loaded_line['query'].lower(),
                    # 'sentence': preprocess_line(loaded_line, args['preprocessing_type']),
                    'rule': replace_rule_entity_types(loaded_line['query'].lower()),
                    'sentence': typed_entity_marker_punct(loaded_line),
                })
                idx += 1
            data.append({
                'id': idx,
                'rule': loaded_line['query'].lower(),
                'sentence': preprocess_line(loaded_line, args['preprocessing_type']),
                # 'rule': replace_rule_entity_types(loaded_line['query'].lower()),
                # 'sentence': typed_entity_marker_punct(loaded_line),
            })
    #         # idx += 1
    with open('/storage/rvacareanu/data/softrules/rules/random/enhanced_syntax_all.jsonl') as fin:
        for line in fin:
            idx += 1
            # if idx < 200_000:
                # continue
            # if idx > 500_000:
                # continue
            loaded_line = json.loads(line)
            preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
            if len(preprocessed_line.split()) > 60:
                continue
            if len(loaded_line['query'].lower().split()) > 11:
                continue
            for i in range(10):
                data.append({
                    'id': idx,
                    # 'rule': loaded_line['query'].lower(),
                    # 'sentence': preprocess_line(loaded_line, args['preprocessing_type']),
                    'rule': replace_rule_entity_types(loaded_line['query'].lower()),
                    'sentence': typed_entity_marker_punct(loaded_line),
                })
                idx += 1
            # data.append({
            #     'id': idx,
            #     'rule': loaded_line['query'].lower(),
            #     'sentence': preprocess_line(loaded_line, args['preprocessing_type']),
            #     # 'rule': replace_rule_entity_types(loaded_line['query'].lower()),
            #     # 'sentence': typed_entity_marker_punct(loaded_line),
            # })
            # idx += 1
    # # # with open('/home/rvacareanu/projects_5_2210/rule_generation/random/surface_1.jsonl') as fin:
    # # #     for line in fin:
    # # #         idx += 1
    # # #         loaded_line = json.loads(line)
    # # #         preprocessed_line = preprocess_line(loaded_line, args['preprocessing_type'])
    # # #         if len(preprocessed_line.split()) > 96:
    # # #             continue
    # # #         if len(loaded_line['query'].lower().split()) > 11:
    # # #             continue
    # # #         data.append({
    # # #             'id': idx,
    # # #             'rule': loaded_line['query'].lower(),
    # # #             'sentence': preprocess_line(loaded_line, args['preprocessing_type']),
    # # #         })

    data = datasets.Dataset.from_list(data)#.select(range(250_000))
    print(data)
    print(data[0])
    data_tok = data\
        .map(lambda x: model.tokenize(x, padding=False, max_length=96), batched=True, batch_size=10_000, cache_file_name=None, load_from_cache_file=False, remove_columns=[x for x in data.column_names if x not in ['id', 'relation']], num_proc=64) \
        .filter(lambda x: sum(x['attention_mask_rule']) < 80 and sum(x['attention_mask_sentence']) < 80, num_proc=32)
    print(data_tok)