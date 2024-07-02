import json
from pprint import pprint
from src.baseline.entity_marker_cross_encoder import preprocess_line
import random

if __name__ == "__main__":
    data = []
    with open('/storage/rvacareanu/code/projects_7_2309/softrules/data/231202/step4/esp/data.jsonl') as fin:
        for line in fin:
            loaded = json.loads(line)
            original = {
                'token'     : loaded['sentence']['token_original'],
                'subj_start': loaded['sentence']['subj_start_original'],
                'subj_end'  : loaded['sentence']['subj_end_original'],
                'obj_start' : loaded['sentence']['obj_start_original'],
                'obj_end'   : loaded['sentence']['obj_end_original'],
                'subj_type' : loaded['sentence']['subj_type_original'],
                'obj_type'  : loaded['sentence']['obj_type_original'],}
            paraphrased = {
                'token'     : loaded['sentence']['token'],
                'subj_start': loaded['sentence']['subj_start'],
                'subj_end'  : loaded['sentence']['subj_end'],
                'obj_start' : loaded['sentence']['obj_start'],
                'obj_end'   : loaded['sentence']['obj_end'],
                'subj_type' : loaded['sentence']['subj_type'],
                'obj_type'  : loaded['sentence']['obj_type'],}
            data.append({
                'original'          : ' '.join(original['token']), 
                'paraphrased'       : ' '.join(paraphrased['token']),
                'original_marked'   : preprocess_line(original, 'typed_entity_marker_punct'), 
                'paraphrased_marked': preprocess_line(paraphrased, 'typed_entity_marker_punct')})

    r = random.Random(1)
    sample = r.choices(data, k=100)
    for x in sample:
        print(json.dumps(x))
        # print("\n")