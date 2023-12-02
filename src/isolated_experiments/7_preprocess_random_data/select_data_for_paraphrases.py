"""
Select data that will be used to generate paraphrases
Select sentences for both enhanced syntax and surface
"""

import datasets
from collections import defaultdict
import tqdm
import random
import json
import numpy as np

if __name__ == "__main__":

    data1 = datasets.load_dataset('json', data_files=['/storage/rvacareanu/data/softrules/rules/random_231201/enhanced_syntax_all.jsonl'])['train']
    data2 = datasets.load_dataset('json', data_files=['/storage/rvacareanu/data/softrules/rules/random_231201/surface_all.jsonl'])['train']

    data_merged = defaultdict(lambda: ([], []))

    for i, x in tqdm.tqdm(enumerate(data1)):
        key = (' '.join(x['token']), x['subj_start'], x['subj_end'], x['obj_start'], x['obj_end'], x['subj_type'], x['obj_type'])
        data_merged[key][0].append(i)

    for i, x in tqdm.tqdm(enumerate(data2)):
        key = (' '.join(x['token']), x['subj_start'], x['subj_end'], x['obj_start'], x['obj_end'], x['subj_type'], x['obj_type'])
        data_merged[key][1].append(i)

    data_filter = [x for x in data_merged.items() if len(x[1][0]) > 0 and len(x[1][1]) > 0]
    random.shuffle(data_filter)

    data_selected = data_filter

    data_selected = [{**{k:v for (k, v) in data1[indices_d1[0]].items() if k != 'query'}, 'query_es': data1[indices_d1[0]]['query'], 'query_s': data2[indices_d2[0]]['query']} for (key, (indices_d1, indices_d2)) in tqdm.tqdm(data_selected)]

    entity_types_to_data = defaultdict(list)
    for x in tqdm.tqdm(data_selected):
        key = (x['subj_type'], x['obj_type'])
        entity_types_to_data[key].append(x)
        
    # Uniform selection: selects from each entity pair type a number of `min_value` samples
    min_value = min([len(x) for x in entity_types_to_data.values()])
    min_value = min(10_000, min_value)
    print([(k, len(v)) for k, v in entity_types_to_data.items()])
    result_uniform = []
    for key in tqdm.tqdm(entity_types_to_data.keys()):
        result_uniform += random.sample(entity_types_to_data[key], k=min_value)

    random.shuffle(result_uniform)

    # Scaled selection: the data is skewed, so selecting uniformly at random
    # without sub-sampling first might lead to a very small number of paraphrases
    # for certain entity types (e.g. min is `10910`, max is `1190126`, total is `21453697` 
    # so we get 0.05% of the minimum and 5% of the maximum)
    # To account for this, we do an np.sqrt scaling. But purely doing np.sqrt scaling
    # would lead to: (1) sub-sampling the least popular entity type as well, and (2) drastically 
    # decreasing the total number of datapoints (e.g. total is `21453697`, total after np.sqrt is 
    # `27402`). Because of (2) and because we want more than a couple of 10k, we do np.sqrt after 
    # subtracting the `min_value`. But this would lead to an almost uniform distribution. Because of this, 
    # we also multiply with `min_value`, which exacerbates differences from `min_value`
    # With this, `result_subsampled` has a total size of `3248786` and the following entity types distribution:
    # ```
    #   [(('ORGANIZATION', 'EMAIL'), 10910), (('PERSON', 'EMAIL'), 23058), (('ORGANIZATION', 'URL'), 38385), 
    #   (('PERSON', 'MONEY'), 45551), (('ORGANIZATION', 'IDEOLOGY'), 48339), (('PERSON', 'RELIGION'), 49127), 
    #   (('PERSON', 'CRIMINAL_CHARGE'), 49302), (('PERSON', 'IDEOLOGY'), 55430), (('CAUSE_OF_DEATH', 'PERSON'), 58257), 
    #   (('STATE_OR_PROVINCE', 'PERSON'), 62149), (('NATIONALITY', 'PERSON'), 67216), (('COUNTRY', 'PERSON'), 68549), 
    #   (('PERSON', 'CAUSE_OF_DEATH'), 72056), (('CITY', 'PERSON'), 73574), (('CITY', 'STATE_OR_PROVINCE'), 74769), 
    #   (('PERSON', 'NATIONALITY'), 74995), (('CITY', 'ORGANIZATION'), 75592), (('ORGANIZATION', 'ORGANIZATION'), 76507), 
    #   (('LOCATION', 'ORGANIZATION'), 77099), (('PERSON', 'STATE_OR_PROVINCE'), 77530), (('PERSON', 'PERSON'), 79529), 
    #   (('COUNTRY', 'ORGANIZATION'), 79650), (('ORGANIZATION', 'STATE_OR_PROVINCE'), 82482), (('PERSON', 'COUNTRY'), 86781), 
    #   (('ORGANIZATION', 'LOCATION'), 91623), (('PERSON', 'CITY'), 91901), (('PERSON', 'LOCATION'), 92490), (('ORGANIZATION', 'COUNTRY'), 94902), 
    #   (('ORGANIZATION', 'CITY'), 95346), (('NUMBER', 'PERSON'), 98475), (('ORGANIZATION', 'PERSON'), 106166), (('DATE', 'PERSON'), 108949), 
    #   (('PERSON', 'NUMBER'), 113188), (('TITLE', 'PERSON'), 116046), (('PERSON', 'DATE'), 118961), (('PERSON', 'TITLE'), 119377), 
    #   (('PERSON', 'ORGANIZATION'), 122998), (('DATE', 'ORGANIZATION'), 123044), (('ORGANIZATION', 'NUMBER'), 124148), (('ORGANIZATION', 'DATE'), 124335)]
    # ```
    # So ratio between max and min is about 10x (from 100x before sub-sampling)

    random.seed(1)
    min_value = min([len(x) for x in entity_types_to_data.values()])
    result_subsampled = []
    for key in tqdm.tqdm(entity_types_to_data.keys()):
        if len(entity_types_to_data[key]) <= min_value:
            result_subsampled += entity_types_to_data[key]
        else:
            result_subsampled += random.sample(entity_types_to_data[key], k=min(int(np.sqrt((len(entity_types_to_data[key]) - min_value) * min_value) + min_value), len(entity_types_to_data[key])))

    random.shuffle(result_subsampled)


    with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/data_es_s_for_paraphrases_uniformdistr_entitytypes.jsonl', 'w+') as fout:
        for line in tqdm.tqdm(result_uniform):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


    # [(('ORGANIZATION', 'EMAIL'), 3348), (('PERSON', 'EMAIL'), 7166), (('ORGANIZATION', 'URL'), 11958), (('PERSON', 'MONEY'), 14201), 
    # (('ORGANIZATION', 'IDEOLOGY'), 14893), (('PERSON', 'RELIGION'), 15121), (('PERSON', 'CRIMINAL_CHARGE'), 15139), (('PERSON', 'IDEOLOGY'), 16990), 
    # (('CAUSE_OF_DEATH', 'PERSON'), 18096), (('STATE_OR_PROVINCE', 'PERSON'), 19178), (('NATIONALITY', 'PERSON'), 20534), (('COUNTRY', 'PERSON'), 20970), 
    # (('PERSON', 'CAUSE_OF_DEATH'), 22398), (('CITY', 'PERSON'), 22643), (('PERSON', 'NATIONALITY'), 22892), (('CITY', 'STATE_OR_PROVINCE'), 22986), 
    # (('CITY', 'ORGANIZATION'), 23299), (('ORGANIZATION', 'ORGANIZATION'), 23481), (('LOCATION', 'ORGANIZATION'), 23647), (('PERSON', 'STATE_OR_PROVINCE'), 23906), 
    # (('COUNTRY', 'ORGANIZATION'), 24400), (('PERSON', 'PERSON'), 24456), (('ORGANIZATION', 'STATE_OR_PROVINCE'), 25489), (('PERSON', 'COUNTRY'), 26768), 
    # (('ORGANIZATION', 'LOCATION'), 27996), (('PERSON', 'CITY'), 28585), (('PERSON', 'LOCATION'), 28672), (('ORGANIZATION', 'COUNTRY'), 29104), 
    # (('ORGANIZATION', 'CITY'), 29224), (('NUMBER', 'PERSON'), 30151), (('ORGANIZATION', 'PERSON'), 32411), (('DATE', 'PERSON'), 33648), 
    # (('PERSON', 'NUMBER'), 35056), (('TITLE', 'PERSON'), 35883), (('PERSON', 'DATE'), 36385), (('PERSON', 'TITLE'), 36718), (('PERSON', 'ORGANIZATION'), 37819), 
    # (('ORGANIZATION', 'NUMBER'), 37938), (('DATE', 'ORGANIZATION'), 38199), (('ORGANIZATION', 'DATE'), 38252)]
    with open('/storage/rvacareanu/data/softrules/rules/random_231201/processed/data_es_s_for_paraphrases_subsampled_entitytypes.jsonl', 'w+') as fout:
        for line in tqdm.tqdm(random.sample(result_subsampled, k=1_000_000)):
            _=fout.write(json.dumps(line))
            _=fout.write('\n')


