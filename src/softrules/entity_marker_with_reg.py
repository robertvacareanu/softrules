"""
Baseline inspired by:
    - https://arxiv.org/abs/2102.01373 for entity type masking
    - https://aclanthology.org/D19-1649.pdf for BERT-PAIR


We will exemplify the entity type masking functions available following 
Table 1 from https://arxiv.org/pdf/2102.01373.pdf. Similarly, we will use 
"Bill was born in Seattle"
- Entity-Mask (see `entity_mask`)
    [SUBJ-PERSON] was born in [OBJ-CITY].
- Entity marker 
    [E1] Bill [/E1] was born in [E2] Seattle [/E2].
- Entity marker (punct) 
    @ Bill @ was born in # Seattle #.
- Typed entity marker 
    〈S:PERSON〉 Bill 〈/S:PERSON〉 was born in〈O:CITY〉 Seattle 〈/O:CITY〉
- Typed entity marker (punct)
    @ * person * Bill @ was born in # ∧ city ∧ Seattle #.
    
"""

import random
import re

from typing import Callable, Literal, Dict

["ORGANIZATION", "DATE", "PERSON", "NUMBER", "TITLE", "DURATION", "MISC", "COUNTRY", "LOCATION", "CAUSE_OF_DEATH", "CITY", "NATIONALITY", "ORDINAL", "STATE_OR_PROVINCE", "PERCENT", "MONEY", "SET", "IDEOLOGY", "CRIMINAL_CHARGE", "TIME", "RELIGION", "URL", "EMAIL", "HANDLE"]


entity_synonyms = [
    ['organization',      'org', 'company', 'firm', 'corporation', 'enterprise'], 
    ['date',              'a specific date', ], 
    ['person',            'per', 'human', 'human being', 'individual'], 
    ['number',            'digits',], 
    ['title',             'designation', 'formal designation'], 
    ['duration',          'time period',], 
    ['misc',              'miscellaneous',], 
    ['country',           'nation', 'state', 'territory'], 
    ['location',          'place', 'area', 'geographic area', 'loc'], 
    ['cause_of_death',    'date of demise', 'cause of death', 'death cause', 'mortal cause'], 
    ['city',              'municipality', 'town', 'populated urban area'], 
    ['nationality',       'citizenship',], 
    ['ordinal',           'ranking',], 
    ['state_or_province', 'region', 'territorial division within a country'], 
    ['percent',           'percentage',], 
    ['money',             'currency',],# 'medium of exchange in the form of coins and banknotes', ], 
    ['set',               'collection', 'group of items', ], 
    ['ideology',          'doctrine', 'system of ideas and ideals'], 
    ['criminal_charge',   'accusation', 'formal allegation'], 
    ['time',              'period', 'time period'], 
    ['religion',          'belief', 'faith', 'spiritual belief', 'worshipper'], 
    ['url',               'web address',], 
    ['email',             'electronic mail',], 
    ['handle',            'username', 'personal identifier'], 
    
]

entity_synonyms_dict = {y:x for x in entity_synonyms for y in x}

def replace_rule_entity_types(
        rule: str, 
        entity_synonyms: Dict[str, str] = entity_synonyms_dict,
        eval_mode = False,
    ):
    """
    :param rule -> The rule to augment
    :param entity_synonyms -> The entities synonyms
    :param eval_mode -> Whether we are in eval mode or not (when in eval mode, no changes)
    """
    if eval_mode:
        return rule
    
    pattern = r'\[entity=(.*?)\]'

    def replace_entity(match):
        captured_text = match.group(1)
        if captured_text.lower().startswith('i-') or captured_text.lower().startswith('b-'):
            print(f"Yes, {captured_text}")
            captured_text = captured_text[2:]

        replacement_text = random.choice(entity_synonyms.get(captured_text, [captured_text]))
        return f'[entity={replacement_text}]'

    return re.sub(pattern, replace_entity, rule)


def typed_entity_marker_punct(
        line, 
        dropout_entity_type_prob: float           = 0.10, 
        switch_to_synonym_entity_type_prob: float = 1.00, 
        entity_synonyms: Dict[str, str] = entity_synonyms_dict,
        eval_mode = False,
    ):
    """
    :param line
    :param dropout_entity_type -> Probability to drop the entity type
    :param entity_synonyms     -> Use some of the synonyms for the entity types (e.g., line says 'PER', we can use 'person' as well)
    """
    if eval_mode:
        if dropout_entity_type_prob > 0.0 or switch_to_synonym_entity_type_prob > 0.0:
            raise ValueError("Non-zero probabilities passed during eval")

    line_tokens = [[x] for x in line['token']]

    if random.random() < switch_to_synonym_entity_type_prob:
        new_subj_type = random.choice(entity_synonyms.get(line['subj_type'].lower(), [line['subj_type'].lower()]))
    else:
        new_subj_type = line['subj_type'].lower()
    if random.random() < switch_to_synonym_entity_type_prob:
        new_obj_type  = random.choice(entity_synonyms.get(line['obj_type'].lower(),  [line['obj_type'].lower()]))
    else:
        new_obj_type = line['obj_type'].lower()

    if random.random() < dropout_entity_type_prob:
        subj_start_marker = f'@ * subject *'
        subj_end_marker   = '@'
    else:
        subj_start_marker = f'@ * {new_subj_type} *'
        subj_end_marker   = '@'

    if random.random() < dropout_entity_type_prob:
        obj_start_marker = f'# ^ object ^'
        obj_end_marker   = '#'
    else:
        obj_start_marker = f'# ^ {new_obj_type} ^'
        obj_end_marker   = '#'


    line_tokens[line['subj_start']] = [subj_start_marker] + line_tokens[line['subj_start']]
    line_tokens[line['subj_end']]   = line_tokens[line['subj_end']] + [subj_end_marker]
    line_tokens[line['obj_start']] = [obj_start_marker] + line_tokens[line['obj_start']]
    line_tokens[line['obj_end']] = line_tokens[line['obj_end']] + [obj_end_marker]
    
    line_tokens = ' '.join([' '.join(x) for x in line_tokens])    

    return line_tokens
    
def typed_entity_marker_punct_v2(
        line, 
        dropout_entity_type_prob: float           = 0.10, 
        switch_to_synonym_entity_type_prob: float = 1.00, 
        entity_synonyms: Dict[str, str] = entity_synonyms_dict,
        eval_mode = False,
    ):
    """
    :param line
    :param dropout_entity_type -> Probability to drop the entity type
    :param entity_synonyms     -> Use some of the synonyms for the entity types (e.g., line says 'PER', we can use 'person' as well)
    """
    if eval_mode:
        if dropout_entity_type_prob > 0.0 or switch_to_synonym_entity_type_prob > 0.0:
            raise ValueError("Non-zero probabilities passed during eval")

    line_tokens = [[x] for x in line['token']]

    if random.random() < switch_to_synonym_entity_type_prob:
        new_subj_type = random.choice(entity_synonyms.get(line['subj_type'].lower(), [line['subj_type'].lower()]))
    else:
        new_subj_type = line['subj_type'].lower()
    if random.random() < switch_to_synonym_entity_type_prob:
        new_obj_type  = random.choice(entity_synonyms.get(line['obj_type'].lower(),  [line['obj_type'].lower()]))
    else:
        new_obj_type = line['obj_type'].lower()

    if random.random() < dropout_entity_type_prob:
        subj_start_marker = f'@ * entity *'
        subj_end_marker   = '@'
    else:
        subj_start_marker = f'@ * {new_subj_type} *'
        subj_end_marker   = '@'

    if random.random() < dropout_entity_type_prob:
        obj_start_marker = f'@ * entity *'
        obj_end_marker   = '@'
    else:
        obj_start_marker = f'@ * {new_obj_type} *'
        obj_end_marker   = '@'


    line_tokens[line['subj_start']] = [subj_start_marker] + line_tokens[line['subj_start']]
    line_tokens[line['subj_end']]   = line_tokens[line['subj_end']] + [subj_end_marker]
    line_tokens[line['obj_start']] = [obj_start_marker] + line_tokens[line['obj_start']]
    line_tokens[line['obj_end']] = line_tokens[line['obj_end']] + [obj_end_marker]
    
    line_tokens = ' '.join([' '.join(x) for x in line_tokens])    

    return line_tokens
    

if __name__ == "__main__":
    """
    An ad-hoc simple test
    """
    import json
    line_as_str = '{"id": "e7798e546c56c8b814a9", "docid": "NYT_ENG_20101210.0152", "relation": "per:alternate_names", "token": ["In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."], "subj_start": 27, "subj_end": 27, "obj_start": 14, "obj_end": 15, "subj_type": "PERSON", "obj_type": "PERSON", "stanford_pos": ["IN", "JJ", "NN", "CC", "IN", "NNP", "NNP", "NNP", ",", "WRB", ",", "RB", "VBN", "IN", "NNP", "NNP", "-LRB-", "DT", "NN", "VBN", "IN", "PRP", "IN", "PRP$", "NN", "-RRB-", ",", "NNP", "VBD", "DT", "JJ", "."], "stanford_ner": ["O", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "O", "O", "MISC", "O"], "stanford_head": [3, 3, 0, 3, 8, 8, 8, 29, 8, 13, 13, 13, 8, 16, 16, 13, 19, 19, 16, 19, 22, 20, 25, 25, 20, 19, 8, 29, 3, 31, 29, 3], "stanford_deprel": ["case", "amod", "ROOT", "cc", "case", "compound", "compound", "nmod", "punct", "advmod", "punct", "advmod", "acl:relcl", "case", "compound", "nmod", "punct", "det", "dep", "acl", "case", "nmod", "case", "nmod:poss", "nmod", "punct", "punct", "nsubj", "conj", "det", "xcomp", "punct"], "tokens": ["In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."], "h": ["meredith", null, [[27]]], "t": ["dandy don", null, [[14, 15]]]}'
    line = json.loads(line_as_str)

    print("\n\n")
    print(typed_entity_marker_punct(line, 0.0, 0.0))
    print("\n\n")
    print(typed_entity_marker_punct(line))
    print(typed_entity_marker_punct(line))
    print(typed_entity_marker_punct(line))
    print(typed_entity_marker_punct(line))
    print(typed_entity_marker_punct(line))
    print("\n\n")
    print("\n\n")
    print("\n\n")
    print(typed_entity_marker_punct_v2(line, 0.0, 0.0))
    print("\n\n")
    print(typed_entity_marker_punct_v2(line))
    print(typed_entity_marker_punct_v2(line))
    print(typed_entity_marker_punct_v2(line))
    print(typed_entity_marker_punct_v2(line))
    print(typed_entity_marker_punct_v2(line))