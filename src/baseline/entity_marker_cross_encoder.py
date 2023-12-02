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

from typing import Callable, Literal

MarkerTypes = Literal["entity_mask", "entity_marker", "entity_marker_punct", "typed_entity_marker", "typed_entity_marker_punct"]

def preprocess_line(original_line, preprocessing_type: MarkerTypes) -> str:
    """
    Preprocess a line and return the processed tokens
    Can handle multiple types of masking:
        - entity_mask
        - entity_marker
        - entity_marker_punct
        - typed_entity_marker
        - typed_entity_marker_punct
    """
    line = {**original_line}
    if line['subj_type'].lower().startswith('b-') or line['subj_type'].lower().startswith('i-'):
        # print(f"Yess: {line['subj_type']}")
        line['subj_type'] = line['subj_type'][2:]
    if line['obj_type'].lower().startswith('b-') or line['obj_type'].lower().startswith('i-'):
        # print(f"Yesss: {line['obj_type']}")
        line['obj_type'] = line['obj_type'][2:]
        
    if preprocessing_type == 'entity_mask': # `John Doe was born in New York City` -> `[SUBJ-PER] was born in [OBJ-LOC]`
        return entity_mask(line)
    elif preprocessing_type == 'entity_marker': # `John Doe was born in New York City` -> `[SUBJ] John Doe [/SUBJ] was born in [OBJ] New York City [/OBJ]`
        subj_start_marker = lambda x: '[SUBJ]'
        subj_end_marker   = lambda x: '[/SUBJ]'
        obj_start_marker  = lambda x: '[OBJ]'
        obj_end_marker    = lambda x: '[/OBJ]'
        return entity_marker(line, subj_start_marker, subj_end_marker, obj_start_marker, obj_end_marker)
    elif preprocessing_type == "entity_marker_punct": # `John Doe was born in New York City` -> ``@ John Doe @ was born in # New York City#`
        subj_start_marker = lambda x: '@'
        subj_end_marker   = lambda x: '@'
        obj_start_marker  = lambda x: '#'
        obj_end_marker    = lambda x: '#'
        return entity_marker(line, subj_start_marker, subj_end_marker, obj_start_marker, obj_end_marker)
    elif preprocessing_type == "typed_entity_marker": # `John Doe was born in New York City` -> `[SUBJ-PESON] John Doe [/SUBJ-PERSON] was born in [OBJ-LOC] New York City [/OBJ-LOC]`
        subj_start_marker = lambda x: f'[SUBJ-{x.upper()}]'
        subj_end_marker   = lambda x: f'[/SUBJ-{x.upper()}]'
        obj_start_marker  = lambda x: f'[OBJ-{x.upper()}]'
        obj_end_marker    = lambda x: f'[/OBJ-{x.upper()}]'
        return entity_marker(line, subj_start_marker, subj_end_marker, obj_start_marker, obj_end_marker)
    elif preprocessing_type == "typed_entity_marker_punct": # `John Doe was born in New York City` -> `@ * person * John Doe @ was born in # ^ loc ^ New York City #`
        subj_start_marker = lambda x: f'@ * {x.lower()} *'
        subj_end_marker   = lambda x: f'@'
        obj_start_marker  = lambda x: f'# ^ {x.lower()} ^'
        obj_end_marker    = lambda x: f'#'
        return entity_marker(line, subj_start_marker, subj_end_marker, obj_start_marker, obj_end_marker)
    else: # Error
        raise ValueError(f"The processing type `{preprocessing_type}` is not recognized. Is everything ok?")



def entity_mask(line, subj_format: Callable[[str], str] = lambda subj_type: f"[SUBJ-{subj_type.upper()}]", obj_format: Callable[[str], str] = lambda obj_type: f"[OBJ-{obj_type.upper()}]"):
    """
    John Doe was born in New York City => [SUBJ-PERSON] was born in [OBJ-CITY]
    """
    line_tokens = [[x] for x in line['token']]
    # Delete SUBJ
    line_tokens = [x if line['subj_start'] > i or line['subj_end'] < i else [] for i, x in enumerate(line_tokens)]
    # Delete OBJ
    line_tokens = [x if line['obj_start'] > i or line['obj_end'] < i else [] for i, x in enumerate(line_tokens)]

    # Add [SUBJ-TYPE] on first position of subject
    line_tokens[line['subj_start']] = [subj_format(line['subj_type'])]
    # Add [OBJ-TYPE] on first position of object
    line_tokens[line['obj_start']] = [obj_format(line['obj_type'])]

    line_tokens = ' '.join([' '.join(x) for x in line_tokens])

    return line_tokens


def entity_marker(
    line, 
    subj_start_marker: Callable[[str], str] = lambda subj_type: f"[SUBJ-{subj_type.upper()}]",
    subj_end_marker  : Callable[[str], str] = lambda subj_type: f"[/SUBJ-{subj_type.upper()}]",
    obj_start_marker : Callable[[str], str] = lambda obj_type : f"[OBJ-{obj_type.upper()}]",
    obj_end_marker   : Callable[[str], str] = lambda obj_type : f"[/OBJ-{obj_type.upper()}]",
):
    """
    John Doe was born in New York City => [SUBJ] John Doe [/SUBJ] was born in [OBJ] New York City [/OBJ]
    """
    line_tokens = [[x] for x in line['token']]

    line_tokens[line['subj_start']] = [subj_start_marker(line['subj_type'])] + line_tokens[line['subj_start']]
    line_tokens[line['subj_end']]   = line_tokens[line['subj_end']] + [subj_end_marker(line['subj_type'])]
    line_tokens[line['obj_start']] = [obj_start_marker(line['obj_type'])] + line_tokens[line['obj_start']]
    line_tokens[line['obj_end']] = line_tokens[line['obj_end']] + [obj_end_marker(line['obj_type'])]
    
    line_tokens = ' '.join([' '.join(x) for x in line_tokens])
    
    return line_tokens

# def entity_marker_punct(
#     line,
#     subj_punct: str = "@",
#     obj_punct: str  = "#",
# ):
#     """
#     """
#     return entity_marker(
#         line=line,
#         subj_start_marker = lambda subj_type: subj_punct,
#         subj_end_marker   = lambda subj_type: subj_punct,
#         obj_start_marker  = lambda obj_type : obj_punct,
#         obj_end_marker    = lambda obj_type : obj_punct,
#     )

# def typed_entity_marker(
#     line,
#     subj_punct: str = "@",
#     obj_punct: str  = "#",
# ):
#     """
#     """
#     return entity_marker(
#         line=line,
#         subj_start_marker = lambda subj_type: subj_punct,
#         subj_end_marker   = lambda subj_type: subj_punct,
#         obj_start_marker  = lambda obj_type : obj_punct,
#         obj_end_marker    = lambda obj_type : obj_punct,
#     )


if __name__ == "__main__":
    """
    An ad-hoc simple test
    """
    import json
    line_as_str = '{"id": "e7798e546c56c8b814a9", "docid": "NYT_ENG_20101210.0152", "relation": "per:alternate_names", "token": ["In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."], "subj_start": 27, "subj_end": 27, "obj_start": 14, "obj_end": 15, "subj_type": "PERSON", "obj_type": "PERSON", "stanford_pos": ["IN", "JJ", "NN", "CC", "IN", "NNP", "NNP", "NNP", ",", "WRB", ",", "RB", "VBN", "IN", "NNP", "NNP", "-LRB-", "DT", "NN", "VBN", "IN", "PRP", "IN", "PRP$", "NN", "-RRB-", ",", "NNP", "VBD", "DT", "JJ", "."], "stanford_ner": ["O", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "O", "O", "MISC", "O"], "stanford_head": [3, 3, 0, 3, 8, 8, 8, 29, 8, 13, 13, 13, 8, 16, 16, 13, 19, 19, 16, 19, 22, 20, 25, 25, 20, 19, 8, 29, 3, 31, 29, 3], "stanford_deprel": ["case", "amod", "ROOT", "cc", "case", "compound", "compound", "nmod", "punct", "advmod", "punct", "advmod", "acl:relcl", "case", "compound", "nmod", "punct", "det", "dep", "acl", "case", "nmod", "case", "nmod:poss", "nmod", "punct", "punct", "nsubj", "conj", "det", "xcomp", "punct"], "tokens": ["In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."], "h": ["meredith", null, [[27]]], "t": ["dandy don", null, [[14, 15]]]}'
    line = json.loads(line_as_str)

    print("\n\n")
    print(preprocess_line(line, 'entity_mask'))
    print(preprocess_line(line, 'entity_marker'))
    print(preprocess_line(line, 'entity_marker_punct'))
    print(preprocess_line(line, 'typed_entity_marker'))
    print(preprocess_line(line, 'typed_entity_marker_punct'))