import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
from src.utils import line_to_hash
import argparse
import json
import tqdm
import torch
from src.baseline.entity_marker_cross_encoder import preprocess_line

from collections import defaultdict, Counter

from src.softrules.different_encoder.model_cliplike_multidata import SoftRulesEncoder, read_rules, get_valdata
from src.utils import tacred_score, compute_results_with_thresholds

from torch.utils.data import DataLoader
import datasets

import scipy as sp
import scipy.special as spp

def read_data(args):
    rules = read_rules(args['rules_path'])
    rules = dict(rules)

    with open(args['train_path']) as fin:
        data = json.load(fin)
        train_rules = [
            {
                'id': i,
                'rule': rules[line_to_hash(x, use_all_fields=True)],
                'relation': x['relation']
            }    
        for i, x in enumerate(data)]

    with open(args['test_path']) as fin:
        data = json.load(fin)
        test_sentences = [
            {
                'id': i,
                'sentence': preprocess_line(line, 'typed_entity_marker_punct_v2'),
                'relation': line['relation']
            }
            for i, line in enumerate(data)
        ]

    return (train_rules, test_sentences)

def custom_rules():
    rules = {
        # 'org:alternate_names'                : '[entity=organization]+ <nsubj known >nmod_as [entity=organization]+',
        # 'org:city_of_headquarters'           : '[entity=organization]+ <nsubj located >nmod_in [entity=city]+',
        # 'org:country_of_headquarters'        : '[entity=organization]+ <nsubj located >nmod_in [entity=city]+',
        # 'org:dissolved'                      : '[entity=organization]+ <nsubj dissolved >nmod_on [entity=date]+',
        # 'org:founded'                        : '[entity=organization]+ <nsubj founded >nmod_on [entity=date]+',
        'org:founded_by'                     : '[entity=organization]+ <nsubj founded >nmod_by [entity=date]+',
        # 'org:member_of'                      : '[entity=organization]+ <nsubj part >nmod_of [entity=organization]+',
        # 'org:members'                        : '[entity=organization]+ <nsubj member >nmod_of [entity=organization, country]+',
        # 'org:number_of_employees/members'    : '[entity=organization]+ <nsubj employes >nmod_of [entity=number]+',
        # 'org:parents'                        : '[entity=organization]+ <nsubj parent >nmod_of [entity=organization]+',
        # 'org:political/religious_affiliation': '[entity=organization]+ <nsubj has >nmod [entity=ideology]+',
        # 'org:shareholders'                   : '[entity=person]+ <nsubj shareholder >nmod_of [entity=organization]+',
        # 'org:stateorprovince_of_headquarters': '',
        # 'org:subsidiaries'                   : '[entity=organization]+ <nsubj subsidiary >nmod_of [entity=organization]+',
        # 'org:top_members/employees'          : '',
        'org:website'                        : '[entity=organization]+ <appos [entity=url]+',
        'per:age'                            : '[entity=person]+ <nsubj is >dobj [entity=age]+',
        'per:alternate_names'                : '[entity=person]+ <nsubj known >nmod_as [entity=name]+',
        'per:cause_of_death'                 : '[entity=person]+ <nsubj died >nmod_of [entity=cause of death]+',
        'per:charges'                        : '[entity=person]+ <nsubj charged >nmod_of [entity=legal charge]+',
        'per:children'                       : '[entity=person]+ <nsubj child >nmod_of [entity=parent]+',
        'per:cities_of_residence'            : '[entity=person]+ <nsubj lives >nmod_in [entity=city]+',
        'per:city_of_birth'                  : '[entity=person]+ <nsubj born >nmod_in [entity=city]+',
        'per:city_of_death'                  : '[entity=person]+ <nsubj died >nmod_in [entity=city]+',
        'per:countries_of_residence'         : '[entity=person]+ <nsubj lives >nmod_in [entity=country]+',
        'per:country_of_birth'               : '[entity=person]+ <nsubj born >nmod_in [entity=country]+',
        'per:country_of_death'               : '[entity=person]+ <nsubj died >nmod_in [entity=country]+',
        'per:date_of_birth'                  : '[entity=person]+ <nsubj born >nmod_on [entity=date]+',
        'per:date_of_death'                  : '[entity=person]+ <nsubj died >nmod_on [entity=date]+',
        'per:employee_of'                    : '[entity=person]+ <nsubj works >nmod_for [entity=employer]+',
        # 'per:origin'                         : '',
        # 'per:other_family'                   : '',
        'per:parents'                        : '[entity=parent]+ >nmod_of [entity=child]+',
        'per:religion'                       : '[entity=person]+ <nsubj has >nmod [entity=religion]+',
        'per:schools_attended'               : '[entity=person]+ >nmod_attended [entity=school]+',
        'per:siblings'                       : '[entity=person]+ >sibling_of [entity=person]+',
        'per:spouse'                         : '[entity=person]+ <nsubj married >nmod_to [entity=person]+',
        'per:stateorprovince_of_birth'       : '[entity=person]+ <nsubj born >nmod_in [entity=state or province]+',
        'per:stateorprovince_of_death'       : '[entity=person]+ <nsubj died >nmod_in [entity=state or province]+',
        'per:stateorprovinces_of_residence'  : '[entity=person]+ <nsubj lives >nmod_in [entity=state or province]+',
        'per:title'                          : '[entity=title]+ <appos [entity=person]+',
    }

    return rules

def custom_rules2():
    rules = {
        'org:founded_by'                     : [
            '[entity=person]+ <nsubj founded >dobj [entity=organization]+', 
            "[entity=founder]+ 's [entity=organization]+", 
            "[entity=person]+ >appos founder >nmod_of [entity=organization]+"

        ],
        'per:employee_of'                    : [
            '[entity=person]+ is a member of [entity=organization]+', 
            "[entity=organization]+ 's number two [entity=person]+"
        ],
        'org:alternate_names'                : [
            "[entity=organization]+ >dep branch >nmod_of [entity=organization]+",
            "[entity=organization]+ -lrb- [entity=organization]+",
        ],
        "per:cities_of_residence"            : [
            "[entity=person]+ <nsubj native >nmod_of [entity=city]+",
            "[entity=city]+ <dep [entity=person]+",
            '[entity=person]+ <nsubj lives >nmod_in [entity=city]+',
            '[entity=person]+ who resides in [entity=city]+',
            '[entity=person]+ lives in [entity=city]+',
        ],
        "per:children"                       : [
            "[entity=person]+ daughter of [entity=person]+",
            "[entity=person]+ mother of [entity=person]+",
            "[entity=person]+ son of [entity=person]+",
            "[entity=person]+ father of [entity=person]+",
        ],
        "per:title"                          : [
            "[entity=title]+ [entity=person]+",
            "[entity=title]+ <compound [entity=person]+",
            "[entity=title]+ <appos [entity=person]+",
        ],
        "per:siblings"                       : [
            '[entity=person]+ is a sibbling of [entity=person]+',
        ],
        "per:religion"                       : [
            '[entity=person]+ <nsubj has >nmod [entity=religion]+',
            '[entity=person]+ is [entity=religion]+',
        ],
        "per:age"                            : [
            "[entity=person]+ , [entity=number]+",
            "[entity=person]+ <nsubj dies >nmod_at [entity=number]+",
            "[entity=person]+ is [entity=number]+",
        ],
        "org:website"                        : [
            "[entity=organization]+ : [entity=url]+",
            "[entity=organization]+ <compound [entity=url]+",
            "[entity=organization]+ [entity=url]+",
        ],
        "per:stateorprovinces_of_residence"  : [
            "[entity=person]+ <nsubj native >nmod_of [entity=state_or_province]+",
            "[entity=person]+ lives in [entity=state_or_province]+",
            "[entity=person]+ who resides in [entity=state_or_province]+",
        ],
        "org:member_of"                      : [
            "[entity=organization]+ joined [entity=organization]+",
            "[entity=organization]+ <nsubj is >nmod_of [entity=organization]+",
            "[entity=organization]+ is part of [entity=organization]+",
            "[entity=organization]+ 's , [entity=organization]+",
        ],
        "org:top_members/employees"          : [
            "[entity=person]+ >appos director >nmod_of [entity=organization]+",
            "[entity=person]+ >appos CEO >nmod_of [entity=organization]+",
            "[entity=person]+ >appos official >nmod_of [entity=organization]+",
            "[entity=person]+ is a top employee of [entity=organization]+",
        ],
        "per:countries_of_residence"         : [
            "[entity=person]+ , [entity=country]+",
            "[entity=person]+ <nsubj native >nmod_of [entity=country]+",
            "[entity=person]+ lives in [entity=country]+",
            "[entity=person]+ 's home is in [entity=country]+",
            "[entity=person]+ who's residence is in [entity=country]+",
        ],
        "org:city_of_headquarters"           : [
            "[entity=organization]+ just outside [entity=city]+",
            "[entity=organization]+ >appos firm >nmod_in [entity=city]+",
            "[entity=location]+ [entity=organization]+",
            "[entity=organization]+ in [entity=city]+",
            "[entity=organization]+ >nmod_in [entity=city]+",
        ],
        "org:members"                        : [
            "[entity=organization]+ includes [entity=country]+",
            "[entity=organization]+ >nmod_of [entity=organization]+",
        ],
        "org:country_of_headquarters"        : [
            "[entity=organization]+ >appos firm >nmod_in [entity=country]+",
        ],
        "per:spouse"                         : [
            "[entity=person]+ and hubby [entity=person]+",
            "[entity=person]+ marriage to [entity=person]+",
            "[entity=person]+ wife , [entity=person]+",
            "[entity=person]+ husband , [entity=person]+",
            "[entity=person]+ and his husband [entity=person]+",
            "[entity=person]+ and his wife [entity=person]+",
        ],
        "org:stateorprovince_of_headquarters": [
            "[entity=organization]+ in [entity=state_or_province]+",
            "[entity=organization]+ >nmod_in [entity=state_or_province]+",
            "[entity=state_or_province]+ office of [entity=organization]+",
            "[entity=organization]+ , with [entity=number]+"
        ],
        "org:number_of_employees/members"    : [
            "[entity=organization]+ , with [entity=number]+",
            "[entity=organization]+ has about [entity=number]+",
        ],
        "org:parents"                        : [
            "[entity=organization]+ , a part of [entity=organization]+",
        ],
        "org:subsidiaries"                   : [
            "[entity=organization]+ >appos arm >nmod:poss [entity=organization]+",
            "[entity=organization]+ 's [entity=organization]+",
            "[entity=organization]+ owned by [entity=organization]+",
        ],
        "per:origin"                         : [
            "[entity=nationality]+ <nsubj [entity=person]+",
            "[entity=country]+ of [entity=person]+",
            "[entity=person]+ is [entity=nationality]+"

        ],
        "org:political/religious_affiliation": [
            "[entity=religion]+ group [entity=organization]+",
            "[entity=religion]+ <amod [entity=organization]+",
        ],
        "per:other_family"                   : [
            "[entity=person]+ aunts , [entity=person]+",
            "[entity=person]+ relative of [entity=person]+",
        ],
        "per:stateorprovince_of_birth"       : [
            "[entity=person]+ was born in [entity=state_or_province]+",
            "[entity=person]+ <nsubj born >nmod_in [entity=state_or_province]+",

        ],
        "org:dissolved"                      : [
            "[entity=organization]+ dissolved on [entity=date]+",
            "[entity=organization]+ <nsubj dissolved >nmod_on [entity=date]+",
            "[entity=organization]+ <nsubj dissolved >nmod_in [entity=date]+",
        ],
        "per:date_of_death"                  : [
            "[entity=person]+ died [entity=date]+",
            "[entity=person]+ <nsubj died >nmod_in [entity=date]+",
        ],
        "org:shareholders"                   : [
            "[entity=person]+ <nsubj shareholders >nmod_at [entity=organization]+",
            "[entity=person]+ owns stock in [entity=organization]+",
            "[entity=person]+ who owns stock in [entity=organization]+",

        ],
        "per:alternate_names"                : [
            "[entity=person]+ , whose real name is [entity=person]+",
            "[entity=person]+ known as [entity=person]+",
            "[entity=person]+ also named [entity=person]+",
        ],
        "per:parents"                        : [
            "[entity=person]+ -lrb- son of [entity=person]+",
            "[entity=person]+ >appos daughter >nmod_of [entity=person]+",
            "[entity=person]+ is the son of [entity=person]+",
        ],
        "per:schools_attended"               : [
            "[entity=person]+ is a graduate of [entity=organization]+",
            "[entity=person]+ <nsubj entered >dobj [entity=organization]+",
            "[entity=person]+ <nsubj graduated >nmod_from [entity=organization]+",
        ],
        "per:cause_of_death"                 : [
            "[entity=person]+ <nsubj died >nmod_of [entity=cause_of_death]+",
            "[entity=person]+ died of [entity=cause_of_death]+",
            "[entity=person]+ <nsubj died >nmod_after battle >nmod_with [entity=cause_of_death]+",
        ],
        "per:city_of_death"                  : [
            "[entity=person]+ <nsubj died >nmod_in [entity=location]+",
        ],
        "per:stateorprovince_of_death"       : [
            "[entity=person]+ died at his home in [entity=state_or_province]+",
            "[entity=person]+ died in [entity=state_or_province]+",
        ],
        "org:founded"                        : [
            "[entity=organization]+ , established in [entity=date]+",
            "[entity=organization]+ , which was founded in [entity=date]+",
        ],
        "per:country_of_birth"               : [
            "[entity=person]+ , who was born in [entity=country]+",
            "[entity=person]+ 's birthplace is [entity=country]+",
            "[entity=person]+ <nsubj born >nmod_in [entity=country]+",
        ],
        "per:date_of_birth"                  : [
            "[entity=person]+ was born on [entity=date]+",
            "[entity=person]+ <nsubjpass born >nmod_on [entity=date]+",
        ],
        "per:city_of_birth"                  : [
            "[entity=person]+ >nmod_in [entity=location]+",
            "[entity=person]+ , who was born in [entity=city]+",
        ],
        "per:charges"                        : [
            "[entity=person]+ was accused with two others on federal charges of [entity=criminal_charge]+", 
            "[entity=person]+ convicted of [entity=criminal_charge]+", 
            "[entity=person]+ was accused of [entity=criminal_charge]+", 
            "[entity=person]+ <nsubj accused >nmod_of [entity=criminal_charge]+", 
        ],
        "per:country_of_death"               : [
            "[entity=person]+ <nsubj died >nmod_in [entity=country]+",
        ],
    }

    return [(x, z) for (x, y) in rules.items() for z in y]

def custom_rules3():
    rules = {
        'org:founded_by'                     : [
            ("[entity=PERSON]+ <nsubj founded >dobj [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", False),

        ],
        'per:employee_of'                    : [
            ("[entity=PERSON]+ works for [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ <nsubj works >nmod_for [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ is employed by [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ is an employee at [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
        ],
        'org:alternate_names'                : [
            ("[entity=SUBJECT ORGANIZATION]+ is also known as [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
        ],
        "per:cities_of_residence"            : [
            ("[entity=PERSON]+ <nsubj native >nmod_of [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ <nsubj moved >nmod_to [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ <nmod:poss home >nmod_in [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ <nsubj lives >nmod_in [entity=CITY]+", "PERSON", "CITY", True),
        ],
        "per:children"                       : [
            ("[entity=CHILD]+ daughter of [entity=PARENT]+", "CHILD", "PARENT", True),
            ("[entity=CHILD]+ son of [entity=PARENT]+", "CHILD", "PARENT", True),
        ],
        "per:title"                          : [
            ("[entity=TITLE]+ [entity=PERSON]+", "TITLE", "PERSON", False),
            ("[entity=TITLE]+ <compound [entity=PERSON]+", "TITLE", "PERSON", False),
            ("[entity=TITLE]+ <appos [entity=PERSON]+", "TITLE", "PERSON", False),
        ],
        "per:siblings"                       : [
            ("[entity=SUBJECT PERSON]+ is a sibbling of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ is brother of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ is sister of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SIBBLING]+ of [entity=SIBBLING]+", "SIBBLING", "SIBBLING", True),
        ],
        "per:religion"                       : [
            ("[entity=PERSON]+ <nsubj has >nmod [entity=RELIGION]+", "PERSON", "RELIGION", True),
            ("[entity=PERSON]+ is [entity=RELIGION]+", "PERSON", "RELIGION", True),
        ],
        "per:age"                            : [
            ("[entity=PERSON]+ , [entity=NUMBER]+", "PERSON", "NUMBER", True),
            ("[entity=PERSON]+ <nsubj dies >nmod_at [entity=NUMBER]+", "PERSON", "NUMBER", True),
            ("[entity=PERSON]+ is [entity=NUMBER]+", "PERSON", "NUMBER", True),
            ("[entity=PERSON]+ ' age is [entity=NUMBER]+", "PERSON", "NUMBER", True),
        ],
        "org:website"                        : [
            ("[entity=ORGANIZATION]+ : [entity=URL]+", "ORGANIZATION", "URL", True),
            ("[entity=ORGANIZATION]+ <compound [entity=URL]+", "ORGANIZATION", "URL", True),
            ("[entity=ORGANIZATION]+ [entity=URL]+", "ORGANIZATION", "URL", True),
        ],
        "per:stateorprovinces_of_residence"  : [
            ("[entity=PERSON]+ <nsubj native >nmod_of [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
            ("[entity=PERSON]+ <nsubj moved >nmod_to [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
            ("[entity=PERSON]+ <nmod:poss home >nmod_in [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
            ("[entity=PERSON]+ <nsubj lives >nmod_in [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
        ],
        "org:member_of"                      : [
            ("[entity=SUBJECT ORGANIZATION]+ joined [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ <nsubj is >nmod_of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ is part of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ 's , [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
        ],
        "org:top_members/employees"          : [
            ("[entity=PERSON]+ >appos employee >nmod_of [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", False),
            ("[entity=PERSON]+ is a top employee of [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", False),
        ],
        "per:countries_of_residence"         : [
            ("[entity=PERSON]+ <nsubj native >nmod_of [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ <nsubj moved >nmod_to [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ <nmod:poss home >nmod_in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ <nsubj lives >nmod_in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
        ],
        "org:city_of_headquarters"           : [
            ("[entity=ORGANIZATION]+ just outside [entity=CITY]+", "ORGANIZATION", "CITY", True),
            ("[entity=ORGANIZATION]+ >appos firm >nmod_in [entity=CITY]+", "ORGANIZATION", "CITY", True),
            ("[entity=CITY]+ [entity=ORGANIZATION]+", "CITY", "ORGANIZATION", False),
            ("[entity=ORGANIZATION]+ in [entity=CITY]+", "ORGANIZATION", "CITY", True),
            ("[entity=ORGANIZATION]+ >nmod_in [entity=CITY]+", "ORGANIZATION", "CITY", True),
            ("[entity=ORGANIZATION]+ has headquarters in [entity=CITY]+", "ORGANIZATION", "CITY", True),
        ],
        "org:members"                        : [
            ("[entity=OBJECT ORGANIZATION]+ includes [entity=SUBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=SUBJECT ORGANIZATION]+ is a member of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ joined [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ >nmod_of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
        ],
        "org:country_of_headquarters"        : [
            ("[entity=ORGANIZATION]+ >appos firm >nmod_in [entity=COUNTRY]+", "ORGANIZATION", "COUNTRY", True),
            ("[entity=ORGANIZATION]+ >appos firm >nmod_in [entity=COUNTRY]+", "ORGANIZATION", "COUNTRY", True),
            ("[entity=COUNTRY]+ [entity=ORGANIZATION]+", "COUNTRY", "ORGANIZATION", False),
            ("[entity=ORGANIZATION]+ in [entity=COUNTRY]+", "ORGANIZATION", "COUNTRY", True),
            ("[entity=ORGANIZATION]+ >nmod_in [entity=COUNTRY]+", "ORGANIZATION", "COUNTRY", True),
            ("[entity=ORGANIZATION]+ has headquarters in [entity=COUNTRY]+", "ORGANIZATION", "COUNTRY", True),
        ],
        "per:spouse"                         : [
            ("[entity=SUBJECT PERSON]+ spouse of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ marriage to [entity=OBJECT PERSON]+", "PERSON", "PERSON", False),
        ],
        "org:stateorprovince_of_headquarters": [
            ("[entity=ORGANIZATION]+ >appos firm >nmod_in [entity=STATE_OR_PROVINCE]+", "ORGANIZATION", "STATE_OR_PROVINCE", True),
            ("[entity=ORGANIZATION]+ >appos firm >nmod_in [entity=STATE_OR_PROVINCE]+", "ORGANIZATION", "STATE_OR_PROVINCE", True),
            ("[entity=STATE_OR_PROVINCE]+ [entity=ORGANIZATION]+", "STATE_OR_PROVINCE", "ORGANIZATION", False),
            ("[entity=ORGANIZATION]+ in [entity=STATE_OR_PROVINCE]+", "ORGANIZATION", "STATE_OR_PROVINCE", True),
            ("[entity=ORGANIZATION]+ >nmod_in [entity=STATE_OR_PROVINCE]+", "ORGANIZATION", "STATE_OR_PROVINCE", True),
            ("[entity=ORGANIZATION]+ has headquarters in [entity=STATE_OR_PROVINCE]+", "ORGANIZATION", "STATE_OR_PROVINCE", True),
        ],
        "org:number_of_employees/members"    : [
            ("[entity=ORGANIZATION]+ , with [entity=NUMBER]+", "ORGANIZATION", "NUMBER", True),
            ("[entity=ORGANIZATION]+ employs nearly [entity=NUMBER]+", "ORGANIZATION", "NUMBER", True),
            ("[entity=ORGANIZATION]+ has about [entity=NUMBER]+", "ORGANIZATION", "NUMBER", True),
            ("[entity=ORGANIZATION]+ <nsubj has >dobj employees >nummod [entity=NUMBER]+", "ORGANIZATION", "NUMBER", True),
            ("[entity=ORGANIZATION]+ <nsubj has >dobj members >nummod [entity=NUMBER]+", "ORGANIZATION", "NUMBER", True),
        ],
        "org:parents"                        : [
            ("[entity=OBJECT ORGANIZATION]+ , a SUBJECT part of [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=OBJECT ORGANIZATION]+ is a SUBJECT branch of [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=OBJECT ORGANIZATION]+ is a SUBJECT subsidiary of [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=SUBJECT ORGANIZATION]+ <nsubj parent >nmod_of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=OBJECT ORGANIZATION]+ <nsubj unit >SUBJECT nmod_of [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=SUBJECT ORGANIZATION]+ <nsubj bought >dobj [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=OBJECT ORGANIZATION]+ >nmod_under [entity=SUBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=OBJECT ORGANIZATION]+ >appos arm >SUBJECT nmod:poss [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=OBJECT ORGANIZATION]+ 's [entity=SUBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=OBJECT ORGANIZATION]+ owned by [SUBJECT entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
        ],
        "org:subsidiaries"                   : [
            ("[entity=SUBJECT SUBSIDIARY]+ , a part of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ is a branch of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ is a subsidiary of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=OBJECT PARENT ORGANIZATION]+ <nsubj SUBJECT parent >nmod_of [entity=SUBSIDIARY ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=SUBJECT ORGANIZATION]+ <nsubj unit >nmod_of [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=OBJECT ORGANIZATION]+ <nsubj bought >SUBJECT dobj [entity=ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", False),
            ("[entity=SUBJECT ORGANIZATION]+ >nmod_under [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ >appos arm >nmod:poss [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ 's [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
            ("[entity=SUBJECT ORGANIZATION]+ owned by [entity=OBJECT ORGANIZATION]+", "ORGANIZATION", "ORGANIZATION", True),
        ],
        "per:origin"                         : [
            ("[entity=NATIONALITY]+ <nsubj [entity=PERSON]+", "NATIONALITY", "PERSON", False),
            ("[entity=NATIONALITY]+ is the nationality of [entity=PERSON]+", "NATIONALITY", "PERSON", False),
            ("[entity=ENTITY]+ is the nationality of [entity=PERSON]+", "ENTITY", "PERSON", False),
            ("[entity=COUNTRY]+ of [entity=PERSON]+", "COUNTRY", "PERSON", False),
            ("[entity=PERSON]+ is [entity=NATIONALITY]+", "PERSON", "NATIONALITY", True),
            ("[entity=PERSON]+ is originally from [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ 's nationality is [entity=NATIONALITY]+", "PERSON", "NATIONALITY", True),

        ],
        "org:political/religious_affiliation": [
            ("[entity=IDEOLOGY]+ group [entity=ORGANIZATION]+", "IDEOLOGY", "ORGANIZATION", False),
            ("[entity=RELIGION]+ group [entity=ORGANIZATION]+", "RELIGION", "ORGANIZATION", False),
            ("[entity=RELIGION]+ <amod [entity=ORGANIZATION]+", "RELIGION", "ORGANIZATION", False),
            ("[entity=ORGANIZATION]+ has a religious affiliation with [entity=RELIGION]+", "ORGANIZATION", "RELIGION", True),
            ("[entity=ORGANIZATION]+ has a political affiliation with [entity=IDEOLOGY]+", "ORGANIZATION", "IDEOLOGY", True),
        ],
        "per:other_family"                   : [
            ("[entity=SUBJECT PERSON]+ is relative of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ is family of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ is related to [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
        ],
        "per:stateorprovince_of_birth"       : [
            ("[entity=PERSON]+ was born in [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
            ("[entity=PERSON]+ <nsubj born >nmod_in [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),

        ],
        "org:dissolved"                      : [
            ("[entity=ORGANIZATION]+ dissolved on [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ existed until [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ dissolved in [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ disbanded in [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ <nsubj dissolved >nmod_on [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ <nsubj dissolved >nmod_in [entity=DATE]+", "ORGANIZATION", "DATE", True),
        ],
        "per:date_of_death"                  : [
            ("[entity=PERSON]+ died [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ <nsubj died >nmod:tmod [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ <nsubj died >nmod_in [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ <nsubj died >nmod_on [entity=DATE]+", "PERSON", "DATE", True),
        ],
        "org:shareholders"                   : [
            ("[entity=PERSON]+ <nsubj shareholders >nmod_at [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ owns stock in [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ is a shareholder in [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=SHAREHOLDER]+ of [entity=ORGANIZATION]+", "SHAREHOLDER", "ORGANIZATION", True),
            ("[entity=PERSON]+ holds shares in [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ who owns stock in [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),

        ],
        "per:alternate_names"                : [
            ("[entity=SUBJECT PERSON]+ , whose real name is [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ known as [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ also named [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
        ],
        "per:parents"                        : [
            ("[entity=SUBJECT PERSON]+ -lrb- son of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ >appos daughter >nmod_of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=SUBJECT PERSON]+ is the son of [entity=OBJECT PERSON]+", "PERSON", "PERSON", True),
            ("[entity=PARENT]+ of [entity=PERSON]+", "PARENT", "PERSON", True),
            ("[entity=PARENT]+ of [entity=CHILD]+", "PARENT", "CHILD", True),
        ],
        "per:schools_attended"               : [
            ("[entity=PERSON]+ is a graduate of [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ <nsubj graduated >nmod_from [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ <nsubj studied >nmod_at [entity=ORGANIZATION]+", "PERSON", "ORGANIZATION", True),
            ("[entity=PERSON]+ attended [entity=SCHOOL]+", "PERSON", "SCHOOL", True),
        ],
        "per:cause_of_death"                 : [
            ("[entity=PERSON]+ <nsubj died >nmod_of [entity=CAUSE_OF_DEATH]+", "PERSON", "CAUSE_OF_DEATH", True),
            ("[entity=PERSON]+ <nsubj died >nmod_from [entity=CAUSE_OF_DEATH]+", "PERSON", "CAUSE_OF_DEATH", True),
            ("[entity=PERSON]+ died of [entity=CAUSE_OF_DEATH]+", "PERSON", "CAUSE_OF_DEATH", True),
            ("[entity=PERSON]+ <nsubj died >nmod_after battle >nmod_with [entity=CAUSE_OF_DEATH]+", "PERSON", "CAUSE_OF_DEATH", True),
        ],
        "per:city_of_death"                  : [
            ("[entity=PERSON]+ <nsubj died >nmod_in [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ died in [entity=CITY]+", "PERSON", "CITY", True),
        ],
        "per:stateorprovince_of_death"       : [
            ("[entity=PERSON]+ died in [entity=STATE_OR_PROVINCE]+", "PERSON", "STATE_OR_PROVINCE", True),
        ],
        "org:founded"                        : [
            ("[entity=ORGANIZATION]+ , established in [entity=DATE]+", "ORGANIZATION", "DATE", True),
            ("[entity=ORGANIZATION]+ , which was founded in [entity=DATE]+", "ORGANIZATION", "DATE", True),
        ],
        "per:country_of_birth"               : [
            ("[entity=PERSON]+ <nsubj born >nmod_in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ <nsubjpass born >nmod_in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ , who was born in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ 's birthplace is [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ born in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
        ],
        "per:date_of_birth"                  : [
            ("[entity=PERSON]+ was born on [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ <nsubjpass born >nmod_on [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ <nsubjpass born >nmod_in [entity=DATE]+", "PERSON", "DATE", True),
            ("[entity=PERSON]+ 's birthday is on [entity=DATE]+", "PERSON", "DATE", True),
        ],
        "per:city_of_birth"                  : [
            ("[entity=PERSON]+ <nsubj born >nmod_in [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ <nsubjpass born >nmod_in [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ , who was born in [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ 's birthplace is [entity=CITY]+", "PERSON", "CITY", True),
            ("[entity=PERSON]+ born in [entity=CITY]+", "PERSON", "CITY", True),
        ],
        "per:charges"                        : [
            ("[entity=PERSON]+ was accused with two others on federal charges of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
            ("[entity=PERSON]+ convicted of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
            ("[entity=PERSON]+ was convicted of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
            ("[entity=CRIMINAL_CHARGE]+ are the charges of [entity=PERSON]+",  "CRIMINAL_CHARGE", "PERSON", False),
            ("[entity=PERSON]+ was accused of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
            ("[entity=PERSON]+ <nsubj accused >nmod_of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
            ("[entity=PERSON]+ <nsubj charged >nmod_of [entity=CRIMINAL_CHARGE]+",  "PERSON", "CRIMINAL_CHARGE", True),
        ],
        "per:country_of_death"               : [
            ("[entity=PERSON]+ <nsubj died >nmod_in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
            ("[entity=PERSON]+ died in [entity=COUNTRY]+", "PERSON", "COUNTRY", True),
        ],
    }


    return [(x, z[0]) for (x, y) in rules.items() for z in y]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,) # /storage/rvacareanu/code/projects_7_2309/softrules/lightning_logs/version_113/checkpoints/epoch=0-step=85000.ckpt
    parser.add_argument("--how_many_rules_to_average", type=int, default=1)
    parser.add_argument("--rules_path", type=str, nargs='+', default=[
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/surface.jsonl", 
        "/storage/rvacareanu/data/softrules/rules/fsre_dataset/TACRED_full_trainonly/enhanced_syntax.jsonl", 
    ])
    parser.add_argument("--train_path", type=str, default="/storage/rvacareanu/data/softrules/fsre_dataset/TACRED_full/train.json")
    parser.add_argument("--test_path",  type=str, default="/storage/rvacareanu/data/softrules/fsre_dataset/TACRED_full/dev.json")

    parser.add_argument("--default_predictions_path",  type=str, default=None)

    parser.add_argument('--use_rules_for_no_relation', action='store_true', help='Whether to use rules for the `no_relation` class.')
    parser.add_argument('--unique_rules', action='store_true')
    parser.add_argument("--keep_rules_above_count", type=int, default=0, help="Keep the rules (<`relation`, `rule`> tuples) that appear more (>) than this number")


    args = vars(parser.parse_args())
    print(args)
    pl.seed_everything(1)
    model = SoftRulesEncoder.load_from_checkpoint(args['checkpoint']).eval()
    model.thresholds = np.linspace(0, 1, 101).tolist()
    model.hyperparameters['append_results_to_file'] = None
    model.hyperparameters['dev_path'] = args['test_path']
    model.hyperparameters['how_many_rules_to_average'] = args['how_many_rules_to_average']

    with open(args['train_path']) as fin:
        train = json.load(fin)

    if args['default_predictions_path'] is not None and args['default_predictions_path'] != '':
        with open(args['default_predictions_path']) as fin:
            default_predictions_path = json.load(fin)
    else:
        default_predictions_path = {}

    train_rules = custom_rules3()
    print(len(train_rules))
    
    # Store the relation associated with each rule
    rules_relations = [x[0] for x in train_rules]

    # Tokenize, then construct the dataset and the dataloader
    rules_tokenized = model.tokenize_rules([x[1].lower() for x in train_rules])
    rules_dataset = datasets.Dataset.from_dict(rules_tokenized)
    rules_dl = DataLoader(rules_dataset, batch_size=64, collate_fn=lambda rule_inputs: model.data_collator_rule(rule_inputs))
    
    # Finally, construct the encodings of the sentences
    rules_encodings = []
    for batch in tqdm.tqdm(rules_dl):
        rules_encodings.append(model.encode_rule({k:v.to(model.device) for (k, v) in batch.items()}).detach().cpu().numpy())

    with open(args['test_path']) as fin:
        test_sentences = json.load(fin)
    
    # Store the relation of each sentence
    sentences_gold = [x['relation'] for x in test_sentences]

    # Tokenize, then construct the dataset and the dataloader
    sentences_tokenized = model.tokenize_sentences([preprocess_line(x, 'typed_entity_marker_punct_v3') if x['subj_type'] == x['obj_type'] else preprocess_line(x, 'typed_entity_marker_punct_v2') for x in test_sentences])
    sentences_dataset = datasets.Dataset.from_dict(sentences_tokenized)
    sentences_dl = DataLoader(sentences_dataset, batch_size=64, collate_fn=lambda sentences_inputs: model.data_collator_sentence(sentences_inputs))

    # Finally, construct the encodings of the sentences
    sentences_encodings = []
    for batch in tqdm.tqdm(sentences_dl):
        sentences_encodings.append(model.encode_sent({k:v.to(model.device) for (k, v) in batch.items()}).detach().cpu().numpy())

    # Stack every rule and every sentence
    x1 = np.vstack(rules_encodings)
    x2 = np.vstack(sentences_encodings)

    # Store the similarities between each rule and each sentence
    result = x1 @ x2.T
    preds_max  = result.max(axis=0)
    preds_ids  = result.argmax(axis=0)
    # preds_rels = np.array([rules_relations[x] for x in result.argmax(axis=0)])

    # Use the standard notation of `gold` to hold the gold predictions
    gold = sentences_gold

    print("-"*10)
    print("Eval 1: simply take the rule with the maximum similarity (equivalent to one of the evaluations below where `how_man=1`)")
    # Iterate over some thresholds; Take the maximum similarity; If it is >= threshold, => predict the the associated relation
    # Otherwise, predict no relation
    # Lastly, compute the tacred score
    # for threshold in np.linspace(0.2, 1, 81).tolist():
    for threshold in np.linspace(0.2, 1, 33).tolist():
        preds_rels = np.array([rules_relations[x] for x in result.argmax(axis=0)])
        preds_rels[preds_max < threshold] = 'no_relation'
        pred = preds_rels.tolist()
        print(threshold, tacred_score(gold, pred, verbose=False))
    print("-"*10)


    # In the evaluation above we do not use `logit_scale`, a parameter used during training to scale the predictions
    # And we do not use softmax
    # Below, it is an evaluation that attempts to fix these two potential issues (they might not be issues, actually)
    relations_to_rule_ids = defaultdict(list)
    for i, r in enumerate(rules_relations):
        relations_to_rule_ids[r].append(i)

    # A map from relation to the indices for the rules associated with this relation
    relations_to_rule_ids = {k:np.array(v) for (k, v) in sorted(relations_to_rule_ids.items(), key=lambda x: x[0])}

    # The value with which to scale the similarities
    logit_scale = torch.exp(model.logit_scale).detach().cpu().numpy()

    relations_to_sent_ids = defaultdict(list)
    for i, r in enumerate(gold):
        relations_to_sent_ids[r].append(i)

    # A map from relation to the indices for the sentences associated with this relation (gold)
    relations_to_sent_ids = {k:np.array(v) for (k, v) in sorted(relations_to_sent_ids.items(), key=lambda x: x[0])}

    # Construct a list of lists; each column (i.e. sentence) has a list with two elements: relation, and scores for that relation
    # so `all_pred_for_col[0]` gives a list associated with sentence 0. And `all_pred_for_col[0][0]` looks like so: (<relation>, <list with similarities between sentence 0 and rules with <relation> relation>)
    all_pred_for_col = []
    for col in tqdm.tqdm(range(result.shape[1])):
        pred_for_col = []
        for relation, ids in relations_to_rule_ids.items():
            pred_for_col.append([relation, sorted(result[:,col][ids], reverse=True)])
        all_pred_for_col.append(pred_for_col)

    print("\n\n")
    print("-"*20)
    print("\n\n")
    print("Average top X")



    # Iterate over how many rules for each relation to consider
    for how_many_to_average in [1, 3, 5]:
        # how_many_to_average=1
        all_sents_text = []
        all_preds = []
        all_rels  = []
        for col in tqdm.tqdm(range(result.shape[1])):
            pred_for_col = [[x[0], np.mean(x[1][:how_many_to_average])] for x in all_pred_for_col[col]]
            all_sents_text.append(preprocess_line(test_sentences[col], 'typed_entity_marker_punct_v2'))
            # all_preds.append((logit_scale*spp.softmax([x[1] for x in pred_for_col])).tolist())
            all_preds.append([x[1] for x in pred_for_col])
            all_rels.append([x[0] for x in pred_for_col])
        final_result = compute_results_with_thresholds(gold, all_preds, all_rels, np.linspace(0.2, 1, 81).tolist(), verbose=False, overwrite_results=default_predictions_path)
        best = max(final_result, key=lambda x: x['f1_tacred'])
        print({'how_many_to_average': how_many_to_average, **best})
        print(compute_results_with_thresholds(gold, all_preds, all_rels, [best['threshold']], verbose=True, overwrite_results=default_predictions_path))

    # for (a,b) in [(i, ast) for i, (ast, ap, ar, g) in enumerate(zip(all_sents_text, all_preds, all_rels, gold)) if g == 'per:parents'][:10]:
    #     print('-'*20)
    #     print(b)
    #     print(sorted(list(zip(all_preds[a], all_rels[a])), key=lambda x: x[0], reverse=True)[:3])
    #     print('-'*20)
    #     print("\n\n")

    # import pandas as pd
    # temp_result = []
    # for (idx, (rule, rule_relation)) in enumerate(train_rules):
    #     for r in relations_to_sent_ids.keys():
    #         for similarity in result[idx, relations_to_sent_ids[r]].tolist():
    #             temp_result.append({
    #                 'relation': r,
    #                 'similarity': similarity,
    #             })
    #     df = pd.DataFrame(temp_result)
    #     df['rule'] = rule
    #     df['rule_relation'] = rule_relation
    #     print("-"*10)
    #     print(rule_relation, rule)
    #     print(df.groupby(by=['relation']).agg({'similarity': ['mean', 'std', 'count']}).reset_index().sort_values(('similarity', 'mean')).reset_index())
    #     print("-"*10)
    # # df.to_csv('/storage/rvacareanu/code/projects_7_2309/softrules/results/231122/interesting_rules/rule3.csv', sep=',', index=False)

    print("\n\n")
    print("-"*20)
    print("\n\n")
    # print("Average top X% instead of top X")

    # # Iterate over how many rules for each relation to consider
    # for how_many_to_average in [0.01, 0.1, 0.2, 0.5, 0.8]:
    #     all_preds = []
    #     all_rels  = []
    #     for col in tqdm.tqdm(range(result.shape[1])):
    #         pred_for_col = [[x[0], np.mean(x[1][:max(int(how_many_to_average * len(x[1])), 1)])] for x in all_pred_for_col[col]]
    #         # all_preds.append((logit_scale*spp.softmax([x[1] for x in pred_for_col])).tolist())
    #         all_preds.append([x[1] for x in pred_for_col])
    #         all_rels.append([x[0] for x in pred_for_col])
    #     final_result = compute_results_with_thresholds(gold, all_preds, all_rels, np.linspace(0.2, 1, 81).tolist(), verbose=False, overwrite_results=default_predictions_path)
    #     best = max(final_result, key=lambda x: x['f1_tacred'])
    #     print({'how_many_to_average': how_many_to_average, **best})

    #     print(compute_results_with_thresholds(gold, all_preds, all_rels, [best['threshold']], verbose=False, overwrite_results=default_predictions_path))

    # # rules = read_rules(args['rules'])
    # # rules = dict(rules)
    # # val_data = get_valdata({**model.hyperparameters, **args}, model=model)

    # # trainer = Trainer(
    # #     accelerator="gpu", 
    # #     precision='16-mixed',
    # # )
    
    # # for vd in val_data:
    # #     print("-"*25)
    # #     print(vd)
    # #     trainer.validate(model=model,dataloaders=DataLoader(dataset=vd, collate_fn=model.collate_tokenized_fn, batch_size=64, num_workers=32))
    # #     print("-"*25)
