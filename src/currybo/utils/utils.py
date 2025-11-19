import torch

from typing import Dict

def dict_get_index(d: Dict, index: int):
    """
    Get the value at an index for each array in dict

    Args:
        d: Dict of key:val pairs where val are arrays
        index: the index to extract from each array

    Returns:
        Dict: The Dict with the index extracted as value for each key
    """
    return {key: d[key][index] for key in d.keys()}

def dict_cat(dict1: Dict, dict2: Dict):
    """
    Concatenate the values of two dicts with the same keys

    Args:
        dict1: Dict of key:val pairs where val are arrays
        dict2: Dict of key:val pairs where val are arrays

    Returns:
        Dict: The dictionary with all concatenated arrays
    """
    return {obj: torch.cat([dict1[obj], dict2[obj]], dim=0) for obj in dict1.keys()}

def keyval_to_dict(input: str):
    """
    Takes a keyval string (`key=val,key2=True`) and turns into a dict.

    Args:
        input: keyval string

    Returns:
        Dict: The dictionary resultig from this keyval string
    """
    is_list = lambda s: s[0] == '[' and s[-1] == ']'

    res = []
    for sub in input.split(','):
        if('=') in sub:
            key, val = sub.split('=', 1)
            if(is_list(val)):
                val = list(map(lambda i: parse(i), val[1:-1].split(';')))
            else:
                val = parse(val)
            res.append([key, val])
    return dict(res)

def parse(val):
    """
    Parse any stringified value to its correct python data type

    Args:
        val: The strigified value

    Returns:
        int|float|str|boolean: The parsed value
    """
    is_number = lambda s: s.replace('.','',1).replace('-','',1).isdigit()
    is_bool = lambda s: s == "True" or s == "False"

    if(is_number(val)):
        return int(val) if val.isdigit() else float(val)
    elif(is_bool(val)):
        return True if val == "True" else False
    return val 
