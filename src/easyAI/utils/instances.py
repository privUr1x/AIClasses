from gc import get_referrers
from copy import deepcopy

def search_instnce_name(inst: object) -> str:
    for referrer in get_referrers(inst):
        if isinstance(referrer, dict):
            for name, val in referrer.items():
                if val is inst:
                    return name 

    return "[Imposible to locate instance name]"

def clone_instance(inst: object) -> object:
    return deepcopy(inst)
