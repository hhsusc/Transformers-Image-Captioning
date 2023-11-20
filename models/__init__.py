from models.pure_transformer import PureT
from models.pure_transformer import PureT_Base
from models.pure_transformer import PureT_Base_22K
from models.pure_transformer import PureT_Swin_v2
from models.pure_transformer import PureT_CSwin

__factory = {
    'PureT': PureT,
    'PureT_Base': PureT_Base,
    'PureT_Base_22K': PureT_Base_22K,
    'PureT_Swin_v2': PureT_Swin_v2,
    'PureT_CSwin': PureT_CSwin
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)