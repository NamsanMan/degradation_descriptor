from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .atten import *
from .encnet import *
from .deeplab import *
from .upernet import *
from .danet import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'atten': get_atten,
        'encnet': get_encnet,
        'upernet': get_upernet,
        'deeplab': get_deeplab,
        'danet': get_danet
    }
    return models[name.lower()](**kwargs)
