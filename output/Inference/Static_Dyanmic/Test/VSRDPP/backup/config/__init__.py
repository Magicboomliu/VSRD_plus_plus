from .defaults import _C as cfg

from .defaults2 import _C as cfg2

from .test_evaluation import _C as cfg3

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Van': -4,
    'Truck': -4,
    'Person_sitting': -2,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1,
}