
from .validation_config import _C as cfg_validation
from .testing_config import _C as cfg_test


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