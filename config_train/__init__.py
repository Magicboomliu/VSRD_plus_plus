
from .vsrd_configs import _C as cfg_vsrd
from .vsrd_pp_configs import _C as cfg_vsrd_pp


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