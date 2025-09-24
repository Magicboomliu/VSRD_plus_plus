
# old splits
from .vsrd24_splits_vsrd_configs import _C as cfg_vsrd_24_splits_vsrd
from .vsrd24_splits_vsrdpp_configs import _C as cfg_vsrd_24_splits_vsrd_pp
from .vsrd24_splits_autolabels_configs import _C as cfg_vsrd_24_splits_autolabels
from .vsrd24_splits_gt_configs import _C as cfg_vsrd_24_splits_gt

# new splits
from .casual_splits_vsrd_configs import _C as cfg_casual_splits_vsrd
from .casual_splits_vsrdpp_configs import _C as cfg_casual_splits_vsrd_pp
from .casual_splits_autolabels_configs import _C as cfg_casual_splits_autolabels

from .causal_splits_gt_configs import _C as cfg_casual_splits_gt



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