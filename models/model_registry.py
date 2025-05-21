from .linear_model import LinearModel

from .schnet import SchNetBase
from .schnet_p3m import SchNet_P3M
from .schnet_ewald import SchNetEwald

from .dpp import DimeNetPlusPlusBase, DimeNetPlusPlus_P3M, DimeNetPlusPlusEwald

from .painn import PaiNN, PaiNN_P3M, PaiNNEwald

from .gemnet import GemNetT, GemNetT_P3M, GemNetTEwald 

from .visnet import ViSNet 
from .e2gnn import E2GNN 

MODEL_REGISTRY = {
    'linear': LinearModel,

    'schnet': SchNetBase,
    'schnet-p3m': SchNet_P3M,
    'schnet-ewald': SchNetEwald,

    'dimenetpp': DimeNetPlusPlusBase,
    'dimenetpp-p3m': DimeNetPlusPlus_P3M,
    'dimenetpp-ewald': DimeNetPlusPlusEwald,

    'painn': PaiNN,
    'painn-p3m': PaiNN_P3M,
    'painn-ewald': PaiNNEwald,

    'gemnet-t': GemNetT,
    'gemnet-t-p3m': GemNetT_P3M,
    'gemnet-t-ewald': GemNetTEwald,

    'visnet': ViSNet,

    'e2gnn': E2GNN,
}