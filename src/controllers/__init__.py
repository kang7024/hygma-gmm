REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .hygma_controller import HYGMA
from .gmm_hygma_controller import GMMHYGMA
from .aerial_hygma_controller import HYGMA as AerialHYGMA  # 🔹 추가

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["hygma_mac"] = HYGMA
REGISTRY["gmm_hygma_mac"] = GMMHYGMA
REGISTRY["aerial_hygma_mac"] = AerialHYGMA  # 🔹 추가
