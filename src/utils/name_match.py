from src.learners.baselines.gdumb import GDUMBLearner
from src.learners.baselines.lump import LUMPLearner
from src.learners.baselines.scr import SCRLearner
from src.learners.baselines.agem import AGEMLearner
from src.learners.baselines.er import ERLearner
from src.learners.baselines.ocm import OCMLearner

from src.learners.baselines.derpp import DERppLearner
from src.learners.baselines.er_ace import ER_ACELearner
from src.learners.baselines.dvc import DVCLearner
from src.learners.fd import FDLearner

from src.buffers.reservoir import Reservoir


learners = {
    'ER':   ERLearner,
    'SCR':  SCRLearner,
    'AGEM': AGEMLearner,
    'LUMP': LUMPLearner,
    'GDUMB': GDUMBLearner,
    'FD': FDLearner,
    'OCM': OCMLearner,
    'DERpp': DERppLearner,
    'ERACE': ER_ACELearner,
    'DVC': DVCLearner,
}

buffers = {
    'reservoir': Reservoir,
}
