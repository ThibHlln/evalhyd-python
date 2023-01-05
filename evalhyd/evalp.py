from typing import List, Dict
from numpy import dtype
from numpy.typing import NDArray
import evalhyd.core


def evalp(q_obs: NDArray[dtype('float64')],
          q_prd: NDArray[dtype('float64')],
          metrics: List[str],
          q_thr: NDArray[dtype('float64')] = None,
          t_msk: NDArray[dtype('bool')] = None,
          m_cdt: NDArray[dtype('S32')] = None,
          bootstrap: Dict[str, int] = None,
          dts: List[str] = None) -> List[NDArray[dtype('float64')]]:
    """Function to evaluate probabilist streamflow predictions"""
    # required arguments
    kwargs = {
        'q_obs': q_obs,
        'q_prd': q_prd,
        'metrics': metrics
    }

    # optional arguments
    if q_thr is not None:
        kwargs['q_thr'] = q_thr
    if t_msk is not None:
        kwargs['t_msk'] = t_msk
    if m_cdt is not None:
        kwargs['m_cdt'] = m_cdt
    if bootstrap is not None:
        kwargs['bootstrap'] = bootstrap
    if dts is not None:
        kwargs['dts'] = dts

    return evalhyd.core._evalp(**kwargs)
