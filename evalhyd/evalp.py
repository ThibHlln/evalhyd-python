from typing import List, Dict
from numpy import dtype
from numpy.typing import NDArray, ArrayLike

try:
    from ._evalhyd import _evalp
except ImportError:
    pass


def evalp(q_obs: NDArray[dtype('float64')],
          q_prd: NDArray[dtype('float64')],
          metrics: List[str],
          q_thr: NDArray[dtype('float64')] = None,
          events: str = None,
          c_lvl: NDArray[dtype('float64')] = None,
          t_msk: NDArray[dtype('bool')] = None,
          m_cdt: ArrayLike = None,
          bootstrap: Dict[str, int] = None,
          dts: ArrayLike = None,
          seed: int = None) -> List[NDArray[dtype('float64')]]:
    """Function to evaluate probabilistic streamflow predictions"""

    # required arguments
    kwargs = {
        'q_obs': q_obs,
        'q_prd': q_prd,
        'metrics': metrics
    }

    # optional arguments
    if q_thr is not None:
        kwargs['q_thr'] = q_thr
    if events is not None:
        kwargs['events'] = events
    if c_lvl is not None:
        kwargs['c_lvl'] = c_lvl
    if t_msk is not None:
        kwargs['t_msk'] = t_msk
    if m_cdt is not None:
        kwargs['m_cdt'] = m_cdt
    if bootstrap is not None:
        kwargs['bootstrap'] = bootstrap
    if dts is not None:
        kwargs['dts'] = dts
    if seed is not None:
        kwargs['seed'] = seed

    return _evalp(**kwargs)
