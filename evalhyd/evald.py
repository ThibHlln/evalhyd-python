from typing import List, Dict
from numpy import dtype
from numpy.typing import NDArray, ArrayLike

try:
    from ._evalhyd import _evald
except ImportError:
    pass


def evald(q_obs: NDArray[dtype('float64')],
          q_prd: NDArray[dtype('float64')],
          metrics: List[str],
          transform: str = None,
          exponent: float = None,
          epsilon: float = None,
          t_msk: NDArray[dtype('bool')] = None,
          m_cdt: ArrayLike = None,
          bootstrap: Dict[str, int] = None,
          dts: ArrayLike = None) -> List[NDArray[dtype('float64')]]:
    """Function to evaluate deterministic streamflow predictions"""

    # required arguments
    kwargs = {
        # convert 1D array into 2D array view
        'q_obs': q_obs.reshape(1, q_obs.size) if q_obs.ndim == 1 else q_obs,
        'q_prd': q_prd.reshape(1, q_prd.size) if q_prd.ndim == 1 else q_prd,
        'metrics': metrics
    }

    # optional arguments
    if transform is not None:
        kwargs['transform'] = transform
    if exponent is not None:
        kwargs['exponent'] = exponent
    if epsilon is not None:
        kwargs['epsilon'] = epsilon
    if t_msk is not None:
        kwargs['t_msk'] = t_msk
    if m_cdt is not None:
        kwargs['m_cdt'] = m_cdt
    if bootstrap is not None:
        kwargs['bootstrap'] = bootstrap
    if dts is not None:
        kwargs['dts'] = dts

    return _evald(**kwargs)
