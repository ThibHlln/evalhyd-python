from typing import List, Dict
from numpy import dtype
from numpy.typing import NDArray

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
          m_cdt: NDArray[dtype('|S32')] = None,
          bootstrap: Dict[str, int] = None,
          dts: NDArray[dtype('|S32')] = None,
          seed: int = None,
          diagnostics: List[str] = None) -> List[NDArray[dtype('float64')]]:
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
    if diagnostics is not None:
        kwargs['diagnostics'] = diagnostics

    # check array ranks
    _expected = {
        'q_obs': 2,
        'q_prd': 4,
        'q_thr': 2,
        'c_lvl': 1,
        't_msk': 4,
        'm_cdt': 2,
        'dts': 1
    }

    for arg, val in _expected.items():
        try:
            if kwargs[arg].ndim != val:
                raise RuntimeError(
                    f"'{arg}' must feature {val} {'axis' if val == 1 else 'axes'}"
                )
        except KeyError:
            pass

    return _evalp(**kwargs)
