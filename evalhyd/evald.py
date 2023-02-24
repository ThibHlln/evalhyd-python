from typing import List, Dict
from numpy import dtype
from numpy.typing import NDArray

try:
    from ._evalhyd import _evald
except ImportError:
    pass


def evald(q_obs: NDArray[dtype('float64')],
          q_prd: NDArray[dtype('float64')],
          metrics: List[str],
          q_thr: NDArray[dtype('float64')] = None,
          events: str = None,
          transform: str = None,
          exponent: float = None,
          epsilon: float = None,
          t_msk: NDArray[dtype('bool')] = None,
          m_cdt: NDArray[dtype('|S32')] = None,
          bootstrap: Dict[str, int] = None,
          dts: NDArray[dtype('|S32')] = None,
          seed: int = None,
          diagnostics: List[str] = None) -> List[NDArray[dtype('float64')]]:
    """Function to evaluate deterministic streamflow predictions"""

    # required arguments
    kwargs = {
        # convert 1D array into 2D array view
        'q_obs': q_obs.reshape(1, q_obs.size) if q_obs.ndim == 1 else q_obs,
        'q_prd': q_prd.reshape(1, q_prd.size) if q_prd.ndim == 1 else q_prd,
        'metrics': metrics
    }

    # optional arguments
    if q_thr is not None:
        kwargs['q_thr'] = (
            q_thr.reshape(1, q_thr.size) if q_thr.ndim == 1 else q_thr
        )
    if events is not None:
        kwargs['events'] = events
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
    if seed is not None:
        kwargs['seed'] = seed
    if diagnostics is not None:
        kwargs['diagnostics'] = diagnostics

    # check array ranks
    _expected = {
        'q_obs': 2,
        'q_prd': 2,
        'q_thr': 2,
        't_msk': 3,
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

    return _evald(**kwargs)
