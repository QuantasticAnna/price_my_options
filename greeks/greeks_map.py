from greeks.delta import compute_delta
from greeks.vega import compute_vega
from greeks.rho import compute_rho
from greeks.theta import compute_theta
from greeks.gamma import compute_gamma

greeks_mapping = {
    'delta': compute_delta,
    'gamma': compute_gamma,
    'vega': compute_vega,
    'theta': compute_theta,
    'rho': compute_rho
}
