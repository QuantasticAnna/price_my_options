from pricer.asian import pricer_asian, plotter_asian
from pricer.barrier import pricer_barrier, plotter_barrier
from pricer.binary import pricer_binary, plotter_binary
from pricer.lookback import pricer_lookback, plotter_lookback
from pricer.range import pricer_range, plotter_range
from pricer.cliquet import pricer_cliquet, plotter_cliquet
from pricer.european import pricer_european, plotter_european



import numpy as np

# Mapping of exotic option types to their respective pricers
PRICER_MAPPING = {
    'asian': pricer_asian,
    'barrier': pricer_barrier,
    'binary': pricer_binary,
    'lookback': pricer_lookback,
    'range': pricer_range,
    'cliquet': pricer_cliquet,
    'european': pricer_european,
}


# rename these one for delta, if we want to ahve sliders that are de syncrhonized for each greek, we define specifiy arrays 
# VOLATILITY_ARRAY = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
VOLATILITY_ARRAY = np.array([0.1, 0.2])

TTM_ARRAY = np.array([0.0198, 0.0992, 0.3968, 0.9999], dtype=float)  #for some reason if i put 1, its fucks up the dataa types between the slider, dict keys of the differentes deltsas etc, solve later

# VEGA CONSTANTS
# VOLATILITY_ARRAY_VEGA = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) #useful?
VOLATILITY_ARRAY_VEGA = np.array([0.1, 0.2]) # useful?

TTM_ARRAY_VEGA = np.array([0.0198, 0.0992, 0.3968, 0.9999], dtype=float)  #for some reason if i put 1, its fucks up the dataa types between the slider, dict keys of the differentes deltsas etc, solve later


H = 0.01

K_RANGE = np.linspace(50, 150, 20)  # should be flexible around strike price (here we assume its 100)
S0_RANGE = np.linspace(50, 150, 30) # should be flexible around stock price (here we assume its 100)

EXOTIC_TYPE = 'asian'

# for barrier options, should be function of the strike, and should be input in dash but not for now 
B_CALL = 90  # Barrier for Down-and-Out Call
B_PUT = 110 

N_SIMULATIONS = 100000

TTM_RANGE = np.linspace(0.1, 1, 10)


# Exotic options dynamically retrieved from menu_bar
EXOTIC_OPTION_TYPES = ["asian", "lookback", "barrier", "european"] 

GREEKS = ["delta", "gamma", "theta", "vega", "rho"]


# Dictionary of exotic option types and their corresponding plotters
PLOTTERS = {
    "asian": plotter_asian,
    "lookback": plotter_lookback,
    "barrier": plotter_barrier,
    "european": plotter_european,
}
