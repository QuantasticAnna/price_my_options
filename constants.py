from pricer.asian import pricer_asian
from pricer.barrier import pricer_barrier
from pricer.binary import pricer_binary
from pricer.lookback import pricer_lookback
from pricer.range import pricer_range
from pricer.cliquet import pricer_cliquet



import numpy as np

# Mapping of exotic option types to their respective pricers
pricer_mapping = {
    'asian': pricer_asian,
    'barrier': pricer_barrier,
    'binary': pricer_binary,
    'lookback': pricer_lookback,
    'range': pricer_range,
    'cliquet': pricer_cliquet,
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

K_RANGE = np.linspace(50, 150, 10)  # should be flexible around strike price (here we assume its 100)
S0_RANGE = np.linspace(50, 150, 10) # should be flexible around stock price (here we assume its 100)

EXOTIC_TYPE = 'asian'

# for barrier options, should be function of the strike, and should be input in dash but not for now 
B = 90