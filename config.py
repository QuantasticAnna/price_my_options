from pricer.asian import pricer_asian
from pricer.barrier import pricer_barrier
from pricer.binary import pricer_binary
from pricer.lookback import pricer_lookback
from pricer.range import pricer_range
from pricer.cliquet import pricer_cliquet

# Mapping of exotic option types to their respective pricers
pricer_mapping = {
    'asian': pricer_asian,
    'barrier': pricer_barrier,
    'binary': pricer_binary,
    'lookback': pricer_lookback,
    'range': pricer_range,
    'cliquet': pricer_cliquet,
}
