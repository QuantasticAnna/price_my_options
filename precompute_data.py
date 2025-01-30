import numpy as np
import joblib
from constants import N_SIMULATIONS

# Temporary: in the first version of the app, Z can not be recompute, its always the same, to facilitate development
# because when we compute Z from the UI, it takes a lot of time 

# Precompute Z and store it
def generate_Z(n_simulations=N_SIMULATIONS, n_steps=252, filename="Z_precomputed.joblib"):
    Z = np.random.standard_normal((n_simulations, n_steps))
    joblib.dump(Z, filename)
    print(f"Precomputed Z saved to {filename}")


if __name__ == '__main__':
        
    # Run this script once to generate and save Z
    generate_Z()