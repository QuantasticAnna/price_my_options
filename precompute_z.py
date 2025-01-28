import numpy as np
import joblib

# Temporary: in the first version of the app, Z can not be recompute, its always the same, to facilitate development
# because when we compute Z from the UI, it takes a lot of time 

# Precompute Z and store it
def generate_Z(n_simulations=50000, n_steps=252, filename="Z_precomputed.joblib"):
    Z = np.random.standard_normal((n_simulations, n_steps))
    joblib.dump(Z, filename)
    print(f"Precomputed Z saved to {filename}")

# Run this script once to generate and save Z
generate_Z()