# Option Pricing Models: Recap

This document provides an overview of various option pricing models, their assumptions, parameters, and the types of options they are best suited for.

---

## 1. Geometric Brownian Motion (GBM)

### Assumptions:
- Constant volatility.
- Continuous price paths (no jumps).
- Log-normal distribution of stock prices.

### Formula:
```math
dS_t = \mu S_t dt + \sigma S_t dW_t
```

### Parameters:
- \(\mu\): Drift rate.
- \(\sigma\): Volatility.

### Best For:
- **Vanilla Options**: European and American options.
- **Simple Exotics**: Barrier options, Asian options (with adjustments).

---

## 2. Jump-Diffusion Model (Merton Model)

### Assumptions:
- Combines continuous GBM with random jumps.
- Jumps follow a Poisson process.

### Formula:
```math
dS_t = \mu S_t dt + \sigma S_t dW_t + J_t dN_t
```

### Parameters:
- \(\mu\), \(\sigma\): Drift and volatility of GBM.
- \(\lambda\): Jump intensity.
- \(\mu_J\), \(\sigma_J\): Mean and standard deviation of jump sizes.

### Best For:
- **Exotics Sensitive to Jumps**: Digital options, Barrier options.

---

## 3. Stochastic Volatility Models (e.g., Heston Model)

### Assumptions:
- Volatility is stochastic and follows its own process.
- Volatility is mean-reverting.

### Formula:
```math
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1
```
```math
dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^2
```

### Parameters:
- \(\mu\): Drift rate.
- \(\kappa\): Speed of mean reversion.
- \(\theta\): Long-term mean variance.
- \(\xi\): Volatility of volatility.
- \(\rho\): Correlation between \(W_t^1\) and \(W_t^2\).

### Best For:
- **Volatility-Sensitive Exotics**: Barrier options, Asian options, Lookback options, Cliquet options, Volatility swaps.

---

## 4. Local Volatility Model (Dupire Model)

### Assumptions:
- Volatility is a deterministic function of stock price and time.
- Calibrated to market prices.

### Formula:
```math
dS_t = \mu S_t dt + \sigma(S_t, t) S_t dW_t
```

### Parameters:
- \(\sigma(S_t, t)\): Local volatility surface.

### Best For:
- **Exotics Requiring Exact Calibration**: Barrier options, Asian options, Lookback options, Autocallables.

---

## 5. Stochastic Volatility Jump-Diffusion Model (Bates Model)

### Assumptions:
- Combines stochastic volatility with jumps.

### Formula:
```math
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1 + J_t dN_t
```
```math
dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^2
```

### Parameters:
- Same as Heston and Merton models.

### Best For:
- **Exotics Sensitive to Both Volatility and Jumps**: Barrier options, Digital options, Cliquet options.

---

## 6. Variance Gamma Model (VG)

### Assumptions:
- Stock returns follow a variance gamma process (pure jumps).
- Captures skewness and kurtosis.

### Formula:
```math
S_t = S_0 e^{(r - q + \omega)t + X_t}
```

### Parameters:
- \(\sigma\): Volatility of the gamma process.
- \(\theta\): Skewness parameter.
- \(\nu\): Kurtosis parameter.

### Best For:
- **Exotics Sensitive to Heavy Tails**: Digital options, Barrier options.

---

## 7. Constant Elasticity of Variance (CEV) Model

### Assumptions:
- Volatility depends on the stock price level.

### Formula:
```math
dS_t = \mu S_t dt + \sigma S_t^{\gamma} dW_t
```

### Parameters:
- \(\mu\), \(\sigma\), \(\gamma\).

### Best For:
- **Exotics with Price-Dependent Volatility**: Barrier options, Asian options.

---

## 8. Regime-Switching Models

### Assumptions:
- Market switches between different regimes (e.g., high/low volatility).

### Formula:
```math
dS_t = \mu_{Z_t} S_t dt + \sigma_{Z_t} S_t dW_t
```

### Parameters:
- Transition probabilities between regimes.
- \(\mu\) and \(\sigma\) for each regime.

### Best For:
- **Exotics in Regime-Shifting Markets**: Cliquet options, Autocallables.

---

## Summary Table

| **Model**                        | **Best For**                                                                 | **Exotics Commonly Priced**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **GBM** DONE                         | Simple exotics with constant volatility.                                   | Barrier, Asian.                                                                            |
| **Jump-Diffusion (Merton)**       | Exotics sensitive to jumps.                                                | Digital, Barrier.                                                                          |
| **Heston (Stochastic Volatility)**| Most exotics, especially volatility-sensitive ones.                        | Barrier, Asian, Lookback, Cliquet, Volatility Swaps.                                       |
| **Local Volatility (Dupire)**     | Exotics requiring exact calibration to market prices.                      | Barrier, Asian, Lookback, Autocallables.                                                  |
| **Bates (SVJD)**                 | Exotics sensitive to both stochastic volatility and jumps.                 | Barrier, Digital, Cliquet.                                                                |
| **Variance Gamma (VG)**           | Exotics sensitive to heavy-tailed returns.                                 | Digital, Barrier.                                                                          |
| **CEV**                          | Exotics with price-dependent volatility.                                   | Barrier, Asian.                                                                            |
| **Regime-Switching**              | Exotics in markets with clear regime shifts.                               | Cliquet, Autocallables.                                                                   |

---

## Key Takeaways:
- **Most Widely Used for Exotics**: Heston and Local Volatility models.
- **Advanced Models**: Bates and Variance Gamma for complex exotics.
- **Simple Models**: GBM and Jump-Diffusion for simpler structures.
- **Market Practice**: Traders prefer models that balance accuracy and computational efficiency.