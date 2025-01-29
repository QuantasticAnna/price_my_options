# Option Pricing Models

## 1. Black-Scholes Model (BS)
The **Black-Scholes-Merton (BSM) model** provides a closed-form solution for European options.

### **Assumptions:**
- The stock follows **Geometric Brownian Motion (GBM)**.
- No arbitrage opportunities.
- The option is **European-style** (only exercisable at maturity).
- The **volatility (σ) is constant**.
- The risk-free interest rate (r) is constant.
- The market is frictionless (no transaction costs or dividends).

### **Formula (Call Option Price):**
```math
C = S_0 N(d_1) - K e^{-rT} N(d_2)
```
```math
d_1 = \frac{\ln(S_0 / K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}
```
```math
d_2 = d_1 - \sigma \sqrt{T}
```

### **Parameters:**
- \( C \) = Call option price.
- \( S_0 \) = Current stock price.
- \( K \) = Strike price.
- \( T \) = Time to maturity.
- \( r \) = Risk-free interest rate.
- \( σ \) = Volatility.
- \( N(d) \) = Cumulative standard normal distribution.

### **Best For:**
- **European options**.
- **Quick pricing** (closed-form solution).

---

## 2. Binomial Tree Model
The **binomial model** uses a discrete-time approach to model stock price movements.

### **Assumptions:**
- The stock price follows a **multiplicative binomial process**.
- Each time step, the stock can **move up (u) or down (d)**.
- The option can be exercised at any time (**American-style options**).

### **Formula (Stock Price Evolution):**
```math
S_i = S_0 u^i d^{(N-i)}
```

Risk-neutral probability:
```math
p = \frac{e^{r \Delta t} - d}{u - d}
```

### **Parameters:**
- \( u \) = Up factor (\( e^{\sigma \sqrt{\Delta t}} \)).
- \( d \) = Down factor (\( 1/u \)).
- \( r \) = Risk-free rate.
- \( Δ t \) = Time step.
- \( N \) = Number of steps.

### **Best For:**
- **American-style options**.
- **Barrier options**.

---

## 3. Finite Difference Methods (FDM)
Numerical solution of the Black-Scholes PDE.

### **Formula (Black-Scholes PDE):**
```math
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0
```

### **Best For:**
- **Complex options** (barriers, lookbacks).
- **American options**.

---

## 4. Trinomial Tree Model
Similar to binomial but with **three price movements**: up, down, or unchanged.

### **Best For:**
- **More accurate American option pricing**.
- **Path-dependent options**.

---

## 5. Monte Carlo Simulation

### **Best For:**
- **Path-dependent exotics**.
- **Multi-asset derivatives**.

---

## 6. Barone-Adesi and Whaley Approximation
A **semi-analytical method** for pricing **American options**.

### **Best For:**
- **American-style options**.
- **Faster pricing than binomial trees**.

---

## 7. GARCH Option Pricing Model
Volatility **varies over time** rather than being constant.

### **Formula (Variance Evolution):**
```math
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
```

### **Best For:**
- **Options with changing volatility**.
- **Risk management models**.

---

## 8. Jump-Diffusion Models (Merton, Kou)

### **Formula (Merton Jump-Diffusion Model):**
```math
dS_t = \mu S_t dt + \sigma S_t dW_t + J_t dN_t
```

### **Best For:**
- **Options sensitive to price jumps**.
- **Extreme market events**.

---

## Summary Table of Models

| **Model**                        | **Best For**                                       | **Key Features**                          |
|-----------------------------------|---------------------------------------------------|-------------------------------------------|
| **Black-Scholes (BSM)**          | European options.                                | Closed-form solution, constant volatility. |
| **Binomial Tree**                 | American & exotic options.                      | Flexible but computationally expensive.   |
| **Trinomial Tree**                | American & barrier options.                     | More accurate than binomial tree.         |
| **Finite Difference (FDM)**       | PDE-based pricing for complex options.          | Stable but requires numerical methods.    |
| **Monte Carlo Simulation**        | Exotic options, multi-asset derivatives.        | Flexible but slow.                        |
| **Barone-Adesi & Whaley**         | American options.                               | Faster than binomial trees.               |
| **GARCH Model**                   | Options with time-varying volatility.           | More realistic volatility modeling.       |
| **Jump-Diffusion (Merton/Kou)**   | Options sensitive to price jumps.               | Models large price movements.             |

---
