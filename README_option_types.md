# Exotic Options Definitions

This document provides definitions for the various exotic options supported by the project.

## 1. Asian Options  DONE
Asian options are path-dependent options where the payoff depends on the average price of the underlying asset over a specified period.

- **Asian Call Option**:
  The holder has the right to buy the underlying asset at a strike price `K`, with the payoff calculated as:
  `Payoff = max(Average Price - K, 0)`

- **Asian Put Option**:
  The holder has the right to sell the underlying asset at a strike price `K`, with the payoff calculated as:
  `Payoff = max(K - Average Price, 0)`

---

## 2. Barrier Options DONE
Barrier options are options that are activated (knock-in) or deactivated (knock-out) if the underlying asset reaches a specific barrier price during the option's life.

- **Down-and-Out Call Option**:
  A call option that becomes worthless if the underlying price falls below a barrier `B`. The payoff is:
  `Payoff = max(S_T - K, 0)` if the stock price S_t > B for all t during the option's life

- **Up-and-Out Put Option**:
  A put option that becomes worthless if the underlying price rises above a barrier `B`. The payoff is:
  `Payoff = max(K - S_T, 0)` if the stock price S_t < B for all t during the option's life

---

## 3. Lookback Options DONE
Lookback options allow the holder to "look back" at the underlying asset's price history to determine the payoff, using the maximum or minimum price during the option's life.

- **Lookback Call Option**:
  The payoff is based on the maximum price of the underlying during the option's life:
  `Payoff = max(S_max - K, 0)`, where S_max is the highest stock price reached during the option's life

- **Lookback Put Option**:
  The payoff is based on the minimum price of the underlying during the option's life:
  `Payoff = max(K - S_min, 0)`, where S_min is the lowest stock price reached during the option's life

---

## 4. Digital (Binary) Options
Digital options provide a fixed payout if the underlying asset meets certain conditions at expiration.

- **Cash-or-Nothing Call**:
  Pays a fixed amount `Q` if the underlying price exceeds the strike price `K` at expiration:
  `Payoff = Q, if S_T > K`; otherwise, `Payoff = 0`

- **Asset-or-Nothing Put**:
  Pays the value of the underlying asset if the price is below the strike price `K` at expiration:
  `Payoff = S_T, if S_T < K`; otherwise, `Payoff = 0`

---

## 5. Cliquet Options
Cliquet options (ratchet options) have a series of reset dates where gains are locked in.

- **Cliquet Option Payoff**:
  The payoff is the sum of gains at each reset date:
  `Payoff = sum(max(S_i - S_{i-1}, 0) for each reset period i)`

---

## 6. Range Options NEXT TODO
Range options depend on how long the underlying asset stays within a predefined price range.

- **Range Option Payoff**:
  The payoff is proportional to the time the underlying spends in the range:
 ` Payoff = K * (Time in Range / Total Time)`, where `K` is a fixed payout multiplier

---

These definitions provide a foundation for understanding the exotic options implemented in the project.
