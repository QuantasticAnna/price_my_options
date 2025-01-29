"""
reusable_components.py

This script contains reusable Dash components and utility functions for building
layouts and callbacks in a consistent and modular manner for the Dash app. 
These components are designed to support different exotic option types 
(e.g., Asian, Lookback, Barrier) by dynamically generating layouts 
and plots while minimizing code duplication.

### Purpose:
- To centralize the creation of shared components (e.g., input forms, plot divs, buttons).
- To allow easy extension of the app to support additional exotic option types.
- To ensure that updates to one component (e.g., adding a new Greek plot) propagate across all 
  exotic option types without requiring redundant code changes.

### Key Features:
1. **Dynamic Component Generation**:
    - Generate layouts for exotic options based on the specified option type 
      (e.g., 'asian', 'lookback').
    - Components include input tables, computation buttons, plot placeholders, etc.

2. **Scalability**:
    - Adding a new exotic option requires minimal changes—just update the option type 
      and specific configurations (e.g., pricing method).

3. **Consistency**:
    - Ensures uniform styling, IDs, and structure across different option types.

4. **Reusability**:
    - Components and layouts can be imported into the main app or other scripts 
      for easy integration.

### Included Functions:
- `generate_option_div(option_type)`: Generates the layout for an exotic option.
- `generate_input_table(option_type)`: Generates the input form for option parameters.
- `generate_plot_divs(option_type)`: Creates placeholders for plots (Delta, Vega, Theta, etc.).
- `generate_computation_buttons(option_type)`: Creates buttons for computing Greeks.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from custom_templates import cyborg_template

empty_fig = go.Figure().update_layout(margin=dict(l=50, r=50, t=50, b=50),
                                      template=cyborg_template)

def generate_input_table(exotic_option_type):
    """
    Generates the input table for exotic option parameters.

    Parameters:
        exotic_option_type (str): The type of exotic option (e.g., 'asian', 'lookback').

    Returns:
        html.Div: A Div containing the input table for the specified exotic option type.
    """
    return html.Div([
        html.H4(f"{exotic_option_type.capitalize()} Option Parameters", className="text-center text-light"),
        dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Parameter", className="text-light", style={"width": "50%"}),
                    html.Th("Value", className="text-light", style={"width": "50%"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Initial Stock Price (S₀):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_S0_{exotic_option_type}", type="number", value=100, step=1,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                    html.Tr([
                        html.Td("Strike Price (K):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_K_{exotic_option_type}", type="number", value=100, step=1,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                    html.Tr([
                        html.Td("Time to Maturity (T):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_T_{exotic_option_type}", type="number", value=1, step=0.1,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                    html.Tr([
                        html.Td("Risk-Free Rate (r):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_r_{exotic_option_type}", type="number", value=0.05, step=0.01,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                    html.Tr([
                        html.Td("Volatility (σ):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_sigma_{exotic_option_type}", type="number", value=0.2, step=0.01,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                ])
            ],
            bordered=True,
            dark=True,
            hover=True,
            responsive=True,
            striped=True,
            style={"margin-top": "20px"}
        )
    ])

def generate_greek_table(exotic_option_type):
    return html.Div([
        html.H4(f"{exotic_option_type.capitalize()} Option Greeks (Call)", className="text-center text-light"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Greek", className="text-light", style={"width": "50%"}),
                html.Th("Call", className="text-light", style={"width": "25%"}),
                html.Th("Put", className="text-light", style={"width": "25%"}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Delta", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_delta_{exotic_option_type}_call", className="text-light")),
                    html.Td(html.Div(id=f"value_delta_{exotic_option_type}_put", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Gamma", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_gamma_{exotic_option_type}_call", className="text-light")),
                    html.Td(html.Div(id=f"value_gamma_{exotic_option_type}_put", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Theta", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_theta_{exotic_option_type}_call", className="text-light")),
                    html.Td(html.Div(id=f"value_theta_{exotic_option_type}_put", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Vega", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_vega_{exotic_option_type}_call", className="text-light")),
                    html.Td(html.Div(id=f"value_vega_{exotic_option_type}_put", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Rho", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_rho_{exotic_option_type}_call", className="text-light")),
                    html.Td(html.Div(id=f"value_rho_{exotic_option_type}_put", className="text-light")),
                ]),
            ])
        ],
            bordered=True,
            dark=True,
            hover=True,
            responsive=True,
            striped=True,
            style={"margin-top": "20px"}
        )
    ])

# Option pricing table
def generate_option_prices_table(exotic_option_type):
    return html.Div([
        html.H4(f"{exotic_option_type.capitalize()} Option Prices", className="text-center text-light"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Option", className="text-light", style={"width": "70%"}),  # Option column
                html.Th("Price", className="text-light", style={"width": "30%"}),  # Price column
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Call", className="text-light", style={"whiteSpace": "nowrap"}),  # Call row
                    html.Td(html.Div(id=f"price_call_{exotic_option_type}", className="text-light")),  # Placeholder for Call price
                ]),
                html.Tr([
                    html.Td("Put", className="text-light", style={"whiteSpace": "nowrap"}),  # Put row
                    html.Td(html.Div(id=f"price_put_{exotic_option_type}", className="text-light")),  # Placeholder for Put price
                ]),
            ])
        ],
            bordered=True,
            dark=True,
            hover=True,
            responsive=True,
            striped=True,
            style={"margin-top": "20px"}
        )
    ])

def generate_greek_div(greek, exotic_option_type):
    """
    Generates a Div for plotting a Greek, with controls and placeholders for various plots.

    Parameters:
        greek (str): The Greek to plot (e.g., 'delta', 'vega', 'rho').
        exotic_option_type (str): The type of exotic option (e.g., 'asian', 'lookback').

    Returns:
        html.Div: A Div containing the Greek's plots, controls, and placeholders.
    """
    # Button IDs
    button_id_vs_stock = f"button_compute_{greek}_vs_stock_price_{exotic_option_type}"
    button_id_vs_strike = f"button_compute_{greek}_vs_strike_price_{exotic_option_type}"
    button_id_vs_ttm = f"button_compute_{greek}_vs_ttm_{exotic_option_type}"

    # Store IDs
    store_id_vs_stock = f"store_plot_{greek}_vs_stock_price_{exotic_option_type}"
    store_id_vs_strike = f"store_plot_{greek}_vs_strike_price_{exotic_option_type}"
    store_id_vs_ttm = f"store_plot_{greek}_vs_ttm_{exotic_option_type}"

    return html.Div([
        # Title
        html.H5(f'{exotic_option_type.capitalize()} {greek.capitalize()}', className="text-center"),

        # Greek Values (e.g., rhoCall, rhoPut)
        html.Div([
            html.Label(f"{greek.capitalize()} Call {exotic_option_type.capitalize()}:"),
            html.P(id=f"{greek}_call_{exotic_option_type}"),
            html.Label(f"{greek.capitalize()} Put {exotic_option_type.capitalize()}:"),
            html.P(id=f"{greek}_put_{exotic_option_type}")
        ]),

        # Store for caching
        dcc.Store(id=store_id_vs_stock),
        dcc.Store(id=store_id_vs_strike),
        dcc.Store(id=store_id_vs_ttm),

        # Greek Plots
        dbc.Row([
            # Column 1: Greek vs Stock Price
            dbc.Col([
                dcc.Graph(id=f"plot_{greek}_vs_stock_price_{exotic_option_type}", style={"height": "500px"}),
                html.Div(html.Button(
                    f"Compute {greek.capitalize()} vs Stock Price", 
                    id=button_id_vs_stock, 
                    n_clicks=0, 
                    className="btn btn-primary mt-3"
                ), style={"textAlign": "center"})
            ], width=4),

            # Column 2: Greek vs Strike Price
            dbc.Col([
                dcc.Graph(id=f"plot_{greek}_vs_strike_price_{exotic_option_type}", style={"height": "500px"}),
                html.Div(html.Button(
                    f"Compute {greek.capitalize()} vs Strike Price", 
                    id=button_id_vs_strike, 
                    n_clicks=0, 
                    className="btn btn-primary mt-3"
                ), style={"textAlign": "center"})
            ], width=4),

            # Column 3: Placeholder for additional plots (e.g., Greek vs TTM)
            dbc.Col([
                dcc.Graph(id=f"plot_{greek}_vs_ttm_{exotic_option_type}", style={"height": "500px"}),
                html.Div(html.Button(
                    f"Compute {greek.capitalize()} vs TTM", 
                    id=button_id_vs_ttm, 
                    n_clicks=0, 
                    className="btn btn-primary mt-3"
                ), style={"textAlign": "center"})
            ], width=4),
        ])
    ], style={'margin-bottom': '20px'})


def generate_greeks_accordion(exotic_option_type):
    """
    Generates an accordion layout for all Greeks for a given exotic option type.

    Parameters:
        exotic_option_type (str): The type of exotic option (e.g., 'asian', 'lookback').

    Returns:
        dbc.Accordion: An accordion containing all Greek plots for the exotic option type.
    """
    return dbc.Accordion([
        dbc.AccordionItem(generate_greek_div("delta", exotic_option_type), title="Delta"),
        dbc.AccordionItem(generate_greek_div("gamma", exotic_option_type), title="Gamma"),
        dbc.AccordionItem(generate_greek_div("vega", exotic_option_type), title="Vega"),
        dbc.AccordionItem(generate_greek_div("theta", exotic_option_type), title="Theta"),
        dbc.AccordionItem(generate_greek_div("rho", exotic_option_type), title="Rho"),
    ], start_collapsed=True, always_open=True)


def generate_main_div(exotic_option_type):
    """
    Generates the main Div for a given exotic option type.

    Parameters:
        exotic_option_type (str): The type of exotic option (e.g., 'asian', 'lookback', 'barrier').

    Returns:
        html.Div: A Div containing the full layout for the specified exotic option type.
    """
    # Barrier-specific table (only for barrier options)
    barrier_table = None
    if exotic_option_type == "barrier":
        barrier_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Parameter", className="text-light", style={"width": "50%"}),
                    html.Th("Value", className="text-light", style={"width": "50%"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("Barrier Level (Call, B_call):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_B_call_{exotic_option_type}", type="number", value=90, step=1,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                    html.Tr([
                        html.Td("Barrier Level (Put, B_put):", className="text-light", style={"whiteSpace": "nowrap"}),
                        html.Td(dcc.Input(id=f"input_B_put_{exotic_option_type}", type="number", value=110, step=1,
                                          style={"height": "30px", 'width': '60px'})),
                    ]),
                ])
            ],
            bordered=True,
            dark=True,
            hover=True,
            responsive=True,
            striped=True,
            style={"margin-top": "20px"}
        )

    return html.Div([
        html.H4(f"{exotic_option_type.capitalize()} Options", style={'margin': '20px'}),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id=f"plot_first_n_simulations_{exotic_option_type}", style={"height": "700px"})), width=8),
            dbc.Col([
                dbc.Row([
                    dbc.Col(generate_input_table(exotic_option_type)),
                    dbc.Col(generate_greek_table(exotic_option_type))
                ]),
                # Add the barrier-specific table if the option type is "barrier"
                dbc.Row(barrier_table) if barrier_table else None,
                dbc.Row(generate_option_prices_table(exotic_option_type)),
                dbc.Row(html.Div(html.Button("Update Parameters", 
                                            id=f"button_update_params_{exotic_option_type}", 
                                            n_clicks=0, 
                                            className="btn btn-primary mt-3"),
                                 style={"textAlign": "center"},))
            ])
        ]),
        dbc.Row([generate_greeks_accordion(exotic_option_type)])
    ], id=f"div_{exotic_option_type}")



models_info = [
    {
        "title": "Geometric Brownian Motion (GBM)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Constant volatility."),
                html.Li("Continuous price paths (no jumps)."),
                html.Li("Log-normal distribution of stock prices.")
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + σ S_t dW_t"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("μ: Drift rate."),
                html.Li("σ: Volatility."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Vanilla Options: European and American options."),
                html.Li("Simple Exotics: Barrier options, Asian options (with adjustments)."),
            ]),
        ]),
    },
    {
        "title": "Jump-Diffusion Model (Merton Model)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Combines continuous GBM with random jumps."),
                html.Li("Jumps follow a Poisson process."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + σ S_t dW_t + J_t dN_t"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("μ, σ: Drift and volatility of GBM."),
                html.Li("λ: Jump intensity."),
                html.Li("μ_J, σ_J: Mean and standard deviation of jump sizes."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics Sensitive to Jumps: Digital options, Barrier options."),
            ]),
        ]),
    },
    {
        "title": "Stochastic Volatility Models (e.g., Heston Model)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Volatility is stochastic and follows its own process."),
                html.Li("Volatility is mean-reverting."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + √v_t S_t dW_t^1"),
            html.P("dv_t = κ (θ - v_t) dt + ξ √v_t dW_t^2"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("μ: Drift rate."),
                html.Li("κ: Speed of mean reversion."),
                html.Li("θ: Long-term mean variance."),
                html.Li("ξ: Volatility of volatility."),
                html.Li("ρ: Correlation between W_t^1 and W_t^2."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Volatility-Sensitive Exotics: Barrier options, Asian options, Lookback options, Cliquet options, Volatility swaps."),
            ]),
        ]),
    }, 
    {
        "title": "Local Volatility Model (Dupire Model)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Volatility is a deterministic function of stock price and time."),
                html.Li("Calibrated to market prices."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + σ(S_t, t) S_t dW_t"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("σ(S_t, t): Local volatility surface."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics Requiring Exact Calibration: Barrier options, Asian options, Lookback options, Autocallables."),
            ]),
        ]),
    },
    {
        "title": "Stochastic Volatility Jump-Diffusion Model (Bates Model)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Combines stochastic volatility with jumps."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + √v_t S_t dW_t^1 + J_t dN_t"),
            html.P("dv_t = κ (θ - v_t) dt + ξ √v_t dW_t^2"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("Same as Heston and Merton models."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics Sensitive to Both Volatility and Jumps: Barrier options, Digital options, Cliquet options."),
            ]),
        ]),
    },
    {
        "title": "Variance Gamma Model (VG)",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Stock returns follow a variance gamma process (pure jumps)."),
                html.Li("Captures skewness and kurtosis."),
            ]),
            html.H5("Formula:"),
            html.P("S_t = S_0 exp((r - q + ω) * t + X_t)"),
            html.P("ω = ln(1 - θ * 𝜈 - 0.5 * θ^2 * 𝜈) / 𝜈"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("σ: Volatility of the gamma process"),
                html.Li("θ: Skewness parameter"),
                html.Li("𝜈: Kurtosis parameter (variance rate of the gamma process)"),
                html.Li("ω: Drift correction term"),
                html.Li("r: Risk-free rate"),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics Sensitive to Heavy Tails: Digital options, Barrier options"),
            ]),
        ]),
    },
    {
        "title": "Constant Elasticity of Variance (CEV) Model",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Volatility depends on the stock price level."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ S_t dt + σ S_t^γ dW_t"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("μ: Drift rate."),
                html.Li("σ: Volatility coefficient."),
                html.Li("γ: Elasticity parameter."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics with Price-Dependent Volatility: Barrier options, Asian options."),
            ]),
        ]),
    },
    {
        "title": "Regime-Switching Models",
        "content": html.Div([
            html.H5("Assumptions:"),
            html.Ul([
                html.Li("Market switches between different regimes (e.g., high/low volatility)."),
            ]),
            html.H5("Formula:"),
            html.P("dS_t = μ_{Z_t} S_t dt + σ_{Z_t} S_t dW_t"),
            html.H5("Parameters:"),
            html.Ul([
                html.Li("Transition probabilities between regimes."),
                html.Li("μ and σ for each regime."),
            ]),
            html.H5("Best For:"),
            html.Ul([
                html.Li("Exotics in Regime-Shifting Markets: Cliquet options, Autocallables."),
            ]),
        ]),
    },
]


models_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(model["content"], title=model["title"]) for model in models_info
    ],
    start_collapsed=True,
    always_open=True,
)

models_table = dbc.Table(
    [
        html.Thead(
            html.Tr([
                html.Th("Model"),
                html.Th("Best For"),
                html.Th("Exotics Commonly Priced"),
            ])
        ),
        html.Tbody([
            html.Tr([html.Td("GBM"), html.Td("Simple exotics with constant volatility."), html.Td("Barrier, Asian.")]),
            html.Tr([html.Td("Jump-Diffusion (Merton)"), html.Td("Exotics sensitive to jumps."), html.Td("Digital, Barrier.")]),
            html.Tr([html.Td("Heston (Stochastic Volatility)"), html.Td("Most exotics, especially volatility-sensitive ones."), html.Td("Barrier, Asian, Lookback, Cliquet, Volatility Swaps.")]),
            html.Tr([html.Td("Local Volatility (Dupire)"), html.Td("Exotics requiring exact calibration to market prices."), html.Td("Barrier, Asian, Lookback, Autocallables.")]),
            html.Tr([html.Td("Bates (SVJD)"), html.Td("Exotics sensitive to both stochastic volatility and jumps."), html.Td("Barrier, Digital, Cliquet.")]),
            html.Tr([html.Td("Variance Gamma (VG)"), html.Td("Exotics sensitive to heavy-tailed returns."), html.Td("Digital, Barrier.")]),
            html.Tr([html.Td("CEV"), html.Td("Exotics with price-dependent volatility."), html.Td("Barrier, Asian.")]),
            html.Tr([html.Td("Regime-Switching"), html.Td("Exotics in markets with clear regime shifts."), html.Td("Cliquet, Autocallables.")]),
        ])
    ],
    bordered=True,
    striped=True,
    hover=True,
)

black_scholes_div = dbc.Card([
    dbc.CardHeader(html.H3("Black-Scholes Model (BS)")),

    dbc.CardBody([

    html.P("The Black-Scholes-Merton (BSM) model provides a closed-form solution for European options."),
    
    html.H4("Assumptions:"),
    html.Ul([
        html.Li("The stock follows Geometric Brownian Motion (GBM)."),
        html.Li("No arbitrage opportunities."),
        html.Li("The option is European-style (only exercisable at maturity)."),
        html.Li("The volatility (σ) is constant."),
        html.Li("The risk-free interest rate (r) is constant."),
        html.Li("The market is frictionless (no transaction costs or dividends)."),
    ]),
    
    html.H4("Formula (Call Option Price):"),
    html.P("C = S_0 N(d_1) - K e^{-rT} N(d_2)"),
    html.P("d_1 = (ln(S_0 / K) + (r + 0.5σ^2)T) / (σ sqrt(T))"),
    html.P("d_2 = d_1 - σ sqrt(T)"),
    
    html.H4("Parameters:"),
    html.Ul([
        html.Li("C = Call option price."),
        html.Li("S_0 = Current stock price."),
        html.Li("K = Strike price."),
        html.Li("T = Time to maturity."),
        html.Li("r = Risk-free interest rate."),
        html.Li("σ = Volatility."),
        html.Li("N(d) = Cumulative standard normal distribution."),
    ]),
    
    html.H4("Best For:"),
    html.Ul([
        html.Li("European options."),
        html.Li("Quick pricing (closed-form solution)."),
    ]),
    ]),

], className="mb-3")

binomial_tree_div = dbc.Card([
    dbc.CardHeader(html.H3("Binomial Tree Model")),

    dbc.CardBody([

    html.P("The binomial model uses a discrete-time approach to model stock price movements."),
    
    html.H4("Assumptions:"),
    html.Ul([
        html.Li("The stock price follows a multiplicative binomial process."),
        html.Li("Each time step, the stock can move up (u) or down (d)."),
        html.Li("The option can be exercised at any time (American-style options)."),
    ]),
    
    html.H4("Formula (Stock Price Evolution):"),
    html.P("S_i = S_0 u^i d^{(N-i)}"),
    
    html.H4("Risk-neutral probability:"),
    html.P("p = (e^{r Δt} - d) / (u - d)"),
    
    html.H4("Parameters:"),
    html.Ul([
        html.Li("u = Up factor (e^{σ sqrt(Δt)})."),
        html.Li("d = Down factor (1/u)."),
        html.Li("r = Risk-free rate."),
        html.Li("Δt = Time step."),
        html.Li("N = Number of steps."),
    ]),
    
    html.H4("Best For:"),
    html.Ul([
        html.Li("American-style options."),
        html.Li("Barrier options."),
    ]),
    ]),
], className="mb-3")


monte_carlo_simulation_div = dbc.Card([
    dbc.CardHeader(html.H3("Monte Carlo Simulation")),

    dbc.CardBody([

        dbc.Row([

    html.P(
        "Monte Carlo simulations are widely used for option pricing when no closed-form solution exists. "
        "They rely on random sampling to estimate the expected payoff of an option under a risk-neutral measure. "
        "This method is particularly useful for pricing complex derivatives, multi-asset options, and path-dependent options "
        "such as Asian or Lookback options."
    ),
    
    html.H4("Best For:"),
    html.Ul([
        html.Li("Path-dependent exotics."),
        html.Li("Multi-asset derivatives."),
    ]),
    ]), 
        dbc.Row([
        dbc.Col(models_accordion, width=4),
        dbc.Col(models_table, width=8)
    ]), 
    ])
], className="mb-3")

finite_difference_methods_div = dbc.Card([
    dbc.CardHeader(html.H3("Finite Difference Methods (FDM)")),

    dbc.CardBody([

    html.P(
        "Finite Difference Methods (FDM) are numerical techniques used to solve the Black-Scholes partial differential equation (PDE). "
        "Instead of relying on a closed-form solution, FDM discretizes the PDE using a grid of stock prices and time steps, "
        "allowing for the pricing of options, including those with early exercise features."
    ),
    
    html.H4("Formula (Black-Scholes PDE):"),
    html.P("∂V/∂t + 0.5 σ² S² ∂²V/∂S² + r S ∂V/∂S - rV = 0"),
    
    html.H4("Best For:"),
    html.Ul([
        html.Li("Complex options (barriers, lookbacks)."),
        html.Li("American options."),
    ]),
    ]),
], className="mb-3")


option_pricing_models_table = dbc.Card([

    dbc.CardHeader(html.H3("Summary")),

    dbc.CardBody([
        
    dbc.Table(
    [
        html.Thead(
            html.Tr([
                html.Th("Model"),
                html.Th("Best For"),
                html.Th("Key Features"),
            ])
        ),
        html.Tbody([
            html.Tr([html.Td("Black-Scholes (BSM)"), html.Td("European options."), html.Td("Closed-form solution, constant volatility.")]),
            html.Tr([html.Td("Binomial Tree"), html.Td("American & exotic options."), html.Td("Flexible but computationally expensive.")]),
            html.Tr([html.Td("Trinomial Tree"), html.Td("American & barrier options."), html.Td("More accurate than binomial tree.")]),
            html.Tr([html.Td("Finite Difference (FDM)"), html.Td("PDE-based pricing for complex options."), html.Td("Stable but requires numerical methods.")]),
            html.Tr([html.Td("Monte Carlo Simulation"), html.Td("Exotic options, multi-asset derivatives."), html.Td("Flexible but slow.")]),
            html.Tr([html.Td("Barone-Adesi & Whaley"), html.Td("American options."), html.Td("Faster than binomial trees.")]),
            html.Tr([html.Td("GARCH Model"), html.Td("Options with time-varying volatility."), html.Td("More realistic volatility modeling.")]),
            html.Tr([html.Td("Jump-Diffusion (Merton/Kou)"), html.Td("Options sensitive to price jumps."), html.Td("Models large price movements.")]),
        ])
    ],
    bordered=True,
    striped=True,
    hover=True, 
)
    ])
])


div_models = html.Div(
    [dbc.Row([monte_carlo_simulation_div]), 
    dbc.Row([black_scholes_div]), 
    dbc.Row([binomial_tree_div]), 
    dbc.Row([finite_difference_methods_div]), 
    dbc.Row([option_pricing_models_table])], 
    id = 'div_models', style = {'width': '70%', 'margin': 'auto'}
)

# TODO: cahnge a bit the style so it looks better
    # add in monte carlo a description of accordion and table: model for stock price, used in monte carlo