
from dash import dcc, html
import dash_bootstrap_components as dbc

stock_models_info = [
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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sigma S_t \\, dW_t$$  
                """,
                mathjax=True
            ),
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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sigma S_t \\, dW_t + J_t \\, dN_t$$  
                """,
                mathjax=True
            ),
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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sqrt{v_t} S_t \\, dW_t^1$$

                $$dv_t = \\kappa (\\theta - v_t) \\, dt + \\xi \\sqrt{v_t} \\, dW_t^2$$  
                """,
                mathjax=True
            ),

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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sigma(S_t, t) S_t \\, dW_t$$  
                """,
                mathjax=True
            ),
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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sqrt{v_t} S_t \\, dW_t^1 + J_t \\, dN_t$$

                $$dv_t = \\kappa (\\theta - v_t) \\, dt + \\xi \\sqrt{v_t} \\, dW_t^2$$  
                """,
                mathjax=True
            ),
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
            dcc.Markdown(
                """
                $$S_t = S_0 \\exp \\left( (r - q + \\omega) t + X_t \\right)$$

                $$\\omega = \\frac{\\ln(1 - \\theta \\nu - 0.5 \\theta^2 \\nu)}{\\nu}$$
                """,
                mathjax=True
            ),

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
            dcc.Markdown(
                """
                $$dS_t = \\mu S_t \\, dt + \\sigma S_t^{\\gamma} \\, dW_t$$  
                """,
                mathjax=True
            ),
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
            dcc.Markdown(
                """
                $$dS_t = \\mu_{Z_t} S_t \\, dt + \\sigma_{Z_t} S_t \\, dW_t$$  
                """,
                mathjax=True
            ),
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


stock_models_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(model["content"], title=model["title"]) for model in stock_models_info
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

black_scholes_div = html.Div([html.H3("Black-Scholes Model (BS)"),

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
    dcc.Markdown(
        """
        $$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

        $$d_1 = \\frac{\\ln(S_0 / K) + (r + 0.5 \\sigma^2)T}{\\sigma \\sqrt{T}}$$

        $$d_2 = d_1 - \\sigma \\sqrt{T}$$
        """,
        mathjax=True
    ),

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
    ])

binomial_tree_div = html.Div([html.H3("Binomial Tree Model"),

    html.P("The binomial model uses a discrete-time approach to model stock price movements."),
    
    html.H4("Assumptions:"),
    html.Ul([
        html.Li("The stock price follows a multiplicative binomial process."),
        html.Li("Each time step, the stock can move up (u) or down (d)."),
        html.Li("The option can be exercised at any time (American-style options)."),
    ]),
    
    html.H4("Formula (Stock Price Evolution):"),
    dcc.Markdown(
        """
        $$S_i = S_0 u^i d^{(N-i)}$$  
        """,
        mathjax=True
    ),
    
    html.H4("Risk-neutral probability:"),
    dcc.Markdown(
        """
        $$p = \\frac{e^{r \\Delta t} - d}{u - d}$$  
        """,
        mathjax=True
    ),
    dcc.Markdown("$$S_i = S_0 u^i d^{(N-i)}$$", mathjax=True),
    
    html.H4("Parameters:"),
    html.Ul([
        html.Li(dcc.Markdown(
        """
        $$u = \\text{Up factor} \\left( e^{\\sigma \\sqrt{\\Delta t}} \\right)$$
        """,
        mathjax=True
    )),
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
    ],)


monte_carlo_simulation_div = html.Div([html.H3("Monte Carlo Simulation"),

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
        dbc.Col(stock_models_accordion, width=4),
        dbc.Col(models_table, width=8)
    ]), 
    ])

finite_difference_methods_div = html.Div([html.H3("Finite Difference Methods (FDM)"),

    html.P(
        "Finite Difference Methods (FDM) are numerical techniques used to solve the Black-Scholes partial differential equation (PDE). "
        "Instead of relying on a closed-form solution, FDM discretizes the PDE using a grid of stock prices and time steps, "
        "allowing for the pricing of options, including those with early exercise features."
    ),
    
    html.H4("Formula (Black-Scholes PDE):"),
dcc.Markdown(
    """
    $$\\frac{\\partial V}{\\partial t} + 0.5 \\sigma^2 S^2 \\frac{\\partial^2 V}{\\partial S^2} 
    + r S \\frac{\\partial V}{\\partial S} - rV = 0$$
    """,
    mathjax=True
),

    
    html.H4("Best For:"),
    html.Ul([
        html.Li("Complex options (barriers, lookbacks)."),
        html.Li("American options."),
    ]),
    ])


option_pricing_models_table = html.Div([html.H3("Summary"),
        
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

intro_text = html.Div([html.H2("Introduction"),
                       
                       html.P("This is the introduction to my option pricer")
                       
                       ], style = {'margin': '20px'})

div_models = html.Div([intro_text,
        dbc.Accordion([
            dbc.AccordionItem(monte_carlo_simulation_div, title="Monte Carlo Simulations"),
            dbc.AccordionItem(black_scholes_div, title="Black-Scholes Model"),
            dbc.AccordionItem(binomial_tree_div, title="Binomial Tree Model"),
            dbc.AccordionItem(finite_difference_methods_div, title="Finite Difference Methods"),
            dbc.AccordionItem(option_pricing_models_table, title="Summary"),
        ], start_collapsed=True, always_open=True)
        ], id = 'div_models', style = {'width': '80%', 'margin': 'auto'})


# TODO: cahnge a bit the style so it looks better
    # add in monte carlo a description of accordion and table: model for stock price, used in monte carlo
    # maybe also center the content of the card 
    # accordion instead of cards?
    # add margin, 
    # fix problem with header of card taht dont use full width