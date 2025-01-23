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
                html.Th("Value", className="text-light", style={"width": "50%"}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Delta", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_delta_{exotic_option_type}", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Gamma", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_gamma_{exotic_option_type}", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Theta", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_theta_{exotic_option_type}", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Vega", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_vega_{exotic_option_type}", className="text-light")),
                ]),
                html.Tr([
                    html.Td("Rho", className="text-light", style={"whiteSpace": "nowrap"}),
                    html.Td(html.Div(id=f"value_rho_{exotic_option_type}", className="text-light")),
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
def generate_option_pricing_table(exotic_option_type):
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
                dcc.Graph(id=f"plot_{greek}_vs_ttm_{exotic_option_type}", style={"height": "500px"})
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
        exotic_option_type (str): The type of exotic option (e.g., 'asian', 'lookback').

    Returns:
        html.Div: A Div containing the full layout for the specified exotic option type.
    """
    return html.Div([
        html.H4(f"{exotic_option_type.capitalize()} Options", style={'margin': '20px'}),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(id=f"plot_first_n_simulations_{exotic_option_type}", style={"height": "700px"})), width=8),
            dbc.Col([
                dbc.Row([
                    dbc.Col(generate_input_table(exotic_option_type)),
                    dbc.Col(generate_greek_table(exotic_option_type))
                ]),
                dbc.Row(generate_option_pricing_table(exotic_option_type)),
                dbc.Row(html.Div(html.Button("Update Parameters", 
                                            id=f"button_update_params_{exotic_option_type}", 
                                            n_clicks=0, 
                                            className="btn btn-primary mt-3"),
                                 style={"textAlign": "center"},))
            ])
        ]),
        dbc.Row([generate_greeks_accordion(exotic_option_type)])
    ], id=f"div_{exotic_option_type}")
