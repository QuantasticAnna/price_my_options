
from dash import Dash, dcc, html, callback, Input, Output, State
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import pandas as pd
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask_caching import Cache
import numpy as np
from datetime import datetime 
from pricer.monte_carlo import monte_carlo_simulations, plotter_first_n_simulations
from pricer.asian import plotter_asian, pricer_asian
from pricer.lookback import plotter_lookback
from custom_templates import cyborg_template
from dash import dash_table
from greeks.delta import compute_delta, delta_vs_stock_price, plot_delta_vs_stock_price, delta_vs_strike_price_for_multiple_volatility, delta_vs_strike_price_for_multiple_ttm
from constants import VOLATILITY_ARRAY, TTM_ARRAY, H, K_RANGE, S0_RANGE, EXOTIC_TYPE

# All stuff relative to Z recomputation are signaled by #ZRECOMPUTE (to do control F easily)
# Uncomment when we start working on giving the user the option to recompute Z, for now, its too slow, 
# # Global variable to store Z
# Z_global = None

# Load precomputed Z
Z_precomputed = joblib.load("Z_precomputed.joblib")


# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY])

server = app.server

# Configure Flask Cache
cache = Cache(server, config={
    "CACHE_TYPE": "SimpleCache",  # Use a simple in-memory cache
    "CACHE_DEFAULT_TIMEOUT": 300  # Cache timeout in seconds
})

app.title = "Price My Options"

menu_bar = html.Div([dmc.SegmentedControl(id = "menu_bar",
                                            value = "asian",
                                            fullWidth = True,
                                            data = [
                                                {"value": "asian", "label": "Asian"},
                                                {"value": "lookback", "label": "Lookback"},
                                                {"value": "value3", "label": "Label 3"},
                                            ])])

#ZRECOMPUTE
# button_generate_z = html.Button("Generate Simulations", id="button_generate_z", n_clicks=0)
# store_z = dcc.Store(id="store_z")
# z_status = html.Div(id="status", style={"margin-top": "20px"})

input_option_params = dbc.Container([html.H4("Option Parameters", className="text-center text-light"),
                                     dbc.Table(
            [html.Thead(html.Tr([html.Th("Parameter", className="text-light", style={"width": "70%"}),
                                 html.Th("Value", className="text-light", style={"width": "30%"}),
                                 ])),
            html.Tbody([html.Tr([html.Td("Initial Stock Price (S0):", className="text-light", style={"whiteSpace": "nowrap"}),
                                html.Td(dcc.Input(id="input_S0", type="number", value=100, step=1, style={"height": "30px",
                                                                                                           'width': '60px'})),]),
                        html.Tr([html.Td("Strike Price (K):", className="text-light", style={"whiteSpace": "nowrap"}),
                                 html.Td(dcc.Input(id="input_K", type="number", value=100, step=1, style={"height": "30px",
                                                                                                         'width': '60px'})),]),
                        html.Tr([html.Td("Time to Maturity (T):", className="text-light", style={"whiteSpace": "nowrap"}),
                                html.Td(dcc.Input(id="input_T", type="number", value=1, step=0.1, style={"height": "30px",
                                                                                                         'width': '60px'})),]),
                        html.Tr([html.Td("Risk-Free Rate (r):", className="text-light", style={"whiteSpace": "nowrap"}),
                                html.Td(dcc.Input(id="input_r", type="number", value=0.05, step=0.01, style={"height": "30px",
                                                                                                            'width': '60px'})),]),
                        html.Tr([html.Td("Volatility (σ):", className="text-light", style={"whiteSpace": "nowrap"}),
                                html.Td(dcc.Input(id="input_sigma", type="number", value=0.2, step=0.01, style={"height": "30px",
                                                                                                                'width': '60px'})),]),
                    ]),],
                        bordered=True,
                        dark=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                        style={"margin-top": "20px"},),],
                fluid=True,
                style={"margin-bottom": "20px"},)

div_update_option_parameters = html.Div(
            html.Button("Update Parameters", id="button_update_params", n_clicks=0, className="btn btn-primary mt-3"),
            style={"textAlign": "center"},
        )

option_pricing_table = html.Div([html.H4("Option Prices", className="text-center text-light"),
                                 dbc.Table([html.Thead(html.Tr([html.Th("Option", className="text-light", style={"width": "70%"}),  # Option column
                                                                html.Th("Price", className="text-light", style={"width": "30%"}),   # Price column
                                                                ])),
                                            html.Tbody([html.Tr([html.Td("Call", className="text-light", style={"whiteSpace": "nowrap"}),  # Call row
                                                                 html.Td(html.Div(id="price_call", className="text-light")),               # Placeholder for Call price
                                                                 ]),
                                                        html.Tr([html.Td("Put", className="text-light", style={"whiteSpace": "nowrap"}),   # Put row
                                                                  html.Td(html.Div(id="price_put", className="text-light")),                # Placeholder for Put price
                                                                  ]),
                                                                  ]),],
                                bordered=True,
                                dark=True,
                                hover=True,
                                responsive=True,
                                striped=True,
                                style={"margin-top": "20px"},
                            )])

# plot_first_n_simulations = dcc.Graph(id="plot_first_n_simulations", style={"height": "600px"})

plot_first_n_simulations_asian = dcc.Graph(id="plot_first_n_simulations_asian", style={"height": "700px"})

plot_first_n_simulations_lookback = dcc.Graph(id="plot_first_n_simulations_lookback", style={"height": "500px"})

# greeks_asian = html.Div([delta_call_asian, 
#                          delta_put_asian, 
#                          delta_call_vs_stock_price_asian,
#                         #  delta_put_vs_stock_price_asian,  # Necessary? Useful?
#                          delta_call_vs_strike_price_asian, 
#                          #  delta_put_vs_strike_price_asian,  # Necessary? Useful?
#                          slider_volatilities_asian,
#                          delta_vs_strike_price_for_multiple_volatility_asian, # again, call and put distinction? 
#                          slider_ttm_asian, 
#                          delta_vs_strike_price_for_multiple_TTM_asian,
#                          ])


# TODO: choose one style between the two tables (options param, greeks table) and apply it to the other tbale
greeks_table_asian = html.Div([html.H4("Option Greeks (Call)", className="text-center text-light"),
                               dbc.Table([html.Thead(html.Tr([html.Th("Greek", className="text-light", style={"width": "50%"}),
                                                              html.Th("Value", className="text-light", style={"width": "50%"}),])),
                                        html.Tbody([html.Tr([html.Td("Delta", className="text-light", style={"whiteSpace": "nowrap"}),
                                                             html.Td(html.Div(id="value_delta", className="text-light")),]),
                                                    html.Tr([html.Td("Gamma", className="text-light", style={"whiteSpace": "nowrap"}),
                                                             html.Td(html.Div(id="value_gamma", className="text-light")),]),
                                                    html.Tr([html.Td("Theta", className="text-light", style={"whiteSpace": "nowrap"}),
                                                             html.Td(html.Div(id="value_theta", className="text-light")),]),
                                                    html.Tr([html.Td("Vega", className="text-light", style={"whiteSpace": "nowrap"}),
                                                             html.Td(html.Div(id="value_vega", className="text-light")),]),
                                                    html.Tr([html.Td("Rho", className="text-light", style={"whiteSpace": "nowrap"}),
                                                             html.Td(html.Div(id="value_rho", className="text-light")),]),]),],
                                bordered=True,
                                dark=True,
                                hover=True,
                                responsive=True,
                                striped=True,
                                style={"margin-top": "20px"},
                            )])

delta_asian_values = html.Div([html.Label("Delta Call Asian:"), 
                               html.P(id = 'delta-call-asian'),
                               html.Label("Delta Put Asian:"), 
                               html.P(id = 'delta-put-asian')])

figure_delta_call_vs_stock_price_asian = dcc.Graph(id="plot_delta_call_vs_stock_price_asian", style={"height": "500px"})

slider_ttm_asian = html.Div([html.Label("Time to Maturity (Years):"),
                                        dcc.Slider(
                                            id="slider-ttm-asian",
                                            min=TTM_ARRAY.min(),
                                            max=TTM_ARRAY.max(),
                                            step=None,
                                            marks={value: f"{value:.4f}" for value in TTM_ARRAY},  # Use values as keys
                                            value=TTM_ARRAY.max()  # Default index corresponding to 1 year
                                        )])

slider_volatilities_asian = html.Div([html.Label("Volatility (%):"),
                                    dcc.Slider(
                                        id="slider-volatility-asian",
                                        min=VOLATILITY_ARRAY.min(), #0.1,
                                        max=VOLATILITY_ARRAY.max(), #0.6,
                                        step=None,
                                        marks = {value: f"{value:.1f}" for value in VOLATILITY_ARRAY},
                                        # marks={i: f"{VOLATILITY_ARRAY[i]}" for i in range(len(VOLATILITY_ARRAY))},
                                        value=VOLATILITY_ARRAY.max(), #0.2,  # Default value
                                    )])

# div_delta_asian_card = dbc.Card([dbc.CardHeader(html.H5('Asian Delta', className="text-center")),
#                             dbc.CardBody([delta_asian_values,
#                                     dbc.Row([dbc.Col([figure_delta_call_vs_stock_price_asian], width=4),
#                                              dbc.Col([dcc.Store(id="delta-vs-strike-different-vol-store"),
#                                                       dcc.Graph(id = 'plot-delta-vs-strike-different-vol'),
#                                                       slider_volatilities_asian,], width=4),  #placeholder figures
#                                              dbc.Col([slider_ttm_asian,
#                                                       dcc.Graph(figure=go.Figure(), id = 'plot-delta-vs-strike-different-ttm')], width=4)]),]), #placeholder figures
#                                     ], style = {'margin-bottom': '20px'})

div_delta_asian = html.Div([html.H5('Asian Delta', className="text-center"),
                            delta_asian_values,
                            dbc.Row([dbc.Col([figure_delta_call_vs_stock_price_asian], width=4),
                                        dbc.Col([dcc.Store(id="delta-vs-strike-different-vol-store"),
                                                dcc.Graph(id = 'plot-delta-vs-strike-different-vol'),
                                                slider_volatilities_asian], width=4),  #placeholder figures
                                        dbc.Col([dcc.Store(id="delta-vs-strike-different-ttm-store"),
                                                 dcc.Graph(id = 'plot-delta-vs-strike-different-ttm'),
                                                 slider_ttm_asian,], width=4)]),]#placeholder figures
                                    , style = {'margin-bottom': '20px'})


div_gamma_asian = dbc.Card([dbc.CardHeader(html.H5('Asian Gamma', className="text-center")),
                            dbc.CardBody([html.P('Values'),
                                            html.P('Plot1'),
                                            html.P('Plot2')],),
                                    ], style = {'margin-bottom': '20px'})

div_vega_asian = dbc.Card([dbc.CardHeader(html.H5('Asian Vega', className="text-center")),
                            dbc.CardBody([html.P('Values'),
                                            html.P('Plot1'),
                                            html.P('Plot2')],),
                                    ], style = {'margin-bottom': '20px'})

div_theta_asian = dbc.Card([dbc.CardHeader(html.H5('Asian Theta', className="text-center")),
                            dbc.CardBody([html.P('Values'),
                                            html.P('Plot1'),
                                            html.P('Plot2')],),
                                    ], style = {'margin-bottom': '20px'})

div_rho_asian = dbc.Card([dbc.CardHeader(html.H5('Asian Rho', className="text-center")),
                            dbc.CardBody([html.P('Values'),
                                            html.P('Plot1'),
                                            html.P('Plot2')],),
                                    ], style = {'margin-bottom': '20px'})

div_greeks_asian = dbc.Container([html.H4("Greeks for Asian Options", className="text-center"),
                                    dbc.Row(dbc.Col(div_delta_asian, width=11), justify="center"),
                                    dbc.Row(dbc.Col(div_gamma_asian, width=11), justify="center"),
                                    dbc.Row(dbc.Col(div_vega_asian, width=11), justify="center"),
                                    dbc.Row(dbc.Col(div_theta_asian, width=11), justify="center"),
                                    dbc.Row(dbc.Col(div_rho_asian, width=11), justify="center"),],
                                    style={"padding-top": "20px",
                                            "padding-left": "20px",
                                            "padding-right": "20px"}
                                        
                                )

div_greeks_asian_accordion = dbc.Container([html.H4("Greeks for Asian Options", className="text-center"),
                                            dbc.Accordion([dbc.AccordionItem(div_delta_asian,
                                                                             title="Delta"),
                                                            dbc.AccordionItem(div_gamma_asian,
                                                                             title="Gamma"),
                                                            dbc.AccordionItem(div_vega_asian,
                                                                              title="Vega"),
                                                            dbc.AccordionItem(div_theta_asian,
                                                                              title="Theta"),
                                                            dbc.AccordionItem(div_rho_asian,
                                                                              title="Rho"),],
                                                        # flush=True,  # Optional: Removes borders between items for a cleaner look
                                                        start_collapsed=True,
                                                        always_open=True,  # Optional: Allow multiple sections to be open at once
                                                            )],
                                            style={
                                                "padding-top": "20px",
                                                "padding-left": "20px",
                                                "padding-right": "20px"
                                            })


div_asian = html.Div([html.H4('Asian', style = {'margin' : '20px'}),
                      dbc.Row([dbc.Col(plot_first_n_simulations_asian, width=8),
                               dbc.Col([dbc.Row([dbc.Col(input_option_params),
                                                dbc.Col(greeks_table_asian),]),
                                        dbc.Row(option_pricing_table),
                                        dbc.Row(div_update_option_parameters)])
                               ]),
                      dbc.Row([div_greeks_asian_accordion,
                               # div_greeks_asian
                               ])],
                id = 'div_asian')

div_lookback = html.Div([html.H4('Lookback', style = {'margin' : '20px'}),
                         plot_first_n_simulations_lookback,],
                id = 'div_lookback')  
                       
div3 = html.Div([html.H4('Div 3', style = {'margin' : '20px'})],
                id = 'div3')

app.layout = html.Div([html.H1('Price My Options'),
                        # input_option_params, # for now we focus on asian, so we put this is div asian
                       # plot_first_n_simulations,
                       menu_bar, 
                       div_asian,
                       div_lookback,
                       div3,
                       # html.Div([button_generate_z, store_z, z_status]) #ZRECOMPUTE
                       ], style = {'margin' : '20px'})

@callback(
    [Output('div_asian', 'hidden'),
     Output('div_lookback', 'hidden'), 
     Output('div3', 'hidden')],
    [Input('menu_bar', 'value')]
)
def show_hidden_div(input_value):
    show_div_asian = True
    show_div_lookback = True
    show_div3 = True

    if input_value == 'asian':
        show_div_asian = False
    elif input_value == 'lookback':
        show_div_lookback = False
    elif input_value == 'value3':
        show_div3 = False

    return(show_div_asian, show_div_lookback, show_div3)

#ZRECOMPUTE
# @app.callback(
#     Output("status", "children"),  # Update status message
#     Output("store_z", "data"),  # Store Z in dcc.Store
#     Input("button_generate_z", "n_clicks")  # Trigger on button click
# )
# def generate_simulations(n_clicks):
#     global Z_global
#     if n_clicks > 0:
#         # Generate Z and store it globally
#         n_simulations = 100000  # Number of simulations
#         Z = np.random.standard_normal((n_simulations, 252))  # 252 days (e.g., trading days in a year)
#         Z_global = Z  # Store in global variable
#         return f"Simulations generated! at {datetime.now()}", Z.tolist()  # Store Z in dcc.Store as JSON serializable format
#     return "Click the button to generate simulations.", None


@app.callback(
    # Output("plot_first_n_simulations", "figure"),
    Output("plot_first_n_simulations_asian", "figure"),
    Output("plot_first_n_simulations_lookback", "figure"),
    Input("button_update_params", "n_clicks"),
    State("input_S0", "value"),
    State("input_K", "value"),
    State("input_T", "value"),
    State("input_r", "value"),
    State("input_sigma", "value")
)
def show_plot_first_n_simulations(n_clicks, S0, K, T, r, sigma): #! Maybe instead of adding a white line at strike, make a grey zone for the OTM (but then we need a specific graph for put, and a specific graph for call)
    if n_clicks > 0 and Z_precomputed is not None:
        Z = np.array(Z_precomputed)  # Convert Z back to NumPy array
        S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=100000)
        # fig_S = plotter_first_n_simulations(S, n_sim_to_plot=10)

        fig_asian = plotter_asian(S, n_sim_to_plot=10)
        # Add a horizontal line at strike K
        fig_asian.add_hline(
            y=K,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Strike Price (K={K})",
            annotation_position="bottom right",
        )

        # Add a horizontal line at strike K
        fig_lookback = plotter_lookback(S, n_sim_to_plot=10)
        fig_lookback.add_hline(
            y=K,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Strike Price (K={K})",
            annotation_position="bottom right",
        )
        # return fig_S, fig_asian, fig_lookback
        return fig_asian, fig_lookback
    # return go.Figure(), go.Figure(), go.Figure() # empty figures
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title=f"First Monte Carlo Simulations of Stock Prices, with avg",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template=cyborg_template
    )
    return empty_fig, empty_fig # empty figures

@app.callback(
    [Output('delta-call-asian', 'children'),
     Output('delta-put-asian', 'children')],
     Input('menu_bar', 'value'), ## at a later stage we will the value of the menu bar? for now, hardcoded for asian
     Input("button_update_params", "n_clicks"),
    State("input_S0", "value"),
    State("input_K", "value"),
    State("input_T", "value"),
    State("input_r", "value"),
    State("input_sigma", "value"))
def show_delta_values_asian(menu_bar_value, n_clicks, S0, K, T, r, sigma):
    h = H
    exotic_type = EXOTIC_TYPE # later will be value of menu bar 

    if n_clicks > 0 and Z_precomputed is not None:
        deltas = compute_delta(Z_precomputed, S0, K, T, r, sigma, h, exotic_type) # for asian, no kwargs
        return html.Div(f"{deltas['delta_call']*100:.2f}%"), html.Div(f"{(1 - deltas['delta_call'])*100:.2f}%")

    else: 
        # return empty values
        return html.Div(''), html.Div('')

@app.callback(
    [Output("value_delta", "children"),
     Output("value_gamma", "children"),
     Output("value_theta", "children"),
    Output("value_vega", "children"),
     Output("value_rho", "children")],
    [Input("input_S0", "value"),
     Input("input_K", "value"),
     Input("input_T", "value"),
     Input("input_sigma", "value"),
     Input("input_r", "value")]
)
def update_greeks(S0, K, T, sigma, r):
    # Example calculations (replace with actual formulas)
    delta = f"{S0 * 0.01:.2f}"  # Placeholder
    gamma = f"{K * 0.005:.2f}"  # Placeholder
    theta = f"{-T * 0.002:.2f}"  # Placeholder
    vega = f"{sigma * 0.1:.2f}"  # Placeholder
    rho = f"{r * 0.2:.2f}"       # Placeholder
    return delta, gamma, theta, vega, rho


@app.callback(
    [Output("price_call", "children"),
     Output("price_put", "children")],
    [Input("button_update_params", "n_clicks"),
     Input("input_S0", "value"),
     Input("input_K", "value"),
     Input("input_T", "value"),
     Input("input_sigma", "value"),
     Input("input_r", "value")],
)
def update_prices(n_clicks, S0, K, T, sigma, r):

    if n_clicks > 0 and Z_precomputed is not None:
        S = monte_carlo_simulations(Z_precomputed, S0, T, r, sigma)
        prices = pricer_asian(S, K, T, r)
        return f"{prices['price_call']:.2f}", f"{prices['price_put']:.2f}"
    
    else:
        return ' ', ' '

@app.callback(
    Output('plot_delta_call_vs_stock_price_asian', 'figure'),
     Input('menu_bar', 'value'), ## at a later stage we will the value of the menu bar? for now, hardcoded for asian
     Input("button_update_params", "n_clicks"),
    State("input_S0", "value"),
    State("input_K", "value"),
    State("input_T", "value"),
    State("input_r", "value"),
    State("input_sigma", "value"))
def show_delta_plots_vs_stock_price(menu_bar_value, n_clicks, S0, K, T, r, sigma):
    h = H
    S0_range = S0_RANGE  
    exotic_type = EXOTIC_TYPE # later will be value of menu bar 

    if n_clicks > 0 and Z_precomputed is not None:

        fig  = plot_delta_vs_stock_price(Z_precomputed, S0_range, K, T, r, sigma, h, exotic_type)# for asian, no kwargs
        return fig

    else: 
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Delta vs Stock Price",
            xaxis_title="Stock Price (S)",
            yaxis_title="Delta",
            legend=dict(
                title="Option Type",
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            template=cyborg_template
    )
        return empty_fig
    


@app.callback(
    Output("delta-vs-strike-different-vol-store", "data"),
    Input("button_update_params", "n_clicks"),
    [
        State("input_S0", "value"),
        State("input_T", "value"),
        State("input_r", "value"),
        State("slider-volatility-asian", "marks"),
    ]
)
def compute_deltas_vs_strike_for_diff_vol(n_clicks, S0, T, r, volatility_marks):
    if n_clicks > 0 and Z_precomputed is not None:
        h = H
        K_range = K_RANGE
        exotic_type = EXOTIC_TYPE 

        # Convert `volatility_marks` to an np.ndarray of numerical values
        volatilities = np.array([float(v) for v in volatility_marks.values()])

        # Compute deltas for all volatilities
        dict_delta_call, dict_delta_put = delta_vs_strike_price_for_multiple_volatility(
            Z_precomputed, S0, volatilities, K_range, T, r, h, exotic_type
        )

        # Return the data to be stored in `dcc.Store`
        return { 'K_range': K_range.tolist(),            
                "dict_delta_call": dict_delta_call,
                "dict_delta_put": dict_delta_put
        }

    # Return an empty dictionary if conditions are not met
    return {}


@app.callback(
    Output("plot-delta-vs-strike-different-vol", "figure"),
    [
        Input("slider-volatility-asian", "value"),
        Input("delta-vs-strike-different-vol-store", "data"),
    ]
)
def show_delta_plots_vs_strike_diff_vol(selected_volatility, stored_data):
    if stored_data and selected_volatility is not None:
        # Extract data from the store
        K_range = np.array(stored_data["K_range"])
        dict_delta_call = stored_data["dict_delta_call"]
        dict_delta_put = stored_data["dict_delta_put"]

        selected_volatility_str = str(selected_volatility)

        # Extract the deltas for the selected volatility
        delta_call = dict_delta_call[selected_volatility_str]
        delta_put = dict_delta_put[selected_volatility_str]

        # Plot the results
        fig = go.Figure()

        # Plot call deltas
        fig.add_trace(go.Scatter(
            x=K_range,
            y=delta_call,
            mode='lines+markers',
            name="Call Delta",
            #line=dict(color='blue')
        ))

        # Plot put deltas
        fig.add_trace(go.Scatter(
            x=K_range,
            y=delta_put,
            mode='lines+markers',
            name="Put Delta",
            #line=dict(color='red')
        ))

        # Style the plot
        fig.update_layout(
            title="Delta vs Strike Price for Different Vol",
            xaxis_title="Strike Price (K)",
            yaxis_title="Delta",
            legend=dict(
                title="Option Type",
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            template=cyborg_template
        )

        return fig

    # Return an empty figure if no data is available
    empty_fig = go.Figure()
        # Style the plot
    empty_fig.update_layout(
        title="Delta vs Strike Price for Different Vol",
        xaxis_title="Strike Price (K)",
        yaxis_title="Delta",
        legend=dict(
            title="Option Type",
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        template=cyborg_template
    )
    return empty_fig


@app.callback(
    Output("delta-vs-strike-different-ttm-store", "data"),
    Input("button_update_params", "n_clicks"),
    [
        State("input_S0", "value"),
        State("input_T", "value"),
        State("input_sigma", "value"),
        State("input_r", "value"),
        State("slider-ttm-asian", "marks"),
    ]
)
def compute_deltas_vs_strike_for_diff_ttm(n_clicks, S0, T, sigma, r, ttm_marks):
    if n_clicks > 0 and Z_precomputed is not None:
        h = H
        K_range = K_RANGE
        exotic_type = EXOTIC_TYPE 

        # Convert `volatility_marks` to an np.ndarray of numerical values
        ttm_array = np.array([float(v) for v in ttm_marks.keys()])

        # Compute deltas for all volatilities
        dict_delta_call, dict_delta_put = delta_vs_strike_price_for_multiple_ttm(
            Z_precomputed, S0, sigma, K_range, ttm_array, r, h, exotic_type
        )

        # Return the data to be stored in `dcc.Store`
        return { 'K_range': K_range.tolist(),            
                "dict_delta_call": dict_delta_call,
                "dict_delta_put": dict_delta_put
        }

    # Return an empty dictionary if conditions are not met
    return {}


@app.callback(
    Output("plot-delta-vs-strike-different-ttm", "figure"),
    [
        Input("slider-ttm-asian", "value"),
        Input("delta-vs-strike-different-ttm-store", "data"),
    ]
)
def show_delta_plots_vs_strike_diff_ttm(selected_ttm, stored_data):
    if stored_data and selected_ttm is not None:
        # Extract data from the store
        K_range = np.array(stored_data["K_range"])
        dict_delta_call = stored_data["dict_delta_call"]
        dict_delta_put = stored_data["dict_delta_put"]

        selected_ttm_str = str(selected_ttm)

        # Extract the deltas for the selected volatility
        delta_call = dict_delta_call[selected_ttm_str]
        delta_put = dict_delta_put[selected_ttm_str]

        # Plot the results
        fig = go.Figure()

        # Plot call deltas
        fig.add_trace(go.Scatter(
            x=K_range,
            y=delta_call,
            mode='lines+markers',
            name="Call Delta",
            #line=dict(color='blue')
        ))

        # Plot put deltas
        fig.add_trace(go.Scatter(
            x=K_range,
            y=delta_put,
            mode='lines+markers',
            name="Put Delta",
            #line=dict(color='red')
        ))

        # Style the plot
        fig.update_layout(
            title="Delta vs Strike Price for Different TTM",
            xaxis_title="Strike Price (K)",
            yaxis_title="Delta",
            legend=dict(
                title="Option Type",
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center"
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            template=cyborg_template
        )

        return fig

    # Return an empty figure if no data is available
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Delta vs Strike Price for Different TTM",
        xaxis_title="Strike Price (K)",
        yaxis_title="Delta",
        legend=dict(
            title="Option Type",
            orientation="h",
            y=-0.2,
            x=0.5,
            xanchor="center"
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        template=cyborg_template
    )
    return empty_fig



# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)