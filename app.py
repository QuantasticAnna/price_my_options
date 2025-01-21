
from dash import Dash, dcc, html, callback, Input, Output, State
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import pandas as pd
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime 
from pricer.monte_carlo import monte_carlo_simulations, plotter_first_n_simulations
from pricer.asian import plotter_asian
from pricer.lookback import plotter_lookback
from custom_templates import cyborg_template
from greeks.delta import compute_delta, delta_vs_stock_price, plot_delta_vs_stock_price

# All stuff relative to Z recomputation are signaled by #ZRECOMPUTE (to do control F easily)
# Uncomment when we start working on giving the user the option to recompute Z, for now, its too slow, 
# # Global variable to store Z
# Z_global = None

# Load precomputed Z
Z_precomputed = joblib.load("Z_precomputed.joblib")

# Predefined volatilities and TTM arrays  # MAKE IT IN A MORE PROPER WAY , constant file? 
volatility_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Annualized volatilities
ttm_array = np.array([0.0198, 0.0992, 0.3968, 1.0])  # Time to maturity in years (5 days, 25 days, 100 days, 1 year)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY])

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


input_option_params = html.Div([
    html.H4("Specify Option Parameters"),
    html.Table([
        html.Tr([
            html.Td(html.Label("Initial Stock Price (S0):")),
            html.Td(dcc.Input(id="input_S0", type="number", value=100, step=1)),
        ]),
        html.Tr([
            html.Td(html.Label("Strike Price (K):")),
            html.Td(dcc.Input(id="input_K", type="number", value=100, step=1)),
        ]),
        html.Tr([
            html.Td(html.Label("Time to Maturity (T):")),
            html.Td(dcc.Input(id="input_T", type="number", value=1, step=0.1)),
        ]),
        html.Tr([
            html.Td(html.Label("Risk-Free Rate (r):")),
            html.Td(dcc.Input(id="input_r", type="number", value=0.05, step=0.01)),
        ]),
        html.Tr([
            html.Td(html.Label("Volatility (σ):")),
            html.Td(dcc.Input(id="input_sigma", type="number", value=0.2, step=0.01)),
        ]),
        html.Tr([
            html.Td(colSpan=2, children=[
                html.Button("Update Parameters", id="button_update_params", n_clicks=0, style={"margin-top": "20px"})
            ], style={"text-align": "center"})
        ])
    ], style={"width": "100%", "border-spacing": "10px", "border-collapse": "separate"})
], style={"margin-bottom": "20px"})

# plot_first_n_simulations = dcc.Graph(id="plot_first_n_simulations", style={"height": "600px"})

plot_first_n_simulations_asian = dcc.Graph(id="plot_first_n_simulations_asian", style={"height": "500px"})

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

delta_asian_values = html.Div([html.Label("Delta Call Asian:"), 
                               html.P(id = 'delta-call-asian'),
                               html.Label("Delta Put Asian:"), 
                               html.P(id = 'delta-put-asian')])

figure_delta_call_vs_stock_price_asian = dcc.Graph(id="plot_delta_call_vs_stock_price_asian", style={"height": "600px"})

slider_volatilities_asian = html.Div([html.Label("Time to Maturity (Years):"),
                                        dcc.Slider(
                                            id="slider-ttm",
                                            min=0,
                                            max=len(ttm_array) - 1,
                                            step=1,
                                            marks={i: f"{ttm_array[i]}y" for i in range(len(ttm_array))},
                                            value=3  # Default index corresponding to 1 year
                                        )])

slider_ttm_asian = html.Div([html.Label("Volatility (%):"),
                                    dcc.Slider(
                                        id="slider-volatility-asian",
                                        min=0.1,
                                        max=0.6,
                                        step=None,
                                        marks={i: f"{volatility_array[i]*100:.0f}%" for i in range(len(volatility_array))},
                                        value=0.2,  # Default value
                                    )])

div_delta_asian = dbc.Card([dbc.CardHeader(html.H5('Asian Delta', className="text-center")),
                            dbc.CardBody([delta_asian_values,
                                    figure_delta_call_vs_stock_price_asian,
                                    slider_volatilities_asian,
                                    slider_ttm_asian,]),
                                    ], style = {'margin-bottom': '20px'})

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

div_greeks_asian_accordion = dbc.Container(
    [
        html.H4("Greeks for Asian Options", className="text-center"),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    div_delta_asian,
                    title="Delta"
                ),
                dbc.AccordionItem(
                    div_gamma_asian,
                    title="Gamma"
                ),
                dbc.AccordionItem(
                    div_vega_asian,
                    title="Vega"
                ),
                dbc.AccordionItem(
                    div_theta_asian,
                    title="Theta"
                ),
                dbc.AccordionItem(
                    div_rho_asian,
                    title="Rho"
                ),
            ],
            # flush=True,  # Optional: Removes borders between items for a cleaner look
            start_collapsed=True,
            always_open=True,  # Optional: Allow multiple sections to be open at once
        )
    ],
    style={
        "padding-top": "20px",
        "padding-left": "20px",
        "padding-right": "20px"
    }
)


div_asian = html.Div([html.H4('Asian', style = {'margin' : '20px'}),
                      dbc.Row([dbc.Col(plot_first_n_simulations_asian, width=9),
                                dbc.Col(input_option_params, width=3)]),
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
    empty_fig.update_layout(template=cyborg_template)
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
    h = 0.01
    exotic_type = 'asian' # later will be value of menu bar 

    if n_clicks > 0 and Z_precomputed is not None:
        deltas = compute_delta(Z_precomputed, S0, K, T, r, sigma, h, exotic_type) # for asian, no kwargs
        return html.Div(f"{deltas['delta_call']*100:.2f}%"), html.Div(f"{(1 - deltas['delta_call'])*100:.2f}%")

    else: 
        # return empty values
        return html.Div(''), html.Div('')


@app.callback(
    Output('plot_delta_call_vs_stock_price_asian', 'figure'),
     Input('menu_bar', 'value'), ## at a later stage we will the value of the menu bar? for now, hardcoded for asian
     Input("button_update_params", "n_clicks"),
    State("input_S0", "value"),
    State("input_K", "value"),
    State("input_T", "value"),
    State("input_r", "value"),
    State("input_sigma", "value"))
def show_delta_plots(menu_bar_value, n_clicks, S0, K, T, r, sigma):
    h = 0.01
    S0_range = np.linspace(50, 150, 20)  # Shoule be flexible depending on input parameters
    exotic_type = 'asian' # later will be value of menu bar 

    if n_clicks > 0 and Z_precomputed is not None:

        fig  = plot_delta_vs_stock_price(Z_precomputed, S0_range, K, T, r, sigma, h, exotic_type)# for asian, no kwargs
        return fig
    
    else: 
        empty_fig = go.Figure()
        empty_fig.update_layout(template=cyborg_template)
        return empty_fig
    

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)