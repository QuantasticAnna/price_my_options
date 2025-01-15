
from dash import Dash, dcc, html, callback, Input, Output, State
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime 
from pricer.monte_carlo import monte_carlo_simulations, plotter_first_n_simulations
from pricer.asian import plotter_asian
from pricer.lookback import plotter_lookback

# Global variable to store Z
Z_global = None

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY])
app.title = "Dash App Template"

menu_bar = html.Div([dmc.SegmentedControl(id = "menu_bar",
                                            value = "asian",
                                            fullWidth = True,
                                            data = [
                                                {"value": "asian", "label": "Asian"},
                                                {"value": "lookback", "label": "Lookback"},
                                                {"value": "value3", "label": "Label 3"},
                                            ])], style = {'margin' : '20px'})

button_generate_z = html.Button("Generate Simulations", id="button_generate_z", n_clicks=0)
store_z = dcc.Store(id="store_z")
z_status = html.Div(id="status", style={"margin-top": "20px"})

input_option_params = html.Div([
                                html.H4("Specify Option Parameters"),
                                html.Label("Initial Stock Price (S0):"),
                                dcc.Input(id="input_S0", type="number", value=100, step=1),
                                
                                html.Label("Strike Price (K):"),
                                dcc.Input(id="input_K", type="number", value=100, step=1),
                                
                                html.Label("Time to Maturity (T):"),
                                dcc.Input(id="input_T", type="number", value=1, step=0.1),
                                
                                html.Label("Risk-Free Rate (r):"),
                                dcc.Input(id="input_r", type="number", value=0.05, step=0.01),
                                
                                html.Label("Volatility (σ):"),
                                dcc.Input(id="input_sigma", type="number", value=0.2, step=0.01),

                                html.Button("Update Parameters", id="button_update_params", n_clicks=0, style={"margin-top": "20px"}),

                            ], style={"margin-bottom": "20px"})

plot_first_n_simulations = dcc.Graph(id="plot_first_n_simulations", style={"height": "600px"})

plot_first_n_simulations_asian = dcc.Graph(id="plot_first_n_simulations_asian", style={"height": "600px"})

plot_first_n_simulations_lookback = dcc.Graph(id="plot_first_n_simulations_lookback", style={"height": "600px"})

div_asian = html.Div([html.H4('Asian', style = {'margin' : '20px'})],
                id = 'div_asian')

div_lookback = html.Div([html.H4('Lookback', style = {'margin' : '20px'})],
                id = 'div_lookback')  
                       
div3 = html.Div([html.H4('Div 3', style = {'margin' : '20px'})],
                id = 'div3')

app.layout = html.Div([menu_bar, 
                       input_option_params,
                       plot_first_n_simulations,
                       plot_first_n_simulations_asian,
                       plot_first_n_simulations_lookback,
                       div_asian,
                       div_lookback,
                       div3,
                       html.Div([button_generate_z, store_z, z_status])])

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

@app.callback(
    Output("status", "children"),  # Update status message
    Output("store_z", "data"),  # Store Z in dcc.Store
    Input("button_generate_z", "n_clicks")  # Trigger on button click
)
def generate_simulations(n_clicks):
    global Z_global
    if n_clicks > 0:
        # Generate Z and store it globally
        n_simulations = 100000  # Number of simulations
        Z = np.random.standard_normal((n_simulations, 252))  # 252 days (e.g., trading days in a year)
        Z_global = Z  # Store in global variable
        return f"Simulations generated! at {datetime.now()}", Z.tolist()  # Store Z in dcc.Store as JSON serializable format
    return "Click the button to generate simulations.", None


@app.callback(
    Output("plot_first_n_simulations", "figure"),
    Output("plot_first_n_simulations_asian", "figure"),
    Output("plot_first_n_simulations_lookback", "figure"),
    Input("button_update_params", "n_clicks"),
    State("input_S0", "value"),
    State("input_T", "value"),
    State("input_r", "value"),
    State("input_sigma", "value"),
    State("store_z", "data")
)
def show_plot_first_n_simulations(n_clicks, S0, T, r, sigma, Z_data):
    if n_clicks > 0 and Z_data is not None:
        Z = np.array(Z_data)  # Convert Z back to NumPy array
        S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=100000)
        fig_S = plotter_first_n_simulations(S, n_sim_to_plot=10)
        fig_asian = plotter_asian(S, n_sim_to_plot=10)
        fig_lookback = plotter_lookback(S, n_sim_to_plot=10)
        return fig_S, fig_asian, fig_lookback
    return go.Figure(), go.Figure(), go.Figure() # empty figures


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)