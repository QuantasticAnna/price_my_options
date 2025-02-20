from dash import Dash, Input, Output, html, State, callback_context, no_update, ctx, dcc
import dash_bootstrap_components as dbc
import numpy as np
import dash_mantine_components as dmc
import joblib
from app_folder.components import generate_main_div, empty_fig, button_run_new_simulations, OVERLAY_STYLE
from app_folder.components_model_div import  div_models
from constants import H, S0_RANGE, K_RANGE, PRICER_MAPPING, TTM_RANGE, EXOTIC_OPTION_TYPES, GREEKS, PLOTTERS, \
    N_SIMULATIONS, JOBLIB_DATA_PRECOMPUTED_Z_S_FILE, JOBLIB_GREEKS_VS_STOCK_PRICE_FILE, JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE, JOBLIB_GREEKS_VS_TTM_FILE, JOBLIB_OPTIONS_PRICES_AND_GREEKS
from greeks.delta import compute_delta
from greeks.gamma import compute_gamma
from greeks.vega import compute_vega
from greeks.theta import compute_theta
from greeks.rho import compute_rho
from greeks.greeks_functions import plot_greek_vs_stock_price, plot_greek_vs_strike_price, plot_greek_vs_ttm, greek_vs_stock_price, greek_vs_strike_price, greek_vs_ttm
import os
from pricer_plotter.monte_carlo import monte_carlo_simulations
from precomputed_data.precompute_data import precompute_heavy_data

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY, "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.9.1/font/bootstrap-icons.min.css"], )

app.title = "Price My Options"

def load_precomputed_data():
    """Loads precomputed data from Joblib file or generates it if missing."""
    if os.path.exists(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE):
        print(f"Loading precomputed data from {JOBLIB_DATA_PRECOMPUTED_Z_S_FILE}")
        return joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)
    else:
        print("Precomputed data not found. Generating new data...")
        precompute_heavy_data()  # Call precompute function to generate and save data
        return joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)  # Load newly generated data

load_precomputed_data()

# Menu bar for selecting option type
menu_bar = html.Div([
    dmc.SegmentedControl(
        id="menu_bar",
        value="models",
        fullWidth=True,
        data=[
            {"value": "models", "label": "Models"},
            {"value": "asian", "label": "Asian"},
            {"value": "lookback", "label": "Lookback"},
            {"value": "barrier", "label": "Barrier"},
            {"value": "european", "label": "European"},
        ]
    )
])
           


# Generate divs for exotic options 
div_asian = generate_main_div("asian")
div_lookback = generate_main_div("lookback")
div_barrier = generate_main_div("barrier")
div_european = generate_main_div("european")

# Define the app layout
app.layout = html.Div([
    html.H1("Price My Options", style={"textAlign": "center", "margin-top": "20px"}),
    menu_bar,
    button_run_new_simulations, # will be used for recompute z, doesn't appear on div_models
    dcc.Store(id="store_is_simulation_updated", data=False),  # Store cached data in memory,
    div_models,
    div_asian,
    div_lookback,
    div_barrier,
    div_european
], style = {'margin' : '30px'})

# Callback to toggle visibility of divs based on the menu bar selection
@app.callback(
    [Output('div_models', 'hidden'), 
     Output('div_asian', 'hidden'),
     Output('div_lookback', 'hidden'),
     Output('div_barrier', 'hidden'),
     Output('div_european', 'hidden')],
    [Input('menu_bar', 'value')],
)
def show_hidden_div(input_value):
    # Default all divs to hidden
    show_div_models = True
    show_div_asian = True
    show_div_lookback = True
    show_div_barrier = True
    show_div_european = True

    # Show only the selected div
    if input_value == 'models':
        show_div_models = False
    if input_value == 'asian':
        show_div_asian = False
    elif input_value == 'lookback':
        show_div_lookback = False
    elif input_value == 'barrier':
        show_div_barrier = False
    elif input_value == 'european':
        show_div_european = False

    return show_div_models, show_div_asian, show_div_lookback, show_div_barrier, show_div_european

# Callback to toggle visibility of the button based on the menu bar selection
@app.callback(
    Output('button_run_new_simulations', 'style'),
    [Input('menu_bar', 'value')],
)
def toggle_button_visibility(input_value):
    """
    This callback hides the button when 'models' is selected in the menu bar.
    Otherwise, it keeps the button visible.
    """
    if input_value == 'models':
        return {"display": "none"}  # Hide the button
    return {"display": "block"}  # Show the button

@app.callback(
    [Output("store_is_simulation_updated", "data")],
    Input("button_run_new_simulations", "n_clicks"),
    prevent_initial_call=True
)
def recompute_data(n_clicks):
    """Recomputes all precomputed data and updates store to trigger graph refresh."""
    if ctx.triggered_id == "button_run_new_simulations":
        precompute_heavy_data()  # Overwrite the Joblib file
        return tuple([True])  # Store becomes True

    return tuple(no_update)


@app.callback(
    [Output(f"plot_first_n_simulations_{exotic}", "figure") for exotic in EXOTIC_OPTION_TYPES],
    [
        Input(f"button_update_params_{exotic}", "n_clicks") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        Input("menu_bar", "value"),
        Input("store_is_simulation_updated", "data")  # New input for simulation updates
    ],
    [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def show_plot_first_n_simulations(*args):
    """
    Generates and updates simulation plots for exotic options.
    Updates when:
      - A specific button is clicked (button_update_params).
      - The menu_bar is changed (to an exotic option type).
      - The store_is_simulation_updated is True (indicating new simulations were run).
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_clicks = args[:n_exotics]  # Button clicks for parameter updates
    menu_selection = args[n_exotics]  # Menu selection value
    simulation_updated = args[n_exotics + 1]  # Whether simulations were updated
    states = args[n_exotics + 2:-2]  # Exclude barrier inputs
    B_call, B_put = args[-2], args[-1]  # Barrier inputs

    # Load latest Joblib data
    precomputed_data = joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)
    Z = precomputed_data['Z']

    figures = []
    split_states = [states[i::n_exotics] for i in range(n_exotics)]

    # Identify which element triggered the callback
    triggered_element = ctx.triggered_id

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):
        S0, K, T, r, sigma = state

        S = monte_carlo_simulations(Z, S0, T, r, sigma, N_SIMULATIONS)

        # Condition 1: Button click triggered update
        if clicks > 0 and triggered_element == f"button_update_params_{exotic}":
            update_plot = True
        # Condition 2: Menu selection matches this exotic type
        elif triggered_element == "menu_bar" and menu_selection == exotic:
            update_plot = True
        # Condition 3: Store flag is True (new simulations ran)
        elif triggered_element == "store_is_simulation_updated" and simulation_updated:
            update_plot = True
        else:
            update_plot = False

        if update_plot:
            if exotic == "barrier":
                if B_call is None or B_put is None:
                    figures.append(empty_fig)
                    continue
                plotter_barrier = PLOTTERS.get(exotic)
                fig_call, fig_put = plotter_barrier(S, B_call, B_put, n_sim_to_plot=10)
                figures.append(fig_call)
            else:
                plotter = PLOTTERS.get(exotic, None)
                if plotter is not None:
                    fig = plotter(S, n_sim_to_plot=10)
                    fig.add_hline(
                        y=K,
                        line=dict(color="white", width=2, dash="dash"),
                        annotation_text=f"Strike Price (K={K})",
                        annotation_position="bottom right",
                    )
                    figures.append(fig)
                else:
                    figures.append(empty_fig)
        else:
            figures.append(empty_fig)

    return tuple(figures)






@app.callback(
    [
        # Outputs for divs displaying Greeks
        Output(f"{greek}_{option_type}_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
        for option_type in ["call", "put"]
    ] +
    [
        # Outputs for the Greek table values
        Output(f"value_{greek}_{exotic}_call", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] +
    [
        Output(f"value_{greek}_{exotic}_put", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] +
    [
        # Outputs for the Option Prices table
        Output(f"price_call_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
    ] +
    [
        Output(f"price_put_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
    ],
    [Input(f"button_update_params_{exotic}", "n_clicks") for exotic in EXOTIC_OPTION_TYPES],
    [
        State(f"input_S0_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greeks_and_prices(*args):
    """
    Callback to compute and display Greek values (Delta, Gamma, Theta, Vega, Rho)
    and Option Prices for multiple exotic options dynamically.

    Uses precomputed values on initial load, then switches to real-time computations
    if the user updates parameters.
    """

    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)
    n_clicks = args[:n_exotics]  # Button clicks for each exotic type
    states = args[n_exotics:-2]  # Exclude barrier-specific inputs (last two states)
    B_call, B_put = args[-2], args[-1]  # Barrier-specific inputs

    # Reshape states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]

    div_results = []
    table_call_results = []
    table_put_results = []
    price_call_results = []
    price_put_results = []

    precomputed_data = joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)  
    Z = precomputed_data['Z']

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):

        # Use precomputed values if the button has not been clicked
        if clicks == 0:

            precomputed_greeks_and_prices = joblib.load(JOBLIB_OPTIONS_PRICES_AND_GREEKS)

            # Load precomputed values for this exotic option
            deltas = precomputed_greeks_and_prices[exotic]["deltas"]
            gammas = precomputed_greeks_and_prices[exotic]["gammas"]
            thetas = precomputed_greeks_and_prices[exotic]["thetas"]
            vegas = precomputed_greeks_and_prices[exotic]["vegas"]
            rhos = precomputed_greeks_and_prices[exotic]["rhos"]
            prices = precomputed_greeks_and_prices[exotic]["prices"]
        else:
            # Compute values dynamically if button clicked
            S0, K, T, r, sigma = state
            h = H
            Z = np.array(Z)  
            S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)
            pricer = PRICER_MAPPING.get(exotic)

            if exotic == "barrier":
                deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                gammas = compute_gamma(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                vegas = compute_vega(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                rhos = compute_rho(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                prices = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
            else:
                deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic)
                gammas = compute_gamma(Z, S0, K, T, r, sigma, h, exotic)
                thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic)
                vegas = compute_vega(Z, S0, K, T, r, sigma, h, exotic)
                rhos = compute_rho(Z, S0, K, T, r, sigma, h, exotic)
                prices = pricer(S, K, T, r)

        # Append results for divs
        div_results.extend([
            html.Div(f"{deltas['delta_call']:.2f}"),
            html.Div(f"{deltas['delta_put']:.2f}"),
            html.Div(f"{gammas['gamma_call']:.2f}"),
            html.Div(f"{gammas['gamma_put']:.2f}"),
            html.Div(f"{thetas['theta_call']:.2f}"),
            html.Div(f"{thetas['theta_put']:.2f}"),
            html.Div(f"{vegas['vega_call']:.2f}"),
            html.Div(f"{vegas['vega_put']:.2f}"),
            html.Div(f"{rhos['rho_call']:.2f}"),
            html.Div(f"{rhos['rho_put']:.2f}"),
        ])

        # Append results for the Greek table
        table_call_results.extend([
            f"{deltas['delta_call']:.2f}",
            f"{gammas['gamma_call']:.2f}",
            f"{thetas['theta_call']:.2f}",
            f"{vegas['vega_call']:.2f}",
            f"{rhos['rho_call']:.2f}",
        ])
        table_put_results.extend([
            f"{deltas['delta_put']:.2f}",
            f"{gammas['gamma_put']:.2f}",
            f"{thetas['theta_put']:.2f}",
            f"{vegas['vega_put']:.2f}",
            f"{rhos['rho_put']:.2f}",
        ])

        # Append results for the Prices table
        price_call_results.append(f"{prices['price_call']:.2f}")
        price_put_results.append(f"{prices['price_put']:.2f}")

    return tuple(div_results + table_call_results + table_put_results + price_call_results + price_put_results)



@app.callback(
    [
        Output(f"store_results_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_stock_price_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_results_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] + [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_stock_price_results(*args):
    """
    Callback to update or retrieve Greek vs Stock Price results.
    - If store is empty, load from 'all_exotic_greeks_results.joblib'.
    - If a button is clicked, compute and update results.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate inputs
    n_clicks = args[:n_exotics * n_greeks]  # Button clicks
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]  # Stored results
    states = args[n_exotics * n_greeks * 2:]  # Parameter states

    # Extract barrier inputs separately
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

    # Reshape states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]

    # Load precomputed results if store is empty
    if all(data is None for data in stored_data):
        if os.path.exists(JOBLIB_GREEKS_VS_STOCK_PRICE_FILE):
            print("Loading precomputed Greeks from file...")
            stored_data = joblib.load(JOBLIB_GREEKS_VS_STOCK_PRICE_FILE)  # Flat list
        else:
            print("No precomputed file found. Returning empty results.")
            return tuple(None for _ in range(n_exotics * n_greeks))

    # Convert stored_data to list if it's a tuple
    updated_results = list(stored_data)

    # Load latest Z from Joblib
    precomputed_data = joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)
    Z = precomputed_data['Z']

    # Compute missing results based on button clicks
    for exotic_index, (exotic, state_set) in enumerate(zip(EXOTIC_OPTION_TYPES, split_states)):
        S0, K, T, r, sigma = state_set

        h = H
        # S0_range = S0_RANGE
        S0_range = np.linspace(S0 * 0.5 , S0 * 1.5, 10)  # We limit the number of points to 10 as it is expensive computation

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            # If a button was clicked, recompute the result
            if triggered_button == f"button_compute_{greek}_vs_stock_price_{exotic}" and Z is not None:

                if exotic == "barrier":
                    results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
                    results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek)
                
                updated_results[output_index] = results  # Store updated results

    return tuple(updated_results)


@app.callback(
    [
        Output(f"plot_{greek}_vs_stock_price_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_results_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_stock_price_from_store(*stored_results):
    """
    Callback to render stored plots in their respective graphs.
    """
    return tuple(
        plot_greek_vs_stock_price(stored_results[i]) if stored_results[i] is not None else empty_fig
        for i in range(len(stored_results))
    )




@app.callback(
    [
        Output(f"store_results_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_strike_price_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_results_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] + [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_strike_price(*args):
    """
    Callback to dynamically update Greek vs Strike Price plots for multiple exotic options.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Extract barrier-specific inputs
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

    # Group states by exotic type
    n_states_per_exotic = 5  # S0, K, T, r, sigma
    split_states = [
        states[i::n_exotics] for i in range(n_exotics)
    ]

    # Load precomputed results if store is empty
    if all(data is None for data in stored_data):
        if os.path.exists(JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE):
            print("Loading precomputed Greeks from file...")
            stored_data = joblib.load(JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE)  # Flat list
        else:
            print("No precomputed file found. Returning empty results.")
            return tuple(None for _ in range(n_exotics * n_greeks))

    # Convert stored_data to list if it's a tuple
    updated_results = list(stored_data)

    for exotic_index, (exotic, state_set) in enumerate(zip(EXOTIC_OPTION_TYPES, split_states)):
        S0, K, T, r, sigma = state_set

        h = H
        # K_range = K_RANGE  # Define a range of strike prices
        K_range = np.linspace(S0 * 0.5 , S0 * 1.5, 10)  # We limit the number of points to 10 as it is expensive computation

        precomputed_data = joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)  
        Z = precomputed_data['Z']

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_button == f"button_compute_{greek}_vs_strike_price_{exotic}" and Z is not None:
                # Handle barrier-specific logic
                if exotic == "barrier":
                    results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
                    results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek)
                updated_results[output_index] = results

    return tuple(updated_results)




@app.callback(
    [
        Output(f"plot_{greek}_vs_strike_price_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_results_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_strike_price_from_store(*stored_results):
    """
    Callback to render stored plots for Greeks vs Strike Price in their respective graphs.
    Returns empty_fig if store data is None.
    """
    return tuple(
        plot_greek_vs_strike_price(stored_results[i]) if stored_results[i] is not None else empty_fig
        for i in range(len(stored_results))
    )


@app.callback(
    [
        Output(f"store_results_{greek}_vs_ttm_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_ttm_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_results_{greek}_vs_ttm_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] + [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_ttm(*args):
    """
    Callback to dynamically update Greek vs TTM plots for multiple exotic options.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Extract barrier-specific inputs
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

    # Group states by exotic type
    n_states_per_exotic = 5  # S0, K, T, r, sigma
    split_states = [
        states[i::n_exotics] for i in range(n_exotics)
    ]

    # Load precomputed results if store is empty
    if all(data is None for data in stored_data):
        if os.path.exists(JOBLIB_GREEKS_VS_TTM_FILE):
            print("Loading precomputed Greeks from file...")
            stored_data = joblib.load(JOBLIB_GREEKS_VS_TTM_FILE)  # Flat list
        else:
            print("No precomputed file found. Returning empty results.")
            return tuple(None for _ in range(n_exotics * n_greeks))

    # Convert stored_data to list if it's a tuple
    updated_results = list(stored_data)

    for exotic_index, (exotic, state_set) in enumerate(zip(EXOTIC_OPTION_TYPES, split_states)):
        S0, K, T, r, sigma = state_set

        h = H
        T_range = TTM_RANGE  # Define a range of TTM values

        precomputed_data = joblib.load(JOBLIB_DATA_PRECOMPUTED_Z_S_FILE)  
        Z = precomputed_data['Z']

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_button == f"button_compute_{greek}_vs_ttm_{exotic}" and Z is not None:
                # Handle barrier-specific logic
                if exotic == "barrier":
                    results = greek_vs_ttm(Z, S0, K, T_range, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
                    results = greek_vs_ttm(Z, S0, K, T_range, r, sigma, h, exotic, greek)
                updated_results[output_index] = results

    return tuple(updated_results)


@app.callback(
    [
        Output(f"plot_{greek}_vs_ttm_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_results_{greek}_vs_ttm_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_ttm_from_store(*stored_results):
    """
    Callback to render stored plots for Greeks vs TTM in their respective graphs.
    Returns empty_fig if store data is None.
    """
    return tuple(
        plot_greek_vs_ttm(stored_results[i]) if stored_results[i] is not None else empty_fig
        for i in range(len(stored_results))
    )



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
