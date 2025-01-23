from dash import Dash, Input, Output, html, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from app_new_folder.components import generate_main_div  # Import reusable components

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY, "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.9.1/font/bootstrap-icons.min.css"], )

app.title = "Price My Options NEW"

# Menu bar for selecting exotic options
menu_bar = html.Div([
    dmc.SegmentedControl(
        id="menu_bar",
        value="asian",
        fullWidth=True,
        data=[
            {"value": "asian", "label": "Asian"},
            {"value": "lookback", "label": "Lookback"},
            {"value": "value3", "label": "Label 3"},
        ]
    )
], style={"margin-bottom": "20px", "padding": "10px"})

# Generate divs for exotic options
div_asian = generate_main_div("asian")
div_lookback = generate_main_div("lookback")
div3 = html.Div(html.H4("Placeholder for Value 3"), id="div_value3", hidden=True)

# Define the app layout
app.layout = html.Div([
    html.H1("Price My Options", style={"textAlign": "center", "margin-top": "20px"}),
    menu_bar,
    div_asian,
    div_lookback,
    div3
])

# Callback to toggle visibility of divs based on the menu bar selection
@app.callback(
    [Output('div_asian', 'hidden'),
     Output('div_lookback', 'hidden'),
     Output('div_value3', 'hidden')],
    [Input('menu_bar', 'value')]
)
def show_hidden_div(input_value):
    # Default all divs to hidden
    show_div_asian = True
    show_div_lookback = True
    show_div3 = True

    # Show only the selected div
    if input_value == 'asian':
        show_div_asian = False
    elif input_value == 'lookback':
        show_div_lookback = False
    elif input_value == 'value3':
        show_div3 = False

    return show_div_asian, show_div_lookback, show_div3

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
