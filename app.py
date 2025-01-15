
from dash import Dash, dcc, html, callback, Input, Output
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
app.title = "Dash App Template"

menu_bar = html.Div([dmc.SegmentedControl(id = "menu_bar",
                                            value = "value1",
                                            fullWidth = True,
                                            data = [
                                                {"value": "value1", "label": "Label 1"},
                                                {"value": "value2", "label": "Label 2"},
                                                {"value": "value3", "label": "Label 3"},
                                            ])], style = {'margin' : '20px'})

div1 = html.Div([html.H4('Div 1', style = {'margin' : '20px'})],
                id = 'div1')

div2 = html.Div([html.H4('Div 2', style = {'margin' : '20px'})],
                id = 'div2')  
                       
div3 = html.Div([html.H4('Div 3', style = {'margin' : '20px'})],
                id = 'div3')

app.layout = html.Div([menu_bar, 
                       div1,
                       div2,
                       div3])

@callback(
    [Output('div1', 'hidden'),
     Output('div2', 'hidden'), 
     Output('div3', 'hidden')],
    [Input('menu_bar', 'value')]
)
def show_hidden_div(input_value):
    show_div1 = True
    show_div2 = True
    show_div3 = True

    if input_value == 'value1':
        show_div1 = False
    elif input_value == 'value2':
        show_div2 = False
    elif input_value == 'value3':
        show_div3 = False

    return(show_div1, show_div2, show_div3)


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)