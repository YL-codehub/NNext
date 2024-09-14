import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(
    [
        html.H1("Interactive Plot with Parameters"),
        # Input fields for parameters
        html.Label("Enter value for X-axis range:"),
        dcc.Input(id="x-range", type="number", value=10),
        html.Label("Enter a multiplier for the function:"),
        dcc.Input(id="multiplier", type="number", value=1),
        # Graph output
        dcc.Graph(id="output-graph"),
    ]
)


# Callback to update the graph based on inputs
@app.callback(
    Output("output-graph", "figure"),
    [Input("x-range", "value"), Input("multiplier", "value")],
)
def update_graph(x_range, multiplier):
    # Generate some data for the plot
    x = np.linspace(0, x_range, 500)
    y = np.sin(x) * multiplier

    # Create the plotly figure
    fig = px.line(x=x, y=y, labels={"x": "X-axis", "y": "Y-axis"})
    fig.update_layout(title=f"Plot of y = sin(x) * {multiplier}")

    return fig


# Run the app
if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server()
