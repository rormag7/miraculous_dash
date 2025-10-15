from dash import Dash, dcc, html, Output, Input, State
import plotly.graph_objects as go
import numpy as np

app = Dash(__name__)

# Sample data
x = np.arange(50)
y = np.cumsum(np.random.randn(50)) + 10
init_y = 12.0  # initial line value

# Base fig
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Data"))

# Legend-visible trace (mirrors draggable shape)
fig.add_trace(go.Scatter(
    x=[x.min(), x.max()],
    y=[init_y, init_y],
    mode="lines",
    line=dict(dash="dash", width=2),
    name=f"Threshold = {init_y:.2f}",
))

# Draggable horizontal shape for the threshold(drag along y)
fig.add_shape(
    type="line",
    xref="paper", yref="y",
    x0=0, x1=1,         # span full plot width
    y0=init_y, y1=init_y,
    line=dict(color="red", width=2, dash="dash"),
)

# Regular layout is fine—NO 'edits' here
fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))

app.layout = html.Div([
    dcc.Graph(
        id="g",
        figure=fig,
        style={"height": "500px"},
        # Put 'editable' and 'edits' here:
        config={
            "editable": True,
            "edits": {"shapePosition": True}  # ← enables dragging shapes
        },
    ),
])

@app.callback(
    Output("g", "figure"),
    Input("g", "relayoutData"),
    State("g", "figure"),
    prevent_initial_call=True
)
def on_drag(relayout, fig):
    if not relayout:
        return fig

    # Get new y from the shape move
    new_y = relayout.get("shapes[0].y0", relayout.get("shapes[0].y1"))
    if new_y is None:
        return fig

    # Use the visible x-axis range to span the line & trace
    # (avoids issues with numpy-serialized x arrays)
    xaxis_range = (
        fig.get("layout", {}).get("xaxis", {}).get("range", [0, 1])
    )
    x0, x1 = xaxis_range[0], xaxis_range[1]

    # Snap the shape to perfectly horizontal across full width
    fig["layout"]["shapes"][0]["xref"] = "paper"  # keep shape spanning full width
    fig["layout"]["shapes"][0]["x0"] = 0
    fig["layout"]["shapes"][0]["x1"] = 1
    fig["layout"]["shapes"][0]["y0"] = new_y
    fig["layout"]["shapes"][0]["y1"] = new_y

    # Ensure the legend-visible trace exists (trace index 1 in our setup)
    if len(fig["data"]) < 2:
        fig["data"].append({
            "type": "scatter",
            "mode": "lines",
            "line": {"dash": "dash", "width": 2},
            "name": f"Threshold = {new_y:.2f}",
            "x": [x0, x1],
            "y": [new_y, new_y],
            "showlegend": True
        })
    else:
        fig["data"][1]["x"] = [x0, x1]
        fig["data"][1]["y"] = [new_y, new_y]
        fig["data"][1]["name"] = f"Threshold = {new_y:.2f}"

    return fig

if __name__ == "__main__":
    app.run(debug=True)
