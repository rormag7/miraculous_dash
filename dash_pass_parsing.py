import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import matplotlib as mpl

# Create a function to automatically idenitify passes


# Create Jet colormap and force value 0 to be black
jet = mpl.colormaps['jet'](np.linspace(0, 1, 256))
jet[0] = [0, 0, 0, 1]  # Set zero value to black

# Convert to Plotly colorscale
plotly_jet = [
    [i / 255, f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})']
    for i, (r, g, b, a) in enumerate(jet)
]

# Load .npz file with 3D array (shape: num_frames x height x width)
data = np.load("trial.npz")
frames = data['arr_0']  # Replace with your actual array name
num_frames, height, width = frames.shape

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Heatmap Video Viewer with Frame Marking"),

    dcc.Graph(id="heatmap-frame", style={"height": "450px", "width": "1000px"}),

    dcc.Slider(
        id="frame-slider",
        min=0,
        max=num_frames - 1,
        value=0,
        step=1,
        marks={0: "Start", num_frames - 1: "End"},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Button("Mark Frame", id="mark-button", n_clicks=0),

    html.Div(id="marked-frames-output", style={"marginTop": "10px"}),

    dcc.Store(id="marked-frames", data=[])
])

@app.callback(
    Output("heatmap-frame", "figure"),
    Input("frame-slider", "value")
)
def update_heatmap(frame_idx):
    heatmap_data = np.rot90(frames[frame_idx])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        zmin=0,
        zmax=np.max(frames)*.75,
        colorscale=plotly_jet,
        colorbar=dict(title="Intensity")
    ))

    fig.update_layout(
        title=f"Frame {frame_idx}",
        xaxis=dict(showticklabels=False,showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
    )

    return fig

@app.callback(
    Output("marked-frames", "data"),
    Output("marked-frames-output", "children"),
    Input("mark-button", "n_clicks"),
    State("frame-slider", "value"),
    State("marked-frames", "data")
)
def mark__frame(n_clicks, current_frame, marked_frames):
    if ctx.triggered_id == "mark-button" and current_frame not in marked_frames:
        marked_frames.append(current_frame)
    marked_frames = sorted(set(marked_frames))
    return marked_frames, f"Marked Frames: {marked_frames}"

if __name__ == "__main__":
    app.run(debug=True)
