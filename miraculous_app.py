import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Video
import csv
import copy
import pandas as pd
from ultralytics import YOLO
step_identification_model_l = YOLO(r'C:\Users\rorym\Downloads\FALL 2025\Applied Project\Code\Model_Weights\Step_Identification\yolov8l_best.pt')  # Loading best trained model

# Create Jet colormap and force value 0 to be black
jet = mpl.colormaps['jet'](np.linspace(0, 1, 256))
jet[0] = [0, 0, 0, 1]  # Set zero value to black
jet_cmap = mpl.colors.ListedColormap(jet)

# Convert to Plotly colorscale
plotly_jet = [
    [i / 255, f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})']
    for i, (r, g, b, a) in enumerate(jet)]



# Sample Figure
fig = go.Figure(
    )

# Add a scatter trace
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode="lines+markers",
    name="Line Plot"
))

# Add a bar trace
fig.add_trace(go.Bar(
    x=[1, 2, 3, 4],
    y=[5, 6, 7, 8],
    name="Bar Plot"
))

# Update layout
fig.update_layout(
    title="Line and Bar Example",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    legend_title="Legend",
    template="plotly_white"
)





# Class labels and colors for plotting bboxes
class_labels = {0:'Incomplete',
                1:'Left',
                2:'Right'}

class_colors = {0:'grey',
                1:'red',
                2:'royalblue'}

app = dash.Dash(__name__)

# Load .npz file with hardcoded filename, this will eventually get replaced
trial_file = "S145_W1.npz" #S145_W1.npz"
trial_name, ext = os.path.splitext(trial_file)
data = np.load(trial_file)
trial_frames = data['arr_0'][0:2000] # WILL NEED TO REMOVE [0:2000] LATER BUT WILL HELP TO SPEED UP DEVELOPMENT

trial_frames = np.rot90(trial_frames, k=1, axes=(1, 2))
num_frames = trial_frames.shape[0]
zmax_val = float(np.max(trial_frames))
marks = {0: "Start", num_frames - 1: "End"}


app.layout = html.Div([
        html.Img(
            src='assets/miraculous.png',
            style={
                'width': '400px',
                'display': 'block',
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'paddingTop': '7px'
            }
        ),
    
    dcc.Tabs(
        id="tabs",
        value="tab-1",  # Default selected tab
        children=[
            dcc.Tab(label="Pass Selection", value="tab-1"),
            dcc.Tab(label="Step Identification", value="tab-2"),
            dcc.Tab(label="Mask Adjustment", value="tab-3"),
            dcc.Tab(label="Metric Calculation", value="tab-4"),
            dcc.Tab(label="Report Generation", value="tab-5")
            
        ]
    ),

    html.Div(id="tabs-content"),  # This will display tab content
    dcc.Store(id="shared-pass-table"), #This will allow pass info to be used among all tabs
    dcc.Store(id="pass-max-dict"), #This will allow pass_max arrays from all passes to be used among all tabs
    dcc.Store(id="bbox-info-dict"), # This will allow for bbox info from all passes to be used among all tabs
    dcc.Store(id="pass-max-z") # Also add a store to hold the z for the selected pass
])



@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if tab == "tab-1":
        return tab1
    
    elif tab == "tab-2":
        return tab2
    
    elif tab == "tab-3":
        return tab3
    
    elif tab == "tab-4":
        return tab4
    
    elif tab == "tab-5":
        return tab5




#####################
## TAB 1 CALLBACKS ##
#####################

@app.callback(
    Output("frame-interval", "disabled"),
    Output("play-pause-btn", "children"),
    Input("play-pause-btn", "n_clicks"),
    State("frame-interval", "disabled"),
    prevent_initial_call=True
)
def toggle_play_pause(n, disabled):
    new_disabled = not disabled
    new_label = "⏸ Pause" if not new_disabled else "▶ Play"
    return new_disabled, new_label

@app.callback(
    Output("frame-slider", "value"),
    Input("frame-interval", "n_intervals"),
    State("frame-slider", "value"),
    State("frame-slider", "min"),
    State("frame-slider", "max"),
    State("frame-slider", "step"),
    prevent_initial_call=True
)
def advance_frame(_n, val, vmin, vmax, step):
    if val is None:
        raise exceptions.PreventUpdate
    nxt = (val + (step or 1))
    return vmin if nxt > vmax else nxt

@app.callback(
    Output("heatmap-frame", "figure"),
    Input("frame-slider", "value")
)
def update_trial_heatmap(frame_idx):
    if frame_idx is None:
        return go.Figure()
    heatmap_data = trial_frames[frame_idx]
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        zmin=0,
        zmax=zmax_val,
        colorscale=plotly_jet,
        colorbar=dict(title="Pressure (kPa)")
    ))
    fig.update_layout(
        title=dict(
        text=f"File: {trial_file} | Frame {frame_idx}",
        y=0.83, x = 0.5),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x", autorange="reversed")
    )
    return fig


@app.callback(
    Output("pass-table", "data"),
    Output("processing-complete-message", "children", allow_duplicate=True),
    Input("add-pass", "n_clicks"),
    Input("remove-pass", "n_clicks"),
    State("pass-table", "data"),
    prevent_initial_call=True
)
def update_pass_table(add_clicks, remove_clicks, table_data):
    triggered = ctx.triggered_id
    table_data = table_data or []
    if triggered == "add-pass":
        next_pass = len(table_data) + 1
        table_data.append({"pass_idx": next_pass, "start_frame": None, "end_frame": None})
    elif triggered == "remove-pass" and table_data:
        table_data.pop()
    
    return table_data, ""

@app.callback(
    Output("processing-complete-message", "children"),
    Output("shared-pass-table", "data"),
    Output("pass-max-dict", "data"),
    Output("bbox-info-dict", "data", allow_duplicate=True),
    Input("process-passes", "n_clicks"),
    State("pass-table", "data"),
    prevent_initial_call=True
)
def process_passes(process_clicks, table_data):
    trial_dir = trial_name
    trial_info_path = f'{trial_dir}/trial_information.csv'
    trial_fieldnames=['pass_idx', 'start_frame', 'end_frame']
    
    height, width = trial_frames[0].shape
    dpi = 100  # Set high enough to avoid scaling
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # fill the whole figure
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # For class IDs: 0 = Incomplete, 1 = Left, 2 = Right
    pass_max_dict = {}
    pass_preds_dict = {}
    pred_fieldnames=['class', 'confidence', 'x0', 'y0', 'x1', 'y1',
                     'type', 'fillcolor', 'editable', 'line']
    
    
    for pass_info in table_data:
        pass_idx = pass_info['pass_idx']
        start_frame = pass_info['start_frame']
        end_frame = pass_info['end_frame']
        pass_max = trial_frames[start_frame:end_frame].max(0)
        pass_max_dict[pass_idx] = pass_max
        pass_dir = f"{trial_name}/Pass{pass_idx}"
        os.makedirs(pass_dir, exist_ok=True)
        
        
        image_path = f'{pass_dir}/{trial_name}_Pass{pass_idx}_image.png'
        pred_path = f'{pass_dir}/{trial_name}_Pass{pass_idx}_predictions.csv'
        
    

        ax.imshow(pass_max, cmap=jet_cmap, aspect='auto')
        fig.savefig(image_path, dpi=dpi)
        plt.close(fig)
        
        results = step_identification_model_l.predict(
            source= image_path,  # could be a file, folder, list, or even a NumPy array
            imgsz=736,                         # match training image size (if resized during training)
            conf=0.25,                         # confidence threshold (optional)
            save=True,                         # save output image(s) with boxes
            project=pass_dir
        )

        
        predictions = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                pred = {"class": class_id, "confidence": confidence, "x0": xyxy[0], "y0": xyxy[1], "x1":xyxy[2], "y1": xyxy[3],
                        "type": "rect", "fillcolor": "rgba(0,0,0,0)", "editable": True, "line": {"color": class_colors[class_id], "width": 3}}
                predictions.append(pred)
                
        
        pass_preds_dict[pass_idx] = predictions
        
        
        with open(pred_path, "w", newline="") as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=pred_fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            writer.writerows(predictions)
            
        with open(trial_info_path, "w", newline="") as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=trial_fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            writer.writerows(table_data)
            
    
    return "Passes successfully processed.", table_data, pass_max_dict, pass_preds_dict



#####################
## TAB 2 CALLBACKS ##
#####################
@app.callback(
    Output('passes-dropdown', 'children'), # Output component and property
    Input("shared-pass-table", "data")
)
def create_pass_dropdown(pass_table_data):
    dropdown_list = []
    for pass_row in pass_table_data:
        pass_idx = pass_row['pass_idx']
        pass_start_end = [pass_row['start_frame'], pass_row['end_frame']]
        pass_entry = {'label': f'Pass {pass_idx}', 'value': str(pass_idx)}
        dropdown_list.append(pass_entry)
    return dcc.Dropdown(id='dropdown-output-container',  # Unique ID for the dropdown
    options=dropdown_list,
    value='1',  # Default selected value
    clearable=False # Prevents clearing the selection
    )

@app.callback(
    Output("pass-max-z", "data"),
    Input("dropdown-output-container", "value"),
    Input("pass-max-dict", "data"),
    prevent_initial_call=True
)
def load_pass_z(pass_selection, pass_max_dict):
    # Normalize key as string
    pid = str(pass_selection)
    return pass_max_dict.get(pid)

#Callback for visualizing and modifying bounding boxes
@app.callback(
    Output("bbox-info-dict", "data"),      # dict: pass_id -> list[shape]
    Output("pass-max", "figure"),
    Output("bbox-table", "data"),
    Input("add-box", "n_clicks"),
    Input("remove-selected", "n_clicks"),
    Input("pass-max", "relayoutData"),        # drag/resize events
    Input("bbox-table", "data_timestamp"),    # table edits
    State("bbox-table", "data"),
    State("bbox-table", "selected_rows"),
    State("bbox-info-dict", "data"),
    Input("pass-max-z", "data"),
    State("dropdown-output-container", "value"),
    prevent_initial_call=True
)
def update_app(add_clicks, remove_clicks, relayout_data,
               table_ts, table_data, selected_rows,
                shapes_by_pass, z, pass_selection):

    triggered = ctx.triggered_id
    # Normalize and copy state
    pid = str(pass_selection)     # pid is the numerical pass id
    shapes_by_pass = copy.deepcopy(shapes_by_pass or {})
    current_shapes = copy.deepcopy(shapes_by_pass.get(pid, []))

    # 1) Add new box
    if triggered == "add-box":
        current_shapes.append(make_new_box())

    # 2) Remove selected
    elif triggered == "remove-selected":
        if selected_rows:
            idx = selected_rows[0]
            if 0 <= idx < len(current_shapes):
                del current_shapes[idx]

    # 3) Handle shape drag/resize from Graph
    elif triggered == "pass-max" and relayout_data:
        # Keys look like: "shapes[0].x0": <val>
        for k, v in relayout_data.items():
            if k.startswith("shapes[") and "]." in k:
                try:
                    idx = int(k.split("[", 1)[1].split("]", 1)[0])
                    attr = k.split("].", 1)[1]
                    if 0 <= idx < len(current_shapes) and attr in {"x0","x1","y0","y1"}:
                        current_shapes[idx][attr] = float(v)
                except Exception:
                    pass  # ignore malformed keys

    # 4) Sync from table edits
    elif triggered == "bbox-table" and table_data:
        for i, row in enumerate(table_data):
            if i < len(current_shapes):
                for k in ("x0","y0","x1","y1"):
                    if k in row and row[k] is not None:
                        current_shapes[i][k] = float(row[k])
                if "class" in row and row["class"] in class_colors:
                    current_shapes[i]["class"] = int(row["class"])
                    if "line" not in current_shapes[i]:
                        current_shapes[i]["line"] = {}
                    current_shapes[i]["line"]["color"] = class_colors[row["class"]]
                    current_shapes[i]["line"]["width"] = 2

    # Save back only for this pass
    shapes_by_pass[pid] = current_shapes


    # Build figure and table for current pass
    title_text = f"File: {trial_file} | Pass {pid}"
    fig = create_figure(current_shapes, z, title_text)

    table_output = []
    for i, s in enumerate(current_shapes):
        table_output.append({
            "step_idx": i,
            "class": s.get("class", 0),
            "x0": round(float(s["x0"]), 0),
            "y0": round(float(s["y0"]), 0),
            "x1": round(float(s["x1"]), 0),
            "y1": round(float(s["y1"]), 0),
        })

    return shapes_by_pass, fig, table_output

# Callback for analyzing data from the selected step
@app.callback(
    Output("CPEI-output", "figure"),
    Input("analyze-selected", "n_clicks"),
    State("bbox-table", "selected_rows"),
    State("bbox-table", "data"),
    State("shared-pass-table","data"),
    State("dropdown-output-container", "value"),
    prevent_initial_call=True
)
def get_CPEI(analyze_clicks, selected_step, bbox_table_data, pass_table_data, selected_pass):
    print(f"Selected Pass: {selected_pass}")
    selected_pass = int(selected_pass)
    selected_step = selected_step[0]
    
    # Getting the info from the selected pass and creating an array of frames from the pass
    selected_pass_info = next((pass_info for pass_info in pass_table_data if pass_info['pass_idx'] == selected_pass), "busted")
    pass_start_frame = selected_pass_info["start_frame"]
    pass_end_frame = selected_pass_info["end_frame"]
    pass_frames = trial_frames[pass_start_frame:pass_end_frame]
    
    # Getting the coordinates of the selected step
    selected_step_info = next((step_info for step_info in bbox_table_data if step_info['step_idx'] == selected_step), "busted")
    x0 = selected_step_info['x0']
    y0 = selected_step_info['y0']
    x1 = selected_step_info['x1']
    y1 = selected_step_info['y1']
    
    print(f"Start Frame: {pass_start_frame}")
    print(f"End Frame: {pass_end_frame}") 
    
    print()
    print(f"Selected Step: {selected_step}")
    print(f"{selected_step_info}")
    
    threshold_kPa=500 # Still need to figure out what the best threshold is
    step_frames, total_pressure_per_frame = get_step_frames(pass_frames, x0, y0, x1, y1, threshold_kPa)
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(step_frames[:].max(0), cmap=jet_cmap)
    plt.close()

    def update(frame):
        im.set_array(step_frames[frame])
        ax.set_title(f'Step_{selected_step} | Total Pressure: {total_pressure_per_frame[frame]} kPa')
        return [im]
    
    ani = FuncAnimation(
        fig,
        update,
        frames=step_frames.shape[0],
        interval=50,      # milliseconds per frame
        blit=False
    )
    
    step_vid_file = f'Step_{selected_step}.mp4'
    ani.save(step_vid_file, writer='ffmpeg', fps=20)
    Video(step_vid_file)
    
    
    
    fig = go.Figure(
        )

    # Add a scatter trace
    fig.add_trace(go.Scatter(
        x=[1, 2, 3, 4],
        y=[10, 11, 12, 13],
        mode="lines+markers",
        name="Line Plot"
    ))

    # Add a bar trace
    fig.add_trace(go.Bar(
        x=[1, 2, 3, 4],
        y=[5, 6, 7, 8],
        name="Bar Plot"
    ))

    # Update layout
    fig.update_layout(
        title="Line and Bar Example",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        legend_title="Legend",
        template="plotly_white"
    )
    
    return fig,

# Helper functions
def make_new_box(x0=7, x1=77, y0=7, y1=77, class_id=0):
    return {
        'type': 'rect',
        'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
        'line': {'color': class_colors[class_id], 'width': 2},
        'fillcolor': 'rgba(0,0,0,0)',
        'editable': True,
        'class': class_id
    }

def create_figure(shapes, z, title_text):
    plot_shapes = []
    ann = []
    padx, pady = -1, -15
    
    for i, s in enumerate(shapes):
        sc = s.copy()
        # color by class id; remove non-plotly keys
        class_id = sc.get('class', 0)
        sc.pop('class', None)
        conf = sc.get('confidence', 0)
        sc.pop('confidence', None)
        sc['line'] = {'color': class_colors.get(class_id, 'grey'), 'width': 2}
        plot_shapes.append(sc)
        
        cx = (float(s["x0"]) + float(s["x1"])) / 2.0
        cy = (float(s["y0"]) + float(s["y1"])) / 2.0
        ann.append(dict(
            x=cx, y=cy, xref="x", yref="y",
            text=str(i), showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.55)",
            xanchor="center", yanchor="middle"
        ))
        

    # Start with heatmap
    fig = go.Figure(
        data=[go.Heatmap(
            z=z, zmin=0, zmax=zmax_val, colorscale=plotly_jet,
            colorbar=dict(title="Pressure (kPa)")
        )],
        layout=dict(
            title=dict(text=title_text, y=0.83, x=0.5),
            shapes=plot_shapes,
            annotations=ann,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x", autorange="reversed"),
            uirevision="keep",
            legend=dict(title="Step Classification:",
                        orientation="h",
                        y=0, x=0.25)    
        )
    )

    # Add one dummy legend item per class
    for cid, label in class_labels.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=class_colors[cid], width=3),
            name=label,
            showlegend=True
        ))

    return fig


# Function to identify the range of frames belonging to the selected step within a specific pass
def get_step_frames(pass_frames, x0, y0, x1, y1, threshold_kPa):
    
    # Ordering coords so the step doesn't get reflected when plotting
    x_min, x_max = min(x0,x1), max(x0,x1)
    y_min, y_max = min(y0,y1), max(y0,y1)
    
    # All frames from the pass within the step region
    all_step_frames = pass_frames[:, y_min:y_max, x_min:x_max]
    
    # Sum pressures in each frame
    total_pressure_per_pass_frame = all_step_frames.sum(axis=(1, 2))


    active_frames = np.where(total_pressure_per_pass_frame > threshold_kPa)
    # Could probably add something in here to be sure the frame gaps aren't too large in case there is random noise before or later in the pass
    #print(f"Active Frames: {active_frames}")
    
    # Setting the number of frames to pad the threshold with in case the first or last active frames are lower than the threshold value
    frame_padding = 3

    start_frame_pred = np.min(active_frames) - frame_padding if np.min(active_frames) - frame_padding >= 0 else 0 # Conditionals in case the padding extends beyond the bounds of the pass
    end_frame_pred = np.max(active_frames) + frame_padding if np.max(active_frames) - frame_padding <= all_step_frames.shape[0] else all_step_frames.shape[0]

    # Slicing the pass frame to get only frames where the step is happening
    pred_step_frames = all_step_frames[start_frame_pred:end_frame_pred]
    total_pressure_per_step_frame = total_pressure_per_pass_frame[start_frame_pred:end_frame_pred]
    
    return pred_step_frames, total_pressure_per_step_frame
 
   


#####################
## TAB 3 CALLBACKS ##
#####################


# Callback to get average step metrics
@app.callback(
    #Output("avg-left-output", "figure"),
    #Output("avg-right-output", "figure"),
    Input("compute-average-metrics", "n_clicks"),
    State("bbox-info-dict", "data"),
    State("shared-pass-table","data"),
    prevent_initial_call=True
)
def compute_average_metrics(compute_avg_clicks, bbox_info, shared_pass_data):
    print()
    print(f"SHARED PASS DATA: {shared_pass_data}")
    print()
    
  
    for pass_id, box_info in bbox_info.items():
        for box in box_info:
            box['start_frame'] = 'meow'
            box['end_frame'] = 'meow'
            box['CoP_x'] = 'meow' #need to actually do this
            box['CoP_y'] = 'meow'
                
    print(bbox_info)
    
    # rotate heatmap and cop by PC1
    
    # identify pass direction and rotate appropriately
    
    # split into left and rightz
    
    # get average step

    

        # Splitting step info into left and right steps

























#####################
##   TAB LAYOUTS   ##
#####################
tab1 = html.Div([
        html.Div([
            dcc.Graph(id="heatmap-frame", style={"height": "435px", "width": "990px", "marginBottom": "0px"}),
            html.Div(html.Button("▶ Play", id="play-pause-btn", n_clicks=0), style={'textAlign': 'center'}),
            dcc.Interval(id="frame-interval", interval=20, disabled=True),  # 20 ms per frame
            dcc.Slider(
                id="frame-slider",
                min=0,
                max=num_frames - 1,
                value=0,
                step=1,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='mouseup'
            )], style={'flex': '2', 'paddingRight': '20px'}),

        html.Div([
            html.H4("Pass Selection Table", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='pass-table',
                columns=[
                    {"name": "Pass #", "id": "pass_idx", "type": "numeric", "editable": False},
                    {"name": "Start Frame", "id": "start_frame", "type": "numeric", "editable": True},
                    {"name": "End Frame", "id": "end_frame", "type": "numeric", "editable": True}
                ],
                data=[],
                editable=True,
                row_deletable=False,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'}
            ),
            html.Div([html.Button("Add Pass", id="add-pass", n_clicks=0, style={'marginTop': '10px', 'marginRight': '10px'}),
                      html.Button("Remove Pass", id="remove-pass", n_clicks=0)], style={'textAlign': 'center'}),
                
            html.Div([html.Button("Save and Process Passes", id="process-passes", n_clicks=0, style={'marginTop': '10px'}),
                      dcc.Loading(id="loading-box", type="default", children=html.Div(id="processing-complete-message"), style={'marginTop': '70px'})] , style={'textAlign': 'center'})
                     
            ], style={'flex': '1'})
    ], style={'display': 'flex', 'flexDirection': 'row'})




tab2 = html.Div([
    html.H4("Select a pass to view"),
    

    html.Div(id='passes-dropdown'), # Div to display dropdown
    
    html.Div([
            html.Div(dcc.Graph(id="pass-max", 
                               style={"height": "420px", "width": "940px", "marginBottom": "0px"},
                               config={'editable': False, 'edits': {'shapePosition': True}}), 
                               style={'flex': "0 0 30%", 'paddingRight': '0px'}),
            html.Div([html.Div([
                html.H4("Step Identification Table", style={'textAlign': 'center'}),
                dash_table.DataTable(
                        id="bbox-table",
                        columns=[
                            {"name": "Step #", "id": "step_idx", "editable": False},
                            {"name": "Class", "id": "class", "presentation": "dropdown"},
                            {"name": "x0", "id": "x0", "type": "numeric"},
                            {"name": "y0", "id": "y0", "type": "numeric"},
                            {"name": "x1", "id": "x1", "type": "numeric"},
                            {"name": "y1", "id": "y1", "type": "numeric"},
                        ],
                        dropdown={
                            "class": {
                                "options": [{"label": class_labels[i], "value": i} for i in class_labels]
                            }
                        },
                        row_selectable="single",
                        editable=True,
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center"},
                    ),
               
                html.Button("Add Box", id="add-box", n_clicks=0, style={"marginTop": "10px", "marginRight": "10px"}),
                html.Button("Remove Selected Box", id="remove-selected", n_clicks=0)
            ], style={'textAlign': 'center', "marginBottom": "10px", 'flex': "0 0 70%"}),
                
                html.Div(html.Button("Analyze Selected Step", id="analyze-selected", n_clicks=0), style={"marginTop": "15px", 'textAlign': 'center'}),
                html.Div(html.Button("Compute Average Metrics", id="compute-average-metrics", n_clicks=0), style={"marginTop": "15px", 'textAlign': 'center'}) # WIP
                
        
    
              ],style={'flex': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row',  
                  "overflow": "hidden",
                  "width": "100%"}),
        html.Div(dcc.Graph(id="CPEI-output"))

])



tab3 = html.Div([
    html.H4("Heatmap and Mask Overlay"),
    html.P("Heatmap with mask and controls would go here..."),
    html.Div([dcc.Graph(id="avg-left-output"), dcc.Graph(id="avg-right-output")])
])



tab4 = html.Div([
    html.H4("Select and Calculate Metrics"),
    html.P("Split this up with sections for AI, CPEI, and FPA")
])


tab5 = html.Div([
    html.H4("Preview and Generate report PDF"),
    html.P("meow meow meow TBD")
])

    
if __name__ == "__main__":
    app.run(debug=True)
   