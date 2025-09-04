import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Video
import csv
import copy
import pandas as pd
import math
from sklearn.decomposition import PCA
from scipy import ndimage as ndi
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
    #Output("CPEI-output", "figure"),
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
    Output("CPEI-output1", "figure"),
    Output("CPEI-output", "figure", allow_duplicate=True),
    Input("compute-average-metrics", "n_clicks"),
    State("bbox-info-dict", "data"),
    State("shared-pass-table","data"),
    prevent_initial_call=True
)
def compute_average_metrics(compute_avg_clicks, bbox_info, shared_pass_data):

    # Setting up empty dictionaries to split up steps
    left_steps = {}
    right_steps = {}
   
  # get step frames and CoP
    for pass_id, box_info in bbox_info.items():
        # Getting the pass frames
        pass_start_frame = next(pass_info['start_frame'] for pass_info in shared_pass_data if pass_info['pass_idx'] == int(pass_id))
        pass_end_frame = next(pass_info['end_frame'] for pass_info in shared_pass_data if pass_info['pass_idx'] == int(pass_id))
        pass_data = trial_frames[pass_start_frame:pass_end_frame]
        
        # Predicting start and end frames for the step and getting the CoP trace
        for step_number, box in enumerate(box_info):
            box['original_step_frames'], box['CoP_x'], box['CoP_y'], box['start_frame'], box['end_frame'] = get_step_frames_and_CoP(box, pass_data, 5000)
            # Adding step number to help ID steps later if necessary. Could prolly find a way to do this from the begining
            box['step_number'] = step_number
            
            plt.imshow(box['original_step_frames'][:].max(0), cmap = jet_cmap)     
            plt.plot(box['CoP_x'], box['CoP_y'])
            plt.title("Original Step")
            plt.show()
    
            box['rc_step_max'], box['rc_CoP_x'], box['rc_CoP_y'], trisect_1, trisect_2 = plot_pc1_aligned(box['original_step_frames'][:].max(0), box['CoP_x'], box['CoP_y'], rot_crop_threshold_kPa=0)
            
            plt.imshow(box['rc_step_max'], cmap = jet_cmap)     
            plt.plot(box['rc_CoP_x'], box['rc_CoP_y'])
            plt.title("Rotated and Cropped Step")
            plt.show()
            
            # If left step
            if box['class'] == 1:
                left_steps[f'P{pass_id}_S{step_number}'] = {'rc_step_max':box['rc_step_max'], 'rc_CoP_x':box['rc_CoP_x'], 'rc_CoP_y':box['rc_CoP_y']}
            # If right step
            elif box['class'] == 2:
                right_steps[f'P{pass_id}_S{step_number}'] = {'rc_step_max':box['rc_step_max'], 'rc_CoP_x':box['rc_CoP_x'], 'rc_CoP_y':box['rc_CoP_y']}
            # If incomplete step, do nothing
            else:
                pass 
          
    
    out_right = align_and_average_heatmaps_padded(right_steps, alignment_threshold_kPa=1, reference_index=0) 
    out_left = align_and_average_heatmaps_padded(left_steps, alignment_threshold_kPa=1, reference_index=0)
    for out_R in [out_right, out_left]:
        # Plotting the masks and heatmaps to be sure everything looks good
        R_step_keys = out_R['step_keys']
        
        # Padded hms and CoPs
        padded_heatmaps = out_R['padded_heatmaps']
        padded_CoPs_list = out_R['padded_cop']
        
        # Padded masks
        padded_masks = out_R['padded_masks']
        
        # Overlap mask
        overlap_mask = out_R['avg_mask']
        
        # Average
        avg_hm = out_R['avg_heatmap']
        avg_cx = out_R['avg_cop']['x']
        avg_cy = out_R['avg_cop']['y']
        
        # Plotting the masks
        Rm_fig, Rm_axes = plt.subplots(1, len(R_step_keys)+1, figsize=(20,5))
        for i, R_step_key in enumerate(R_step_keys):
            padded_m = padded_masks[i]
            Rm_axes[i].imshow(padded_m, origin='upper',cmap=jet_cmap)  # same coordinate frame
            Rm_axes[i].set_title(f"Step {R_step_key} Mask")
        Rm_axes[len(R_step_keys)].imshow(overlap_mask, origin='upper',cmap=jet_cmap)  # same coordinate frame
        Rm_axes[len(R_step_keys)].set_title("Step Masks Overlapped")
        Rm_fig.tight_layout()
        
        # Plotting the heatmaps
        R_fig, R_axes = plt.subplots(1, len(R_step_keys)+1, figsize=(20,5))
        for i, R_step_key in enumerate(R_step_keys):
            padded_CoP_x = padded_CoPs_list[i]['x']
            padded_CoP_y = padded_CoPs_list[i]['y']
            padded_hm = padded_heatmaps[i]
            
            R_axes[i].imshow(padded_hm, origin='upper',cmap=jet_cmap)  # same coordinate frame
            R_axes[i].plot(padded_CoP_x, padded_CoP_y, linewidth=2)    # overlay avg CoP trajectory
            R_axes[i].set_title(f"Step {R_step_key} Heatmap")
            #plt.show()
        R_axes[len(R_step_keys)].imshow(avg_hm, origin='upper',cmap=jet_cmap)  # same coordinate frame
        R_axes[len(R_step_keys)].plot(avg_cx, avg_cy, linewidth=2)    # overlay avg CoP trajectory
        R_axes[len(R_step_keys)].set_title("Steps Average Heatmap")
        R_fig.tight_layout()
        plt.show()
        
        
       
    fig = go.Figure()

    # Heatmap (origin='upper' -> reverse y-axis in Plotly)
  
    fig= px.imshow(
         avg_hm,
         color_continuous_scale = plotly_jet,                 
         #zmin=np.nanmin(avg_hm),
         #zmax=np.nanmax(avg_hm),
         #colorbar=dict(title="Value"),
         #hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z}<extra></extra>"
     )
    
    
    # Avg CoP trajectory
    fig.add_trace(
        go.Scatter(
            x=avg_cx,
            y=avg_cy,
            mode="lines",
            line=dict(width=2),
            name="Avg CoP"
        )
    )
    
    # (Optional) mark start/end of CoP
    # fig.add_trace(go.Scatter(x=[avg_cx[0]], y=[avg_cy[0]], mode="markers", name="Start"))
    # fig.add_trace(go.Scatter(x=[avg_cx[-1]], y=[avg_cy[-1]], mode="markers", name="End"))
    
    fig.update_layout(
        title="Steps Average Heatmap",
        #template="plotly_white",
        #margin=dict(l=10, r=10, t=40, b=10),
        #xaxis=dict(constrain="domain"),
        #yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed")  # keeps image coords + square pixels
    )

    return fig, fig

# ---------- helpers ----------

def make_active_mask(hm, alignment_threshold_kPa):
    """Boolean mask of 'active' cells using an absolute kPa threshold. NaNs never active."""
    return np.isfinite(hm) & (hm >= alignment_threshold_kPa)

def phase_correlation_shift(ref_mask, mov_mask):
    """Integer-pixel shift (dy, dx) to align mov_mask to ref_mask via phase correlation."""
    A = ref_mask.astype(float)
    B = mov_mask.astype(float)
    FA = np.fft.rfft2(A)
    FB = np.fft.rfft2(B)
    R  = FA * np.conj(FB)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft2(R, s=A.shape)
    peak_y, peak_x = np.unravel_index(np.argmax(cc), cc.shape)
    H, W = A.shape
    dy = peak_y if peak_y <= H // 2 else peak_y - H
    dx = peak_x if peak_x <= W // 2 else peak_x - W
    return int(dy), int(dx)

def shift_with_nan(arr, dy, dx):
    """Shift a float heatmap by (dy, dx). Newly exposed pixels become NaN."""
    return ndi.shift(arr, shift=(dy, dx), order=1, mode='constant', cval=np.nan, prefilter=False)

def shift_mask(mask, dy, dx):
    """Shift a boolean mask by (dy, dx) with nearest-neighbor so it stays crisp."""
    shifted = ndi.shift(mask.astype(float), shift=(dy, dx), order=0, mode='constant', cval=0.0, prefilter=False)
    return shifted.astype(bool)


def pad_to_target_canvas(arr, target_shape, is_mask=False):
    """
    Pad a 2D array into the center of a fixed canvas (no resizing).
    Returns (canvas, (r0, c0)) where (r0, c0) is the top-left insertion offset.
    Heatmaps: float canvas filled with NaN; Masks: bool canvas filled with False.
    """
    Ht, Wt = target_shape
    H, W = arr.shape
    if H > Ht or W > Wt:
        raise ValueError(f"Array {H}x{W} is larger than target canvas {Ht}x{Wt} (no resizing allowed).")

    r0 = (Ht - H) // 2
    c0 = (Wt - W) // 2

    if is_mask:
        canvas = np.zeros((Ht, Wt), dtype=bool)
    else:
        canvas = np.full((Ht, Wt), 0, dtype=float)

    canvas[r0:r0+H, c0:c0+W] = arr
    return canvas, (r0, c0)

def pad_list_to_target(arrs, target_shape, is_mask=False):
    """Returns (padded_list, offsets_list). offsets[i] = (r0, c0) used for arrs[i]."""
    padded, offsets = [], []
    for a in arrs:
        p, off = pad_to_target_canvas(a, target_shape, is_mask=is_mask)
        padded.append(p); offsets.append(off)
    return padded, offsets

def nanmean_stack(arrs):
    """NaN-safe per-pixel mean over identically shaped float arrays. NaN where no coverage."""
    stack = np.stack(arrs, axis=0)
    valid = np.isfinite(stack)
    num = np.nansum(stack, axis=0)
    den = valid.sum(axis=0)
    out = num / np.maximum(den, 1)
    out[den == 0] = np.nan
    return out

def _resample_polyline_xy(x, y, n_points=101):
    """

    Resample a 2D polyline (x,y) to a fixed number of points using
    uniform param (index-based) interpolation. NaNs are dropped.
    Returns (xr, yr) as float arrays length n_points; NaNs if insufficient points.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    x = x[valid]; y = y[valid]
    t = np.linspace(0, len(x)-1, n_points)
    idx = np.arange(len(x))
    xr = np.interp(t, idx, x)
    yr = np.interp(t, idx, y)
    return xr, yr

def _clip_to_canvas(x, y, H, W):
    """Optionally clip coordinates to the canvas bounds (keeps floats)."""
    x = np.clip(x, 0, W-1)
    y = np.clip(y, 0, H-1)
    return x, y

# ---------- core pipeline (padding + translation + CoP averaging) ----------

def align_and_average_heatmaps_padded(
    side_steps_dict, # Only contains either L or R steps. 
    alignment_threshold_kPa,
    reference_index=0,
    avg_cop_points=101
):
    """
    Inputs
    ------
    all_steps_info : dict
        {'step_i': {'rc_step_max': 2D float array,
                    'rc_CoP_x':  1D float array (columns),
                    'rc_CoP_y':  1D float array (rows)}},
                    lots of other info that won't be used...
        'rc' indicates that the data has been rotated vertically and cropped
    alignment_threshold_kPa : float
        Absolute threshold (kPa) to build binary masks for alignment.
    reference_index : int
        Which padded sample to use as the registration reference (by insertion order of dict).
    avg_cop_points : int

        Number of samples for the time-normalized average CoP trajectory.

    Returns (dict)
    --------------
    {
      'padded_heatmaps':   list[H x W float],     # before alignment (NaN background)
      'padded_masks':      list[H x W bool],      # before alignment
      'aligned_heatmaps':  list[H x W float],     # after alignment
      'aligned_masks':     list[H x W bool],      # after alignment
      'avg_heatmap':       H x W float,           # NaN where no coverage
      'overlap_mask':      H x W bool,            # union of aligned masks
      'avg_mask':          H x W float,           # per-pixel fraction in [0,1]
      'shifts':            list[(dy, dx)],        # one per input, ref gets (0,0)
      'padded_cop':        list[{'x': 1D, 'y': 1D}],
      'aligned_cop':       list[{'x': 1D, 'y': 1D}],
      'avg_cop':           {'x': 1D length=avg_cop_points, 'y': 1D length=avg_cop_points}
    }
    """
    # Preserve insertion order of the dict
    step_keys = list(side_steps_dict.keys())

    # 0) Collect heatmaps and infer target canvas (largest H/W + small margin)
    H_list, W_list, heatmaps, cop_x_list, cop_y_list = [], [], [], [], []
    for k in step_keys:
        hm = side_steps_dict[k]['rc_step_max']
        H_list.append(hm.shape[0]); W_list.append(hm.shape[1])
        heatmaps.append(hm)
        cop_x_list.append(np.asarray(side_steps_dict[k]['rc_CoP_x'], float))
        cop_y_list.append(np.asarray(side_steps_dict[k]['rc_CoP_y'], float))

    target_H = max(H_list) + 3
    target_W = max(W_list) + 3
    target_shape = (target_H, target_W)

    # 1) Build masks on original bboxes (absolute threshold)
    masks = [make_active_mask(hm, alignment_threshold_kPa) for hm in heatmaps]

    # 2) Pad to fixed canvas (centered) and keep per-sample offsets
    padded_heatmaps, heat_offsets = pad_list_to_target(heatmaps, target_shape, is_mask=False)
    padded_masks,    mask_offsets = pad_list_to_target(masks,    target_shape, is_mask=True)
    # (offsets are identical for heatmap/mask, but we keep both for clarity)
    # heat_offsets[i] = (r0, c0) added to original CoP to land in the canvas before alignment

    # 3) Reference mask (already padded)
    ref_mask = padded_masks[reference_index]

    # 4) Align everyone to the reference (on masks), then shift heatmaps accordingly
    aligned_heatmaps, aligned_masks, shifts = [], [], []
    for i, (hm_pad, m_pad) in enumerate(zip(padded_heatmaps, padded_masks)):
        if i == reference_index or m_pad.sum() < 5:
            aligned_heatmaps.append(hm_pad)
            aligned_masks.append(m_pad)
            shifts.append((0, 0))
            continue
        dy, dx = phase_correlation_shift(ref_mask, m_pad)
        aligned_heatmaps.append(shift_with_nan(hm_pad, dy, dx))
        aligned_masks.append(shift_mask(m_pad, dy, dx))
        shifts.append((dy, dx))

    # 5) Averages & overlaps
    avg_heatmap = nanmean_stack(aligned_heatmaps)
    mask_stack  = np.stack([m.astype(float) for m in aligned_masks], axis=0)
    avg_mask    = mask_stack.mean(axis=0)          # fraction (0..1)
    overlap_mask = mask_stack.sum(axis=0) > 0

    # 6) Transform CoP traces into the same canvas & alignment
    padded_cop = []
    aligned_cop = []

    for i, (cx, cy) in enumerate(zip(cop_x_list, cop_y_list)):
        # drop non-finite samples
        valid = np.isfinite(cx) & np.isfinite(cy)
        cx = cx[valid]; cy = cy[valid]

        # add padding offset to land in canvas coords
        r0, c0 = heat_offsets[i]  # rows (y), cols (x)
        cx_pad = cx + c0
        cy_pad = cy + r0

        # apply alignment shift to reach the avg_heatmap frame
        dy, dx = shifts[i]
        cx_aln = cx_pad + dx
        cy_aln = cy_pad + dy

        padded_cop.append({'x': cx_pad, 'y': cy_pad})
        aligned_cop.append({'x': cx_aln, 'y': cy_aln})

    # 7) Time-normalized average CoP (resample each aligned trace to a common length)
    resampled_x = []
    resampled_y = []
    for tr in aligned_cop:
        xr, yr = _resample_polyline_xy(tr['x'], tr['y'], n_points=avg_cop_points)
        resampled_x.append(xr); resampled_y.append(yr)

    resampled_x = np.stack(resampled_x, axis=0)  # [Nsteps, T]
    resampled_y = np.stack(resampled_y, axis=0)

    # NaN-safe mean (shouldn’t have NaNs if traces had >=2 valid points)
    avg_cop_x = np.nanmean(resampled_x, axis=0)
    avg_cop_y = np.nanmean(resampled_y, axis=0)

    return {
        'padded_heatmaps':  padded_heatmaps,
        'padded_masks':     padded_masks,
        'aligned_heatmaps': aligned_heatmaps,
        'aligned_masks':    aligned_masks,
        'avg_heatmap':      avg_heatmap,
        'overlap_mask':     overlap_mask,
        'avg_mask':         avg_mask,
        'shifts':           shifts,
        'padded_cop':       padded_cop,        # list of per-step traces in padded (pre-align) canvas
        'aligned_cop':      aligned_cop,       # list of per-step traces in aligned canvas
        'avg_cop':          {'x': avg_cop_x, 'y': avg_cop_y, 'n_points': avg_cop_points},
        'target_shape':     target_shape,      # handy to keep around
        'step_keys':        step_keys          # maps list indices back to your dict keys
    }

# Function to rotate heatmap, crop it to activated frames and CoP and retreive trisections
def plot_pc1_aligned(step_data, CoP_x, CoP_y, rot_crop_threshold_kPa):
    H, W = step_data.shape
    
    # rot_crop_threshold_kPa is the threshold pressure used to rotate and crop the step
    mask = np.isfinite(step_data) & (step_data >  rot_crop_threshold_kPa)
    
    
    yy, xx = np.indices((H, W))
    X = np.column_stack([xx[mask].ravel(), yy[mask].ravel()])

    pca = PCA(n_components=2).fit(X)
    comp1 = pca.components_[0]
    PC1_angle_rads = np.arctan2(comp1[1], comp1[0])
    PC1_angle = np.degrees(PC1_angle_rads)

    # Rotate image (you’re using reshape=True)
    rotated = ndi.rotate(
        step_data,
        angle=PC1_angle,     # CCW degrees
        order=1,
        reshape=True,
        mode='constant',
        cval=0.0,
        prefilter=False
    )

    # Rotate CoP the *same way* (rotation + translation due to reshape=True)
    CoP_x_new, CoP_y_new = rotate_CoP_trace(
        np.asarray(CoP_x), np.asarray(CoP_y),
        H, W,
        angle_deg=PC1_angle,   # MUST match the angle passed to ndi.rotate
        reshape=True
    )

    # Getting rid of nan values in the CoP trace. Note that this will change the shape of the CoP trace array so the video will get funky
    non_nan_mask_x = ~np.isnan(CoP_x_new)
    non_nan_mask_y = ~np.isnan(CoP_y_new)

    CoP_x_new = CoP_x_new[non_nan_mask_x]
    CoP_y_new = CoP_y_new[non_nan_mask_y]


    # Crop the heatmap, shift CoP, and plit the foot into 3 regions
    rot_H, rot_W = rotated.shape

    
    # Find the horizontal bounds of the foot
    with np.errstate(all='ignore'):
        col_max = np.nanmax(rotated, axis=0)
    active = col_max > rot_crop_threshold_kPa

    # Find first and last active columns and slice between them
    left = int(np.argmax(active))                   # first True
    right = rot_W - int(np.argmax(active[::-1]))        # one past last True
    heatmap_new = rotated[:, left:right]

    # Shift the CoP trace
    CoP_x_new = CoP_x_new - left


    # ROTATING TO VERTICAL
    # Averaging the first and last ~10% of points that are not NaN
    CoP_x_start = np.mean(CoP_x_new[~np.isnan(CoP_x_new)][:len(CoP_x_new)//10])
    CoP_x_end = np.mean(CoP_y_new[~np.isnan(CoP_y_new)][-len(CoP_y_new)//10:])

    
    # Getting the center of the heatmap
    H, W = heatmap_new.shape

    
    # If the CoP goes from left to right, rotate the heatmap and the cop 90 deg CW, otherwise 90 CCW
    if CoP_x_start > CoP_x_end:
        CoP_x_new, CoP_y_new = rotate_CoP_trace(
        np.asarray(CoP_x_new), np.asarray(CoP_y_new),
        H, W,
        angle_deg=-90,   # MUST match the angle passed to ndi.rotate
        reshape=True)
        heatmap_new = np.rot90(heatmap_new, k=-1)

    else:
        CoP_x_new, CoP_y_new = rotate_CoP_trace(
        np.asarray(CoP_x_new), np.asarray(CoP_y_new),
        H, W,
        angle_deg=90,   
        reshape=True)
        heatmap_new = np.rot90(heatmap_new, k=1)
    
    


    # Get the trisections
    cropped_H = heatmap_new.shape[0]
    trisect_1 = round(cropped_H / 3)
    trisect_2 = round(cropped_H *2 / 3)

    # Get boundary points
    #x_upper, y_upper, x_lower, y_lower = get_ML_boundary_points(heatmap_new, trisect_1, trisect_2, boundary_pt_threshold_kPa=3)

    return heatmap_new, CoP_x_new, CoP_y_new, trisect_1, trisect_2#, x_upper, y_upper, x_lower, y_lower 

# Helper functions

def get_CoP(step_data):
    # step_data: (frames, height, width)
    F, H, W = step_data.shape
    
    # pixel coordinate grids (row = y, col = x). By default: origin at top-left
    yy, xx = np.indices((H, W))  # yy shape (H,W), xx shape (H,W)
    
    
    # Sums over space for each frame
    w_sum   = step_data.sum(axis=(1, 2))             # (F,)
    x_wsum  = (step_data * xx).sum(axis=(1, 2))      # (F,)
    y_wsum  = (step_data * yy).sum(axis=(1, 2))      # (F,)
    
    # Avoid divide-by-zero: where total pressure is 0, set COP to NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        CoP_x = x_wsum / w_sum    # (F,)
        CoP_y = y_wsum / w_sum    # (F,)
    return CoP_x, CoP_y 

# Funcion to rotate the CoP trace according to the reshaped heatmap
def rotate_CoP_trace(x, y, H, W, angle_deg, reshape=True):
    """
    Rotate point coordinates (x=cols, y=rows) the same way ndi.rotate does.

    Returns rotated (x', y') in the coordinate system of the output image.
    If reshape=True, includes the translation that places data on the expanded canvas.
    """
    # Center of original image
    cx, cy = W / 2.0, H / 2.0

    # Rotation matrix (same convention as your code: standard CCW in (x,y))
    theta = np.deg2rad(-angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Rotate points about the original center
    pts = np.vstack([x - cx, y - cy])          # shape (2, N)
    xr, yr = R @ pts
    xr += cx; yr += cy                          # back to image coords

    if not reshape:
        return xr, yr

    # --- Compute the translation that ndi.rotate(reshape=True) applies ---
    # Rotate the four image corners, then translate so min corner maps to (0,0)
    # Use pixel-center coordinates [0..W-1], [0..H-1]
    corners = np.array([[0,      0     ],
                        [W - 1., 0     ],
                        [W - 1., H - 1.],
                        [0,      H - 1.]], dtype=float)
    cpts = (R @ (corners.T - np.array([[cx],[cy]]))).T + np.array([cx, cy])

    min_x, min_y = cpts[:,0].min(), cpts[:,1].min()
    # This is the offset that places the rotated content within the new canvas
    tx, ty = -min_x, -min_y

    return xr + tx, yr + ty



# Function to get predicted step frames and CoP trace. The outputs step_data, CoP_x, and CoP_y will only output frames/values within the predicted start and end frames. We will still output true start and end frames for debugging purposes
def get_step_frames_and_CoP(step_info, pass_data, threshold_kPa):

    # Getting the coords of the bounding box corners, rounding to the nearest pixel
    y_min = math.floor(step_info['y0'])
    y_max = math.ceil(step_info['y1'])
    x_min = math.floor(step_info['x0'])
    x_max = math.ceil(step_info['x1'])
    
    # All frames from the pass within the step region
    step_data = pass_data[:, y_min:y_max, x_min:x_max]
    #step_data = np.rot90(step_data, axes=(1,2))
    # Sum pressures in each frame
    total_pressure_per_frame = step_data.sum(axis=(1, 2))
    
  
    
    active_frames = np.where(total_pressure_per_frame > threshold_kPa)

    # Setting the number of frames to pad the threshold with in case the first or last active frames are lower than the threshold value
    frame_padding = 5

    start_frame_pred = np.min(active_frames) - frame_padding if np.min(active_frames) - frame_padding >= 0 else 0 # Conditionals in case the padding extends beyond the bounds of the pass
    end_frame_pred = np.max(active_frames) + frame_padding if np.max(active_frames) - frame_padding <= step_data.shape[0] else step_data.shape[0]


    # Getting the step data within the predicted frames
    pred_step_data = step_data[start_frame_pred:end_frame_pred]

    # Getting the CoP excursion for the predicted step data
    CoP_x, CoP_y = get_CoP(pred_step_data)
    
    return pred_step_data, CoP_x, CoP_y, start_frame_pred, end_frame_pred    


# Function to get the CoP for a step
def get_CoP(step_data):
    # step_data: (frames, height, width)
    F, H, W = step_data.shape
    
    # pixel coordinate grids (row = y, col = x). By default: origin at top-left
    yy, xx = np.indices((H, W))  # yy shape (H,W), xx shape (H,W)
    
    
    # Sums over space for each frame
    w_sum   = step_data.sum(axis=(1, 2))             # (F,)
    x_wsum  = (step_data * xx).sum(axis=(1, 2))      # (F,)
    y_wsum  = (step_data * yy).sum(axis=(1, 2))      # (F,)
    
    # Avoid divide-by-zero: where total pressure is 0, set COP to NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        CoP_x = x_wsum / w_sum    # (F,)
        CoP_y = y_wsum / w_sum    # (F,)
    return CoP_x, CoP_y 






















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
        html.Div([dcc.Graph(id="CPEI-output"), dcc.Graph(id="CPEI-output1")])

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
   