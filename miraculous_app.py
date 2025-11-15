import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Video
import csv
import copy
import pandas as pd
import math
import base64
import json
#import io
from io import BytesIO
from pathlib import Path
from flask import send_file
from svglib.svglib import svg2rlg
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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



# Class labels and colors for plotting bboxes
class_labels = {0:'Incomplete',
                1:'Left',
                2:'Right'}

class_colors = {0:'grey',
                1:'royalblue',
                2:'red'}

app = dash.Dash(__name__)

# Load .npz file with hardcoded filename, this will eventually get replaced
trial_file = "S145_W1.npz" 
trial_name, ext = os.path.splitext(trial_file)
data = np.load(trial_file)
trial_frames = data['arr_0'][0:1500] # WILL NEED TO REMOVE [:] LATER BUT WILL HELP TO SPEED UP DEVELOPMENT
sample_rate = 100 #Hz
tile_size = 0.5 # cm 

trial_frames = np.rot90(trial_frames, k=1, axes=(1, 2))
num_frames = trial_frames.shape[0]
zmax_val = float(np.max(trial_frames))
marks = {0: "Start", num_frames - 1: "End"}
"""
initial_pathology_options = [
    {"label": "Cerebral Palsy", "value": "Cerebral Palsy"},
    {"label": "Club Foot", "value": "Club Foot"},
    {"label": "ACL Pre Op", "value": "ACL Pre Op"},
    {"label": "ACL Post Op", "value": "ACL Post Op"},
    {"label": "NA", "value": "NA"},
]
initial_project_options = [
    {"label": "Markerless Integration Study", "value": "Markerless Integration Study"},
    {"label": "Club Foot Correction Project", "value": "Club Foot Correction Project"},
    {"label": "ACL Rehab Project", "value": "ACL Rehab Project"},
    {"label": "NA", "value": "NA"},
]
"""
# initial project and pathology options are stored in this file
dropdown_filename ="dropdown_values.csv"

pathology_list = []
project_list = []

with open(dropdown_filename, mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Skipping header row
    next(csv_reader)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        if row[0] == "Pathology":
            pathology_list.append(row[1])
        else:
           project_list.append(row[1])

initial_pathology_options = [{"label": pathology, "value": pathology} for pathology in pathology_list]
initial_project_options = [{"label": project, "value": project} for project in project_list]

#####################
##   TAB LAYOUTS   ##
#####################
tab1 = html.Div([
        html.Div([
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
        ], style={'display': 'flex', 'flexDirection': 'row'}),
        
        
        html.Div([html.H4("Input Patient Information", style={'textAlign': 'left', 'marginTop': '30px'}),
                  html.Label("First Name: "),
                  dcc.Input(id="first-name", type="text", placeholder="Enter first name"),
                  
                  html.Br(),
                  html.Label("Last Name: "),
                  dcc.Input(id="last-name", type="text", placeholder="Enter last name", style={'marginTop': '10px'}),
                  
                  html.Br(),
                  html.Div([
                            html.Label("Sex: ", style={"marginRight": "10px"}),
                            dcc.Dropdown(
                                id="sex",
                                options=[
                                    {"label": "Male", "value": "Male"},
                                    {"label": "Female", "value": "Female"},
                                ],
                                placeholder="Select sex",
                                style={"width": "150px"}
                            )
                        ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
                  
                  html.Label("Body Weight (N): "),
                  dcc.Input(id="body-weight", type="number", placeholder="0", style={'marginTop': '10px'}),
                  
                  html.Br(),
                  html.Label("Date of Birth (YYYY-MM-DD): "),
                  dcc.Input(id="birth-date", type="text", placeholder="YYYY-MM-DD", style={'marginTop': '10px'}),

                  html.Br(),
                  html.Label("Date of Assessment (YYYY-MM-DD): "),
                  dcc.Input(id="assessment-date", type="text", placeholder="YYYY-MM-DD", style={'marginTop': '10px'}),

                  html.Br(),
                  html.Div([
                            html.Label("Recording Type: ", style={"marginRight": "10px"}),
                            dcc.Dropdown(
                                id="recording-type",
                                options=[
                                    {"label": "Pathological", "value": "Pathological"},
                                    {"label": "Normative", "value": "Normative"},
                                ],
                                placeholder="Select recording type",
                                style={"width": "177px"}
                            )
                        ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
                  
                  
                  html.Div([
                            html.Label("Pathology: ", style={"marginRight": "10px"}),
                            dcc.Dropdown(
                                id="pathology",
                                options=initial_pathology_options,
                                searchable=True, 
                                value=None,
                                placeholder="Type to search…",
                                style={"width": "150px"}
                            ),
                            html.Button("Add this as new option", id="add-pathology", n_clicks=0, style={"marginLeft": 7}),
                            dcc.Store(id="new-pathology-option", data=""), 
                            dcc.Store(id="pathology-options-store", data=initial_pathology_options),
                        ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"},),

    
                    html.Div([
                              html.Label("Project: ", style={"marginRight": "10px"}),
                              dcc.Dropdown(
                                  id="project",
                                  options=initial_project_options,
                                  searchable=True,  
                                  placeholder="Type to search…",
                                  style={"width": "150px"}
                              ),
                              html.Button("Add this as new option", id="add-project", n_clicks=0, style={"marginLeft": 7}),
                              dcc.Store(id="new-project-option", data=""), 
                              dcc.Store(id="project-options-store", data=initial_project_options),
                          ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
    

                  html.Label("Notes: "),
                  dcc.Input(id="patient-notes", type="text", placeholder="", style={'marginTop': '10px', 'width': '250px'}),
                
                  html.Br(),
                  html.Button("Save Patient Information", id="save-patient-info", n_clicks=0, style={"marginTop": "20px"}),
                  html.Div(id="patient-info-save-message", style={"marginTop": "10px", "color": "green"})
                 ])

    ])


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
            ], style={'textAlign': 'center', "marginBottom": "5px", 'flex': "0 0 70%"}),
                
               
        
                html.Div(
                    [
                        html.Div([
                            html.Button("Analyze Selected Step", id="analyze-selected", n_clicks=0,
                                        style={"marginTop": "5px"}),
                            dcc.Loading(
                                id="loading-analyze",
                                type="default",
                                children=html.Div(id="analyze-complete-message"),
                                style={"marginTop": "50px"}
                            ),
                        ], style={"marginRight": "5px"}), 
                
                        html.Div([
                            html.Button("Compute Average Metrics", id="compute-average-metrics", n_clicks=0,
                                        style={"marginTop": "5px"}),
                            dcc.Loading(
                                id="loading-compute",
                                type="default",
                                children=html.Div(id="averaging-complete-message"),
                                style={"marginTop": "50px"}
                            ),
                        ]),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "flex-start",  # keep tops aligned
                        "marginBottom": "50px",
                        "gap": "5px",               # space between the two blocks
                    },
                )
        
    
              ],style={'flex': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row',  
                  "overflow": "hidden",
                  "width": "100%"}),
        html.Div([dcc.Graph(id="CPEI-output"), dcc.Graph(id="CPEI-output1")])

])


tab5 = html.H4("Single Step View")
                  
tab3 = html.Div([
    html.Div(dcc.Graph(id="avg-steps-hm", style={"height": 520, "width": "100%"})),
    html.Div(dcc.Graph(id="avg-force-mag", style={"height": 520, "width": "100%"})),
    html.Div(
    [
        html.H4("Average Metrics Table", style={"textAlign": "center"}),

        html.Div(
            dash_table.DataTable(
                id="avg-metrics-table",
                columns=[],
                data=[],
                style_table={"width": "50%"},
                style_cell={"textAlign": "center"},
                style_header={"fontWeight": "bold"},
                style_data_conditional=[
                    {   # Bold + gray background for first column
                        "if": {"column_id": "Foot"},
                        "fontWeight": "bold",
                        "backgroundColor": "#f2f2f2",
                    }
                ],
            ),
            style={"display": "flex", "justifyContent": "center"},
        ),
    ]
),
             
])



tab4 = html.Div([
    html.H4("Preview and Generate report PDF"),
    html.P("Dropdown for trial std or normative range?"),
    html.P("Dropdown to change color of metric value in the table based on if it is high, low, or in range?"),
    html.P("Specify file name and save location? This may be have a default based on pathology and research project but can be changed"),
    html.Div(
        [
            html.Div(
                [html.Button("Generate PDF", id="generate-pdf", n_clicks=0, style={"marginTop": "16px"})],
            ),
            dcc.Download(id="download-pdf"),

            html.Hr(),
            html.H4("PDF Preview"),

            # Controls for the PDF iframe
            html.Div(
                [
                    html.Button("Refresh Preview", id="refresh-preview", n_clicks=0,
                                style={"marginRight": "8px"}),
                    dcc.Checklist(
                        id="fit-width",
                        options=[{"label": " Fit preview to tab width", "value": "fit"}],
                        value=[],
                        style={"display": "inline-block", "verticalAlign": "middle"}
                    ),
                ],
                style={"marginBottom": "8px"}
            ),

            # True PDF preview
            dcc.Loading(
                html.Iframe(
                    id="pdf-preview",
                    style={
                        # 8.5x11 in at ~96 dpi => ~816x1056 px (nice physical feel)
                        "width": "816px",
                        "height": "1056px",
                        "border": "1px solid #ccc",
                        "background": "white",
                        "boxShadow": "0 0 8px rgba(0,0,0,0.15)"
                    }
                ),
                type="default"
            ),
            html.Div(id="pdf-preview-note",
                     style={"color": "#667", "fontSize": "12px", "marginTop": "6px"}), #originally 667-1
        ],
        style={"padding": "16px"}
    )
])



#APP LAYOUT
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
            dcc.Tab(label="Single Step Analysis", value="tab-5", children=tab5),
            dcc.Tab(label="Average Step Analysis", value="tab-3", children=tab3),
            dcc.Tab(label="Report Generation", value="tab-4")
            
        ]
    ),

    html.Div(id="tabs-content"),  # This will display tab content
    dcc.Store(id="shared-pass-table"), #This will allow pass info to be used among all tabs
    dcc.Store(id="pass-max-dict"), #This will allow pass_max arrays from all passes to be used among all tabs
    dcc.Store(id="bbox-info-dict"), # This will allow for bbox info from all passes to be used among all tabs
    dcc.Store(id="pass-max-z"), # Also add a store to hold the z for the selected pass
    dcc.Store(id="avg-left-data"),
    dcc.Store(id="avg-right-data"),
    dcc.Store(id="patient-info-store"), # This stores the patient info inputted on the first tab
    dcc.Store(id="avg-steps-json"),
    dcc.Store(id="avg-force-mag-json") # Storing the figures in json format to be passed between tabs
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
    
    #elif tab == "tab-3":
        #return tab3
    
    elif tab == "tab-4":
        return tab4



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
        raise dash.exceptions.PreventUpdate
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
    Output("processing-complete-message", "children", allow_duplicate=True),
    Output("shared-pass-table", "data", allow_duplicate=True),
    Output("pass-max-dict", "data", allow_duplicate=True),
    Output("bbox-info-dict", "data", allow_duplicate=True),
    Output("tabs", "value", allow_duplicate=True),                 
    Input("process-passes", "n_clicks"),
    State("pass-table", "data"),
    prevent_initial_call=True
)
def process_passes(process_clicks, table_data):
    # Only run on a real click of the button
    if not process_clicks or ctx.triggered_id != "process-passes":
        raise dash.exceptions.PreventUpdate
    print('running?')
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
                pred_pad = 3 # padded 3 to predictions to be sure nothing gets cut off
                pred = {"class": class_id, "confidence": confidence, "x0": xyxy[0]-pred_pad, "y0": xyxy[1]-pred_pad, "x1":xyxy[2]+pred_pad, "y1": xyxy[3]+pred_pad,
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
            
    return "Passes successfully processed.", table_data, pass_max_dict, pass_preds_dict, "tab-2"

@app.callback(
    Output("patient-info-store", "data"),
    Output("patient-info-save-message","children"),
    Input("save-patient-info", "n_clicks"),
    State("first-name", "value"),
    State("last-name", "value"),
    State("sex", "value"),
    State("body-weight", "value"),
    State("birth-date", "value"),
    State("assessment-date", "value"),
    State("recording-type", "value"),
    State("pathology", "value"),
    State("project", "value"),
    State("patient-notes", "value"),
    prevent_initial_call=True
)
def save_patient_info(n_clicks, first_name, last_name, sex, body_weight, 
                      birth_date, assessment_date, recording_type, pathology, 
                      project, notes):
    # Preventing updates if the button hasn't actually been clicked
    if not n_clicks:
       raise dash.exceptions.PreventUpdate
       
    # Store inputs as a dictionary
    patient_data = {
        "first_name": first_name,
        "last_name": last_name,
        "sex": sex,
        "body_weight": body_weight,
        "birth_date": birth_date,
        "assessment_date": assessment_date,
        "recording_type": recording_type,
        "pathology": pathology,
        "project": project,
        "notes": notes
    }
    print("Saved Patient Info:", patient_data)  # for debugging
    save_message = "Patient information successfully saved."
    return patient_data, save_message

# Callbacks to add new pathology
@app.callback(
    Output("new-pathology-option", "data"),
    Input("pathology", "search_value"),
    prevent_initial_call=True
)
def cache_pathology_search_text(s):
    if not s:
        # nothing typed → no changes
        return dash.no_update
    return (s or "").strip()

@app.callback(
    Output("pathology-options-store", "data"),
    Output("pathology", "value"),
    Input("add-pathology", "n_clicks"),
    State("new-pathology-option", "data"),
    State("pathology-options-store", "data"),
    prevent_initial_call=True
)
def add_pathology_option(n_clicks, typed, stored_options):
    text = (typed or "").strip()
    print(f'New pathology added: {text}')
    if not text:
        # nothing typed → no changes
        return stored_options, dash.no_update

    # case-insensitive de-duplication
    existing_lower = {opt["value"].lower() for opt in stored_options}
    if text.lower() in existing_lower:
        # already exists → just select it
        return stored_options, text

    new_opt = {"label": text, "value": text}
    updated = stored_options + [new_opt]
    
    # Adding to CSV
    with open(dropdown_filename, 'a', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['Pathology',text])
        
    return updated, text

# 3) Keep the dropdown options in sync with the canonical store
@app.callback(
    Output("pathology", "options"),
    Input("pathology-options-store", "data")
)
def sync_patholgy_options(pathology_data):
    return pathology_data


# Callbacks to add new project
@app.callback(
    Output("new-project-option", "data"),
    Input("project", "search_value"),
    prevent_initial_call=True
)
def cache_project_search_text(s):
    if not s:
        # nothing typed → no changes
        return dash.no_update
    return (s or "").strip()

@app.callback(
    Output("project-options-store", "data"),
    Output("project", "value"),
    Input("add-project", "n_clicks"),
    State("new-project-option", "data"),
    State("project-options-store", "data"),
    prevent_initial_call=True
)
def add_project_option(n_clicks, typed, stored_options):
    text = (typed or "").strip()
    print(f'New project added: {text}')
    if not text:
        # nothing typed → no changes
        return stored_options, dash.no_update

    # case-insensitive de-duplication
    existing_lower = {opt["value"].lower() for opt in stored_options}
    if text.lower() in existing_lower:
        # already exists → just select it
        return stored_options, text

    new_opt = {"label": text, "value": text}
    updated = stored_options + [new_opt]
    
    # Adding to CSV
    with open(dropdown_filename, 'a', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(['Project',text])
    return updated, text

# 3) Keep the dropdown options in sync with the canonical store
@app.callback(
    Output("project", "options"),
    Input("project-options-store", "data")
)
def sync_project_options(project_data):
    return project_data

#####################
## TAB 2 CALLBACKS ##
#####################
@app.callback(
    Output('passes-dropdown', 'children'), # Output component and property
    Input("shared-pass-table", "data"),
)
def create_pass_dropdown(pass_table_data):
    print(f'PASS TABLE DATA: {pass_table_data}')
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

# Callback to display average step heatmaps and average table data
@app.callback(
    Output("avg-steps-hm", "figure"),
    Output("avg-steps-json", "data"),
    Output("avg-force-mag", "figure"),
    Output("avg-force-mag-json", "data"),
    Input("tabs", "value"),
    Input("avg-left-data", "data"),
    Input("avg-right-data", "data"),
    State("patient-info-store", "data"),
    prevent_initial_call=True
)
def create_avg_figs(tab, left_data, right_data, patient_info):
    if tab != "tab-3" or not left_data or not right_data:
        raise dash.exceptions.PreventUpdate
    
    # Unpack
    hm_l = np.asarray(left_data["avg_heatmap"])
    cx_l = left_data["avg_cop"]["x"]
    cy_l = left_data["avg_cop"]["y"]

    hm_r = np.asarray(right_data["avg_heatmap"])
    cx_r = right_data["avg_cop"]["x"]
    cy_r = right_data["avg_cop"]["y"]

    # Shared scale (compute per-array, then take scalar min/max)
    # This avoids stacking arrays of different shapes.
    try:
        vmin = float(min(np.nanmin(hm_l), np.nanmin(hm_r)))
        vmax = float(max(np.nanmax(hm_l), np.nanmax(hm_r)))
    except ValueError:
        # Handles all-NaN slices or other oddities
        raise dash.exceptions.PreventUpdate

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=("Averaged Maximum Left Step", "Averaged Maximum Right Step")
    )

    # Heatmaps (shared coloraxis → single colorbar)
    fig.add_trace(
        go.Heatmap(z=hm_l, zmin=vmin, zmax=vmax, coloraxis="coloraxis"),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=hm_r, zmin=vmin, zmax=vmax, coloraxis="coloraxis"),
        row=1, col=2
    )

    # CoP overlays (hide duplicate legends on right)
    fig.add_trace(go.Scatter(x=cx_l, y=cy_l, mode="lines", name="Avg CoP", line=dict(width=2, color='coral')), 1, 1)
    fig.add_trace(go.Scatter(x=[cx_l[0]],  y=[cy_l[0]],  mode="markers",
                             marker=dict(size=9, color='orange'),
                             name="Start"), 1, 1)
    
    fig.add_trace(go.Scatter(x=[cx_l[-1]], y=[cy_l[-1]], mode="markers", 
                             marker=dict(size=9, color='lime'),
                             name="End"),   1, 1)

    fig.add_trace(go.Scatter(x=cx_r, y=cy_r, mode="lines", name="Avg CoP", line=dict(width=2,  color='coral'), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=[cx_r[0]],  y=[cy_r[0]],  mode="markers", 
                             marker=dict(size=9, color='orange'),
                             name="Start", showlegend=False), 1, 2)
    
    fig.add_trace(go.Scatter(x=[cx_r[-1]], y=[cy_r[-1]], mode="markers", 
                             marker=dict(size=9, color='lime'),
                             name="End", showlegend=False), 1, 2)

    # Layout & axes
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis=dict(
            colorscale=plotly_jet,
            colorbar=dict(title="kPa", y=0.5, yanchor="middle")
        ),
        height=520,
        legend=dict(
        orientation="h",
        x=0.5, y=-0.05,      # bottom center, below plot
        xanchor="center",
        yanchor="top"
    )
    )


    fig.update_yaxes(
        title_text="Length (cm)",
        tickvals=np.arange(0,101,10),             # positions in pixels
        ticktext=[v * 0.5 for v in np.arange(0,101,10)],  # labels in cm
        autorange="reversed", 
        scaleanchor="x",
        scaleratio=1,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Width (cm)",
        constrain="domain", 
        tickvals=np.arange(0,101,10),             # positions in pixels
        ticktext=[v * 0.5 for v in np.arange(0,101,10)],  # labels in cm
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Length (cm)",
        tickvals=np.arange(0,101,10),             # positions in pixels
        ticktext=[v * 0.5 for v in np.arange(0,101,10)],  # labels in cm
        autorange="reversed", 
        scaleanchor="x2", 
        scaleratio=1,
        row=1, col=2
    )
    
    fig.update_xaxes(
        title_text="Width (cm)",
        constrain="domain", 
        tickvals=np.arange(0,101,10),             # positions in pixels
        ticktext=[v * 0.5 for v in np.arange(0,101,10)],  # labels in cm
        row=1, col=2
    )
        
    
    # Average force curve code below...
    body_weight = patient_info["body_weight"]
    
    left_avg_mag = np.array(left_data['avg_magnitude_curve'])
    left_std_mag = np.array(left_data['std_magnitude_curve'])
    left_avg_mag_BW = np.array(left_data['avg_magnitude_curve']) / body_weight * 100 # Normalized BW percentages
    left_std_mag_BW = np.array(left_data['std_magnitude_curve']) / body_weight * 100
    
    right_avg_mag = np.array(right_data['avg_magnitude_curve'])
    right_std_mag = np.array(right_data['std_magnitude_curve'])
    right_avg_mag_BW = np.array(right_data['avg_magnitude_curve']) / body_weight * 100
    right_std_mag_BW = np.array(right_data['std_magnitude_curve']) / body_weight * 100
    
    left_mag_x = np.linspace(0, 100, len(left_avg_mag))  # left step cycle percentage
    right_mag_x = np.linspace(0, 100, len(right_avg_mag)) # right step cycle percentage
    
    init_thresh = 100  # initial line value
    
    mag_fig = go.Figure()
    
    # Average curve
    mag_fig.add_trace(
        go.Scatter(
            x=left_mag_x,
            y=left_avg_mag_BW,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Left Average"
        )
    )
    
    # Standard deviation band (shaded area)
    mag_fig.add_trace(
        go.Scatter(
            x=np.concatenate([left_mag_x, left_mag_x[::-1]]), 
            y=np.concatenate([left_avg_mag_BW - left_std_mag_BW, (left_avg_mag_BW + left_std_mag_BW)[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),  # hide line
            hoverinfo="skip",
            showlegend=True,
            name="±1 Left Std Dev"
        )
    )
    
    # Average curve
    mag_fig.add_trace(
        go.Scatter(
            x=right_mag_x,
            y=right_avg_mag_BW,
            mode="lines",
            line=dict(color="red", width=2),
            name="Right Average"
        )
    )
    
    # Standard deviation band (shaded area)
    mag_fig.add_trace(
        go.Scatter(
            x=np.concatenate([right_mag_x, right_mag_x[::-1]]), 
            y=np.concatenate([right_avg_mag_BW - right_std_mag_BW, (right_avg_mag_BW + right_std_mag_BW)[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),  # hide line
            hoverinfo="skip",
            showlegend=True,
            name="±1 Right Std Dev"
        )
    )
    
    # Legend-visible trace (mirrors draggable shape)
    mag_fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[init_thresh, init_thresh],
        mode="lines",
        line=dict(dash="dash", width=2),
        name=f"{init_thresh:.0f}% BW",
    ))

    # Draggable horizontal shape for the threshold(drag along y)
    mag_fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1,         # span full plot width
        y0=init_thresh, y1=init_thresh,
        line=dict(color="orange", width=2, dash="dash"),
    )
    
    # Layout
    mag_fig.update_layout(
        title=dict(
        text="Average Step Force Profile",
        x=0.5,              # 0 = left, 0.5 = center, 1 = right
        y=0.95,             # vertical placement (1.0 is top, <1 moves it down)
        xanchor="center",   # anchor relative to x
        yanchor="top" ),      # anchor relative to y
        xaxis_title="Step cycle (%)",
        yaxis_title="Body Weight %",
        template="plotly_white",
        legend=dict(
        orientation="h",
        x=0.5, y=-0.25,      # bottom center, below plot
        xanchor="center",
        yanchor="bottom"
        ) 
    )
        
    # Updating the format and scaling of the figures for the report PDF
    fig_copy = go.Figure(fig)
    report_fig = tune_figure_for_pdf(fig_copy)
    
    mag_fig_copy = go.Figure(mag_fig)
    report_mag_fig = tune_figure_for_pdf(mag_fig_copy,12,4,-1)
    
    # Converting each figure to json to be stored for later use
    avg_steps_json = report_fig.to_json()
    force_mag_json = report_mag_fig.to_json()
    return fig, avg_steps_json, mag_fig, force_mag_json

def tune_figure_for_pdf(fig, content_w_in=12, content_h_in=7, legend_y=-.23, base_font="Helvetica"):
    """
    Freeze the figure size and add spacing so legends/titles don't overlap in PDF.
    content_w_in/h_in = the space you intend on the PDF page (inches).
    """
    PT = 72
    report_fig = fig
    width_pt  = int(round(content_w_in * PT))
    height_pt = int(round(content_h_in * PT))

    # Fonts to match your ReportLab styles
    report_fig.update_layout(
        width=width_pt,
        height=height_pt,
        template="plotly_white",
        #font=dict(family=base_font, size=17, color="green"),
        title=dict(
            font=dict(family=f"{base_font}-Bold", size=12, color="black"),
            x=0.0, xanchor="left", pad=dict(t=4, b=6, l=0, r=0)
        ),
        # Give extra top margin if legend sits above the plot
        margin=dict(l=60, r=20, t=80, b=60),  # adjust once; stable thereafter
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    # Axis title/label spacing + auto margins for dense ticks
    #report_fig.update_xaxes(title_standoff=10, automargin=True)
    #report_fig.update_yaxes(title_standoff=10, automargin=True)
    
    title_font_size = 10
    axis_font_size = 10
    

    
    report_fig.for_each_xaxis(_update_axis)
    report_fig.for_each_yaxis(_update_axis)
    
    anns = list(report_fig.layout.annotations or [])
    for ann in anns:
        # Only touch subplot titles (they usually have xref like 'x1', 'x2' and yref like 'paper')
        ann.font = dict(family=f"{base_font}-Bold", size=17, color="black")
        # Optional nudge up to avoid crowding the top axis labels
        if getattr(ann, "yshift", None) in (None, 0):
            ann.yshift = 8
    
    # Legend: put it above the plotting area in a row and leave room via margin.t
    report_fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom", y=legend_y,
            xanchor="center",  x=0.5,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            itemwidth=80,         # helps long labels wrap consistently
            tracegroupgap=8,
            itemsizing="constant"
        )
    )
    return report_fig



def _update_axis(ax):
   ax.update(
       title_font=dict(family="Helvetica", size=12, color="black"),
       tickfont=dict(family="Helvetica", size=12, color="black"),
       title_standoff=10,
       automargin=True,
   )

# Callback to get average step metrics and add metrics to average metrics table
@app.callback(
    Output("averaging-complete-message", "children"),
    Output("avg-left-data", "data"),
    Output("avg-right-data", "data"),
    Output("avg-metrics-table", "data"),
    Output("avg-metrics-table", "columns"),
    Output("tabs", "value"),                 
    Input("compute-average-metrics", "n_clicks"),
    State("bbox-info-dict", "data"),
    State("shared-pass-table","data"),
    prevent_initial_call=True
)
def compute_average_metrics(compute_avg_clicks, bbox_info, shared_pass_data):
    if not compute_avg_clicks:  # covers None or 0
        raise dash.exceptions.PreventUpdate
    
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
    
            box['rc_step_max'], box['rc_CoP_x'], box['rc_CoP_y'], box['trisect_1'], box['trisect_2'] = plot_pc1_aligned(box['original_step_frames'][:].max(0), box['CoP_x'], box['CoP_y'], rot_crop_threshold_kPa=0)

        
            plt.imshow(box['rc_step_max'], cmap = jet_cmap)
            plt.hlines(box['trisect_1'], 0, box['rc_step_max'].shape[1]-1)
            plt.hlines(box['trisect_2'], 0, box['rc_step_max'].shape[1]-1)
            plt.plot(box['rc_CoP_x'], box['rc_CoP_y'])
            plt.title(f"Rotated and Cropped P{pass_id}_S{step_number}")
            plt.show()
            
            #Plotting the total pressure magnitude per step frames
            # Each sensor is .000025m^2 and the original pressure is in kPa, so multiplying out to get force in Newtons: F = P*A
            step_frame_force_magnitude = ((1000*.000025)*box['original_step_frames'][:]).sum(axis=(1, 2))
            plt.plot(range(len(step_frame_force_magnitude)), step_frame_force_magnitude)
            plt.title('Force Magnitude (N)')
            plt.show()
            
            # If left step
            if box['class'] == 1:
                left_steps[f'P{pass_id}_S{step_number}'] = {'rc_step_max':box['rc_step_max'], 'rc_CoP_x':box['rc_CoP_x'], 'rc_CoP_y':box['rc_CoP_y'], 'trisect_1':box['trisect_1'], 'trisect_2':box['trisect_2'], 'step_frame_force_magnitude':step_frame_force_magnitude}
            # If right step
            elif box['class'] == 2:
                right_steps[f'P{pass_id}_S{step_number}'] = {'rc_step_max':box['rc_step_max'], 'rc_CoP_x':box['rc_CoP_x'], 'rc_CoP_y':box['rc_CoP_y'], 'trisect_1':box['trisect_1'], 'trisect_2':box['trisect_2'],'step_frame_force_magnitude':step_frame_force_magnitude}
            # If incomplete step, do nothing
            else:
                pass 
    
    avg_right = align_and_average_heatmaps_padded(right_steps, alignment_threshold_kPa=1, reference_index=0) 
    avg_left = align_and_average_heatmaps_padded(left_steps, alignment_threshold_kPa=1, reference_index=0)
    

    
    for out_R in [avg_right, avg_left]:
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
            Rm_axes[i].set_xlabel("Width (px)")
            Rm_axes[i].set_ylabel("Length (px)")
        Rm_axes[len(R_step_keys)].imshow(overlap_mask, origin='upper',cmap=jet_cmap)  # same coordinate frame
        Rm_axes[len(R_step_keys)].set_title("Step Masks Overlapped")
        Rm_axes[len(R_step_keys)].set_xlabel("Width (px)")
        Rm_axes[len(R_step_keys)].set_ylabel("Length (px)")
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
            R_axes[i].set_xlabel("Width (px)")
            R_axes[i].set_ylabel("Length (px)")
        R_axes[len(R_step_keys)].imshow(avg_hm, origin='upper',cmap=jet_cmap)  # same coordinate frame
        R_axes[len(R_step_keys)].plot(avg_cx, avg_cy, linewidth=2)    # overlay avg CoP trajectory
        R_axes[len(R_step_keys)].set_title("Steps Average Heatmap")
        R_axes[len(R_step_keys)].set_xlabel("Width (px)")
        R_axes[len(R_step_keys)].set_ylabel("Length (px)")
        R_fig.tight_layout()
        plt.show()
        
        # Plotting the average magnitude of pressure throughout the duration of the step
        y_upper = out_R['avg_magnitude_curve'] + out_R['std_magnitude_curve']
        y_lower = out_R['avg_magnitude_curve'] - out_R['std_magnitude_curve']
        plt.plot(range(len(out_R['avg_magnitude_curve'])), out_R['avg_magnitude_curve'])
        plt.fill_between(range(len(out_R['avg_magnitude_curve'])), y_lower, y_upper, color='lightblue', alpha=0.5, label='Standard Deviation')
        plt.title("Average Force Magnitude (N)")
        plt.show()
        
    #"Foot Length (cm)": '245.3 \u00B1 4',
    # Getting data for average metrics table

    avg_step_duration = [] # In seconds
    std_step_duration = []
    avg_step_max_force = [] # In newtons
    std_step_max_force = []
    avg_CoP_distance = [] # In cm, each tile is .5 x .5 cm
    std_CoP_distance = []
    avg_contact_area = [] # In cm^2, each tile is .5 x .5 cm
    std_contact_area = []
    avg_step_length = [] # In cm, each tile is .5 x .5 cm
    std_step_length = []
    avg_step_width = [] # In cm, each tile is .5 x .5 cm
    std_step_width = []
    
    
    for side_steps in [left_steps, right_steps]:
        step_durations = [] 
        step_max_force = []
        CoP_distances = []
        for step_key in side_steps:
            # Getting the number of frames in each step
            step_durations.append(len(side_steps[step_key]['step_frame_force_magnitude'])/100) # frame count / sample rate = time in seconds
            # Getting the max pressure in each step
            step_max_force.append(max(side_steps[step_key]['step_frame_force_magnitude']))
            # Getting the distance of the CoP trajectory
            x = side_steps[step_key]['rc_CoP_x']
            y = side_steps[step_key]['rc_CoP_y']
            CoP_distances.append(np.linalg.norm(np.diff(np.c_[x, y], axis=0), axis=1).sum()*tile_size)
        
        # Getting average and standard deviation over all steps for the left or right side
        avg_step_duration.append(np.mean(np.array(step_durations)))
        std_step_duration.append(np.std(np.array(step_durations)))
        
        avg_step_max_force.append(np.mean(np.array(step_max_force)))
        std_step_max_force.append(np.std(np.array(step_max_force)))
        
        avg_CoP_distance.append(np.mean(np.array(CoP_distances)))
        std_CoP_distance.append(np.std(np.array(CoP_distances)))
    
    # Pulling more info from the align_and_average_heatmaps_padded outputs to get step geometry
    for step_masks in [avg_right, avg_left]:
        step_keys = step_masks['step_keys']
        contact_areas = []
        step_lengths = []
        step_widths = []
        arch_indexes = []
        
        
        # Going through each individual step mask and summing all true tiles to get area
        aligned_masks = step_masks['aligned_masks']
        aligned_trisects_list = step_masks['aligned_trisections']
        for step_key, aligned_mask, aligned_trisects in zip(step_keys, aligned_masks, aligned_trisects_list):

            # Calculating contact area in cm^2 and adding to contact areas list for averaging
            contact_area = np.sum(aligned_mask)*tile_size**2
            contact_areas.append(contact_area)
            
            # Calculating length in cm and adding to length list for averaging
            # Find rows containing at least one True
            rows_with_true = np.any(aligned_mask, axis=1)
            
            # Get the indices of these rows
            true_row_indices = np.where(rows_with_true)[0]
            
            # Extract the first and last row indices
            if true_row_indices.size > 0:
                first_true_row_index = true_row_indices[0]
                last_true_row_index = true_row_indices[-1]
                
                step_length = (last_true_row_index - first_true_row_index)*tile_size
            else:
                step_length = 0
            
            step_lengths.append(step_length)

            # Calculate width by only scanning above the first trisection and finding the longest continuous line

            # Calculating the arch index
            
            
            
            plt.imshow(aligned_mask, cmap=jet_cmap)
            plt.title(f'Step ID: {step_key} | Contact Area: {contact_area} cm\u00b2\nLength : {step_length} cm | Width : {777} cm')
            plt.hlines(aligned_trisects[0], 0, aligned_mask.shape[1]-1 )
            plt.hlines(aligned_trisects[1], 0, aligned_mask.shape[1]-1 )
            plt.show()
        
        avg_contact_area.append(np.mean(np.array(contact_areas)))
        std_contact_area.append(np.std(np.array(contact_areas)))
        
        avg_step_length.append(np.mean(np.array(step_lengths)))
        std_step_length.append(np.std(np.array(step_lengths)))
        
        
            
     # saving the metrics
    left_metrics = {
        "Step Duration (sec)": f'{avg_step_duration[0]:.2f}  \u00B1 {std_step_duration[0]:.2f}',
        "Contact Area (cm\u00b2)": f'{avg_contact_area[0]:.1f} \u00B1 {std_contact_area[0]:.1f}',
        "Foot Length (cm)": f'{ avg_step_length[0]:.1f} \u00B1 {std_step_length[0]:.1f}',
        "Foot Width (cm)": 'WIP',
        "Peak Pressure (kPa)": 'WIP',
        "Average Pressure (kPa)": 'WIP',
        "Maximum Force (N)": f'{avg_step_max_force[0]:.0f}  \u00B1 {std_step_max_force[0]:.0f}',
        "CoP Distance (cm)": f'{avg_CoP_distance[0]:.1f} \u00B1 {std_CoP_distance[0]:.1f}',
        "CoP Diplacement (cm)": 'WIP',
        "Walking Arch Index (%)": 'WIP',
        "CPEI (%)": 'WIP',
        "FPA (\u00b0)": 'WIP',
    }
    
    right_metrics = {
        "Step Duration (sec)": f'{avg_step_duration[1]:.2f}  \u00B1 {std_step_duration[1]:.2f}',
        "Contact Area (cm\u00b2)": f'{avg_contact_area[1]:.1f} \u00B1 {std_contact_area[1]:.1f}',
        "Foot Length (cm)": f'{ avg_step_length[1]:.1f} \u00B1 {std_step_length[1]:.1f}',
        "Foot Width (cm)": 'WIP',
        "Peak Pressure (kPa)": 'WIP',
        "Average Pressure (kPa)": 'WIP',
        "Maximum Force (N)": f'{avg_step_max_force[1]:.0f}  \u00B1 {std_step_max_force[1]:.0f}',
        "CoP Distance (cm)": f'{avg_CoP_distance[1]:.1f} \u00B1 {std_CoP_distance[1]:.1f}',
        "CoP Diplacement (cm)": 'WIP',
        "Walking Arch Index (%)": 'WIP',
        "CPEI (%)": 'WIP',
        "FPA (\u00b0)": 'WIP',
    }
    
    # Build table rows
    rows = []
    rows.append({"Foot": "Left Foot",  **left_metrics})
    rows.append({"Foot": "Right Foot", **right_metrics})
    
    # Build columns dynamically (order taken from first row’s keys)
    columns = [{"name": key, "id": key} for key in rows[0].keys()]       
        
    return "DONE", avg_left, avg_right, rows, columns, "tab-3"
        


# ---------- helpers ----------

# Function to resample the pressure magnitude of each step for averaging
def resample_pressure_magnitudes(arr, n_points=100):
    """Resample 1D array to fixed length using linear interpolation"""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, arr)



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
                    'step_frame_force_magnitude': 1D float array
                    'trisect_1':
                    'trisect_2':
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
      'aligned_trisections': list[aligned_trisect_1, aligned_trisect_2)] # y coords shifted with alignment
      'avg_cop':           {'x': 1D length=avg_cop_points, 'y': 1D length=avg_cop_points}
      'avg_magnitude_curve': avg_magnitude_curve, # avg pressure magnitude throughout step duration
      'std_magnitude_curve': std_magnitude_curve
    }
    """
    # Preserve insertion order of the dict
    step_keys = list(side_steps_dict.keys())

    # 0) Collect heatmaps and infer target canvas (largest H/W + small margin)
    H_list, W_list, heatmaps, cop_x_list, cop_y_list, trisections_list = [], [], [], [], [], []
    for k in step_keys:
        hm = side_steps_dict[k]['rc_step_max']
        H_list.append(hm.shape[0]); W_list.append(hm.shape[1])
        heatmaps.append(hm)
        trisections_list.append([side_steps_dict[k]['trisect_1'], side_steps_dict[k]['trisect_2']])
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

    # 3) Reference mask (already padded)
    ref_mask = padded_masks[reference_index]

    # 4) Align everyone to the reference (on masks), then shift heatmaps accordingly
    aligned_heatmaps, aligned_masks, aligned_trisections, shifts = [], [], [], []
    for i, (hm_pad, m_pad, trisect) in enumerate(zip(padded_heatmaps, padded_masks, trisections_list)):
        if i == reference_index or m_pad.sum() < 5:
            aligned_heatmaps.append(hm_pad)
            aligned_masks.append(m_pad)
            #aligned_trisections.append([trisect[0], trisect[1]])
            shifts.append((0, 0))
            continue
        dy, dx = phase_correlation_shift(ref_mask, m_pad)
        aligned_heatmaps.append(shift_with_nan(hm_pad, dy, dx))
        aligned_masks.append(shift_mask(m_pad, dy, dx))
        #aligned_trisect_1 = trisect[0] + dy
        #aligned_trisect_2 = trisect[1] + dy
        #aligned_trisections.append([aligned_trisect_1, aligned_trisect_2])
        shifts.append((dy, dx))

    # 5) Averages & overlaps
    avg_heatmap = nanmean_stack(aligned_heatmaps)
    mask_stack  = np.stack([m.astype(float) for m in aligned_masks], axis=0)
    avg_mask    = mask_stack.mean(axis=0)          
    overlap_mask = mask_stack.sum(axis=0) > 0

    # 6) Transform CoP traces and trisects into the same canvas & alignment
    padded_cop = []
    aligned_cop = []
    aligned_trisections = []

    for i, (cx, cy, trisect) in enumerate(zip(cop_x_list, cop_y_list, trisections_list)):
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

        
        # This is new
        aligned_trisect_1 = trisect[0] + r0 + dy
        aligned_trisect_2 = trisect[1] + r0 + dy
        aligned_trisections.append([aligned_trisect_1, aligned_trisect_2])
        
        

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


    # Resampling and averaging magnitudes of each step throughout the duration of the step
    # Resample each step to the same length
    resampled_force_magnitudes = [resample_pressure_magnitudes(val["step_frame_force_magnitude"], n_points=100)
                 for val in side_steps_dict.values()]
    
    resampled_force_magnitudes = np.vstack(resampled_force_magnitudes)
    
    # Compute mean and std across steps
    avg_magnitude_curve = np.mean(resampled_force_magnitudes, axis=0)
    std_magnitude_curve = np.std(resampled_force_magnitudes, axis=0)
    
    
    
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
        'aligned_trisections': aligned_trisections,
        'target_shape':     target_shape,      # handy to keep around
        'step_keys':        step_keys,          # maps list indices back to your dict keys
        'avg_magnitude_curve': avg_magnitude_curve, # avg pressure magnitude throughout step duration
        'std_magnitude_curve': std_magnitude_curve  # std ...
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
    
  
    # Finding all frames where the cumulative pressure exceeds the threshold
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


#####################
## TAB 4 CALLBACKS ##
#####################

# Callback to preview the PDF
@app.callback(
    Output("pdf-preview", "src"),
    Output("pdf-preview", "style"),
    Output("pdf-preview-note", "children"),
    Input("refresh-preview", "n_clicks"),
    Input("patient-info-store", "data"),
    Input("avg-steps-json", "data"),
    Input("avg-force-mag-json", "data"),
    Input("avg-metrics-table", "data"),
    State("fit-width", "value"),
    prevent_initial_call=False
)
def update_pdf_preview(_n_clicks, patient_info, fig_json, fig2_json, metrics_table, fit):
    print("TAB 4 STUFF!")
    print(patient_info)
    print()
    print(metrics_table)
    if not patient_info or not fig_json or not metrics_table:
        return dash.no_update, dash.no_update, "Waiting for patient info, plot, and metrics..."
    
    # Build the real PDF bytes
    try:
        pdf_bytes = build_pdf_bytes(
            patient_info=patient_info,
            fig_json=fig_json,
            fig2_json=fig2_json,
            metrics_table=metrics_table

        )
    except Exception as e:
        return dash.no_update, dash.no_update, f"Could not render preview: {e}"

    data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()

    # Style switching for fit-width
    fixed_style = {
        "width": "816px",
        "height": "1056px",
        "border": "1px solid #ccc",
        "background": "white",
        "boxShadow": "0 0 8px rgba(0,0,0,0.15)"
    }
    fluid_style = {
        "width": "100%",
        "height": "calc(100vh - 260px)",  # adjust for your header/controls
        "border": "1px solid #ccc",
        "background": "white"
    }
    style_out = fluid_style if ("fit" in (fit or [])) else fixed_style

    note = "Preview shows the actual PDF (Letter, 8.5×11 in) with real page breaks and margins."
    return data_url, style_out, note

def build_pdf_bytes(patient_info, fig_json, fig2_json, metrics_table):
    # Convert Plotly figure to PNG bytes (high-res)
    fig = go.Figure(**json.loads(fig_json))
    fig2 = go.Figure(**json.loads(fig2_json))
    

    # Match PDF font (Helvetica) and adjust sizes to harmonize
    fig.update_layout(
        font=dict(
            family="Helvetica",   # matches ReportLab default
            size=18,              # same as ReportLab Normal text
            color="black"
        ),
        
        title_font=dict(
            family="Helvetica-Bold",
            size=22,
            color="black"
        ),
        

    )
    png_bytes = fig.to_image(format="png", scale=1, engine="kaleido")
    
    
    # Match PDF font (Helvetica) and adjust sizes to harmonize
    fig2.update_layout(
        font=dict(
            family="Helvetica",   # matches ReportLab default
            size=18,              # same as ReportLab Normal text
            color="black"
        ),
        
        title_font=dict(
            family="Helvetica-Bold",
            size=22,
            color="black"
        ),
        

    )
    png2_bytes = fig2.to_image(format="png", scale=1, engine="kaleido")

    # ReportLab doc
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=0.25*inch, rightMargin=0.25*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    story = []

    # Header
    title = "Plantar Pressure Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.25*inch))
    

    available_width = doc.width  # from your SimpleDocTemplate/BaseDocTemplate
    patient_table = make_two_up_info_table(patient_info, doc_width=available_width, gutter=0.25*inch)
    story.append(patient_table)
    

    # Plot image
    img = Image(BytesIO(png_bytes)) # svg2rlg(BytesIO(png_bytes))
    img2 = Image(BytesIO(png2_bytes))
    """
    target_w, target_h = 8.0 * inch, 5.0 * inch
    sx = target_w / float(img.width)
    sy = target_h / float(img.height)
    img.width *= sx
    img.height *= sy
    for elem in img.contents:
        elem.scale(sx, sy)
    """
    img.drawHeight = 5 * inch
    img.drawWidth = 8 * inch
    img2.drawHeight = 8/3 * inch
    img2.drawWidth = 8 * inch
    story.append(img)
    story.append(img2)
    story.append(Spacer(1, 0.25*inch))
    
    # Metrics table
    story.append(Paragraph("<b>Summary Metrics</b>", styles["Heading3"]))
    table_header = list(metrics_table[0].keys())
    #table_header = [col["name"] for col in metrics_columns]
    table_rows = [list(side.values()) for side in metrics_table]
    print(table_header)
    print(table_rows)
    #table_rows = [[row.get(col["id"], "") for col in metrics_columns] for row in metrics_data]
    table_data = [table_header] + table_rows
    
    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.grey),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    
    
     # --------------- HEADER IMAGES ---------------
    # Define a small helper function to draw logos at corners
    def header_logos(canvas, doc):
        logo_size = 2.3 * inch  # logo width and height
        page_w, page_h = LETTER
    
        # Top-left corner
        canvas.drawImage(
            "assets/miraculous.png",     # e.g. "assets/pch_logo.png"
            x=doc.leftMargin,
            y=page_h - logo_size + 0.7*inch,   # slightly below top edge
            width=logo_size,
            height=logo_size,
            preserveAspectRatio=True,
            mask='auto'
        )
        """
        # Top-right corner
        canvas.drawImage(
            "assets/PCH_Logo.png",    
            x=page_w - doc.rightMargin - logo_size,
            y=page_h - logo_size + 0.7*inch,
            width=logo_size,
            height=logo_size,
            preserveAspectRatio=True,
            mask='auto'
        )
        """
        # --- Footer text ---
        footer_text = datetime.now().strftime("Report generated on %B %d, %Y %H:%M")
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(LETTER[0] / 2.0, 0.3 * inch, footer_text)

    
    doc.build(story, onFirstPage=header_logos, onLaterPages=header_logos)
    buf.seek(0)
    return buf.read()

# New 11/4
def make_two_up_info_table(info_dict, doc_width, gutter=0.25*inch,
                           font_name="Helvetica", font_size=9):
    """
    Create a 2-up (two items per row) patient info block with no visible grid.
    Each cell shows 'Label: Value' as a Paragraph, wrapping as needed.
    - info_dict: ordered dict-like (insertion order respected)
    - doc_width: available width on the page (use doc.width)
    - gutter: space between the two columns
    """
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "info_cell",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=font_size + 2,
        spaceBefore=0, spaceAfter=0
    )

    # Compute two equal column widths with a gutter in between
    col_w = (doc_width - gutter) / 2.0
    col_widths = [col_w, col_w]

    # Turn key/value pairs into rows of two cells
    items = list(info_dict.items())
    rows = []
    for i in range(0, len(items), 2):
        left_k, left_v = items[i]
        left_p = Paragraph(f"<b>{left_k}:</b> {'' if left_v is None else left_v}", cell_style)

        if i + 1 < len(items):
            right_k, right_v = items[i+1]
            right_p = Paragraph(f"<b>{right_k}:</b> {'' if right_v is None else right_v}", cell_style)
        else:
            right_p = Paragraph("", cell_style)

        rows.append([left_p, right_p])

    tbl = Table(rows, colWidths=col_widths, spaceBefore=0, spaceAfter=0)

    # Invisible table: zero paddings, no lines; keep text aligned & tidy
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
        # No GRID/BORDER rules → visually invisible
    ]))

    return tbl

# Callback to generate and download the PDF
@app.callback(
    Output("download-pdf", "data"),
    Input("generate-pdf", "n_clicks"),
    State("patient-info-store", "data"),
    State("figure-json-store", "data"),
    State("metrics-store", "data"),
    prevent_initial_call=True
)
def generate_pdf(n, patient_info, fig_json, metrics_payload):
    if not patient_info or not fig_json or not metrics_payload:
        return dash.no_update
    pdf_bytes = build_pdf_bytes(
        patient_info=patient_info,
        fig_json=fig_json,
        metrics_columns=metrics_payload["columns"],
        metrics_data=metrics_payload["data"]
    )
    filename = f"pressure_report_{patient_info.get('last_name','patient')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return dcc.send_bytes(pdf_bytes, filename)



if __name__ == "__main__":
    app.run(debug=False)
   