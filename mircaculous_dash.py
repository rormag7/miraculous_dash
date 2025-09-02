import dash
from dash import dcc, html, Input, Output, State
import dash_canvas
from dash_canvas import DashCanvas
import base64
import io
from PIL import Image
import plotly.express as px
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment

# Layout
app.layout = html.Div([
    html.Div([
        html.Img(
            src='assets/miraculous.png',
            style={
                'width': '400px',
                'display': 'block',
                'marginLeft': 'auto',
                'marginRight': 'auto',
                'paddingTop': '20px'
            }
        )
    ]),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Upload Folder with Pass Images"),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ', html.A('Select File')
                    ]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px'
                    },
                    accept='.png'
                ),

                html.Hr(),
                html.H5("Metadata Input"),
                dbc.Input(id='subject-id', placeholder='Subject ID', type='text', style={'marginBottom': '10px'}),
                dbc.Input(id='foot-side', placeholder='Foot (Left/Right)', type='text', style={'marginBottom': '10px'}),
                dbc.Input(id='notes', placeholder='Notes', type='text', style={'marginBottom': '10px'}),

                dbc.Button("Save Session", id='save-button', color='primary', style={'marginTop': '10px'}),
                html.Div(id='save-output', style={'marginTop': '10px', 'color': 'green'})
            ], width=4),

            dbc.Col([
                html.H4("Draw Bounding Box"),
                DashCanvas(
                    id='canvas',
                    width=512,
                    height=512,
                    scale=1,
                    lineWidth=2,
                    goButtonTitle='Apply',
                    hide_buttons=['zoom', 'pan', 'reset'],
                    tool='rectangle'
                ),
            ], width=8)
        ])
    ], fluid=True)
], style={
    'backgroundColor': 'white',
    'minHeight': '100vh',
    'padding': '0px 20px'
})

# Store image data between callbacks
@app.callback(
    Output('canvas', 'image_content'),
    Output('canvas', 'image_filename'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_canvas_image(contents, filename):
    if contents is None:
        return None, None
    return contents, filename

@app.callback(
    Output('save-output', 'children'),
    Input('save-button', 'n_clicks'),
    State('subject-id', 'value'),
    State('foot-side', 'value'),
    State('notes', 'value'),
    State('canvas', 'json_data'),
    prevent_initial_call=True
)
def save_data(n_clicks, subject_id, foot, notes, canvas_data):
    # Simulate saving to disk/database
    return f"Saved data for Subject: {subject_id}, Foot: {foot}, Notes: {notes}, Boxes: {len(canvas_data['objects']) if canvas_data else 0}"

if __name__ == '__main__':
    app.run(debug=True)
