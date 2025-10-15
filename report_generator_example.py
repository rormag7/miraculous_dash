# -*- coding: utf-8 -*-
"""
Report tab with live previews + true PDF preview + download.
Run:
    python report_generator_with_pdf_preview.py
"""

import json
import base64
from io import BytesIO
from datetime import datetime

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_table
import plotly.graph_objects as go
import plotly.io as pio

# ---- ReportLab imports ----
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# ---------------------------------------------------------
# Demo data sources (replace with your real Stores/outputs)
# ---------------------------------------------------------
def make_demo_figure():
    fig = go.Figure()
    fig.add_scatter(
        x=list(range(0, 101, 5)),
        y=[i**0.5 for i in range(0, 101, 5)],
        mode="lines",
        name="Pressure Magnitude",
    )
    fig.update_layout(
        title="Average Step Pressure Profile",
        xaxis_title="Step cycle (%)",
        yaxis_title="Pressure magnitude (a.u.)",
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=60),
    )
    return fig


demo_patient_info = {
    "first_name": "Alex",
    "last_name": "Smith",
    "dob": "2001-05-12",
    "gender": "Male",
    "assessment_date": "2025-10-10",
    "notes": "Baseline assessment."
}

demo_metrics_columns = [
    {"name": "Metric", "id": "metric"},
    {"name": "Left Foot", "id": "left"},
    {"name": "Right Foot", "id": "right"},
]
demo_metrics_data = [
    {"metric": "Foot Length (cm)", "left": "24.5 ± 0.3", "right": "24.6 ± 0.4"},
    {"metric": "Foot Width (cm)", "left": 9.8, "right": 10.2},
    {"metric": "Max Pressure (kPa)", "left": 312, "right": 298},
    {"metric": "Avg Pressure (kPa)", "left": 142, "right": 137},
    {"metric": "Max Force (N)", "left": 487, "right": 501},
    {"metric": "CPEI (%)", "left": 12.3, "right": 11.8},
    {"metric": "FPA (°)", "left": 7.4, "right": 8.1},
]


# ---------------------------------------------------------
# PDF builder (uses ReportLab + Plotly -> PNG via kaleido)
# ---------------------------------------------------------
def build_pdf_bytes(patient_info: dict, fig_json: str, metrics_columns, metrics_data):
    # Convert Plotly figure to PNG bytes (high-res)
    fig = go.Figure(**json.loads(fig_json))

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
        
        xaxis=dict(title_font=dict(family="Helvetica", size=18, color="black")),
        yaxis=dict(title_font=dict(family="Helvetica", size=18, color="black"))
    )
    png_bytes = fig.to_image(format="png", scale=1, engine="kaleido")

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
    story.append(Paragraph(datetime.now().strftime("Generated on %B %d, %Y %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    # Patient info block
    p_rows = [
        f"<b>Name:</b> {patient_info.get('first_name','')} {patient_info.get('last_name','')}",
        f"<b>DOB:</b> {patient_info.get('dob','')}",
        f"<b>Gender:</b> {patient_info.get('gender','')}",
        f"<b>Assessment Date:</b> {patient_info.get('assessment_date','')}",
        f"<b>Notes:</b> {patient_info.get('notes','') or '—'}",
    ]
    for row in p_rows:
        story.append(Paragraph(row, styles["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    # Plot image
    img = Image(BytesIO(png_bytes))
    img.drawHeight = 3.2 * inch
    img.drawWidth = 6.0 * inch
    story.append(Paragraph("<b>Average Step Pressure Profile</b>", styles["Heading3"]))
    story.append(img)
    story.append(Spacer(1, 0.25*inch))

    # Metrics table
    story.append(Paragraph("<b>Summary Metrics</b>", styles["Heading3"]))
    table_header = [col["name"] for col in metrics_columns]
    table_rows = [[row.get(col["id"], "") for col in metrics_columns] for row in metrics_data]
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

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------
# Dash app
# ---------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    [
        # Stores: wire these up from your real app
        dcc.Store(id="patient-info-store", data=demo_patient_info),
        dcc.Store(id="metrics-store", data={"columns": demo_metrics_columns, "data": demo_metrics_data}),
        dcc.Store(id="figure-json-store", data=make_demo_figure().to_json()),

        dcc.Tabs(
            id="tabs", value="tab-report",
            children=[
                dcc.Tab(label="Analysis", value="tab-analysis",
                        children=[html.Div("... your analysis UI here ...", style={"padding": "16px"})]),
                dcc.Tab(label="Report", value="tab-report",
                        children=[
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
                                             style={"color": "#666", "fontSize": "12px", "marginTop": "6px"}),
                                ],
                                style={"padding": "16px"}
                            )
                        ]),
            ]
        ),
    ]
)
"""
# -------------------------
# Preview callbacks (UI)
# -------------------------
@app.callback(
    Output("patient-info-preview", "children"),
    Input("patient-info-store", "data")
)
def show_patient_info(info):
    if not info:
        return html.Div("No patient information available.")
    rows = [
        f"Name: {info.get('first_name','')} {info.get('last_name','')}",
        f"DOB: {info.get('dob','')}",
        f"Gender: {info.get('gender','')}",
        f"Assessment Date: {info.get('assessment_date','')}",
        f"Notes: {info.get('notes','')}",
    ]
    return html.Div(
        [html.Div(r) for r in rows],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "12px",
            "background": "#f9f9f9",
            "maxWidth": "520px"
        }
    )


@app.callback(
    Output("plot-preview", "figure"),
    Input("figure-json-store", "data")
)
def show_plot(fig_json):
    if not fig_json:
        return go.Figure()
    return go.Figure(**json.loads(fig_json))


@app.callback(
    Output("metrics-preview", "columns"),
    Output("metrics-preview", "data"),
    Input("metrics-store", "data")
)
def show_metrics_table(metrics_payload):
    if not metrics_payload:
        return [], []
    return metrics_payload["columns"], metrics_payload["data"]
"""

# ---------------------------------------------------------
# True PDF preview (real pagination/margins)
# ---------------------------------------------------------
@app.callback(
    Output("pdf-preview", "src"),
    Output("pdf-preview", "style"),
    Output("pdf-preview-note", "children"),
    Input("refresh-preview", "n_clicks"),
    Input("patient-info-store", "data"),
    Input("figure-json-store", "data"),
    Input("metrics-store", "data"),
    State("fit-width", "value"),
    prevent_initial_call=False
)
def update_pdf_preview(_n_clicks, patient_info, fig_json, metrics_payload, fit):
    if not patient_info or not fig_json or not metrics_payload:
        return dash.no_update, dash.no_update, "Waiting for patient info, plot, and metrics..."

    # Build the real PDF bytes
    try:
        pdf_bytes = build_pdf_bytes(
            patient_info=patient_info,
            fig_json=fig_json,
            metrics_columns=metrics_payload["columns"],
            metrics_data=metrics_payload["data"]
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


# ---------------------------------------------------------
# Generate + download PDF (uses same builder as preview)
# ---------------------------------------------------------
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
    app.run(debug=True)
