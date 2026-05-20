# Miraculous – Plantar Pressure Analysis Tool

![Miraculous Logo](assets/miraculous.png)

Miraculous is an interactive, web-based application for analyzing plantar pressure data. Built with Python and Plotly Dash, it provides a full analysis pipeline — from raw pressure recordings through step identification, biomechanical metric computation, and PDF report generation — all within a clean, browser-based interface.

---

## Features

- **Animated pressure heatmap viewer** — Scrub through or play back frame-by-frame pressure recordings
- **Pass segmentation** — Define multiple walking passes from a single trial by selecting start/end frames
- **AI-powered step identification** — Automatically detects and classifies left foot, right foot, and incomplete steps using a fine-tuned YOLOv8 model
- **Manual bounding box editing** — Add, remove, drag, resize, and reclassify detected step bounding boxes directly on the heatmap
- **Single step analysis** — Isolate and inspect individual steps
- **Average step analysis** — Compute and visualize averaged pressure distributions and biomechanical metrics across a trial
- **Patient information management** — Record patient demographics, pathology, project, and clinical notes; add new dropdown options on the fly
- **PDF report generation** — Generate and download a formatted clinical PDF report with pressure visualizations, metrics, and patient information

---

## Repository Structure

```
.
├── miraculous_app.py          # Main application — all UI and logic
├── dropdown_values.csv        # Persistent pathology and project dropdown options
├── assets/
│   ├── miraculous.png         # App logo
│   └── PCH_Logo.png           # Institution logo (used in reports)
├── model_weights/
│   └── yolov8l_best.pt        # Trained YOLOv8-Large step identification model
├── S115_W1_P1.npz             # Sample trial data files
├── S120_W1.npz
├── S133_W1.npz
├── S140_W1.npz
├── S140_W1_1.npz
├── S145_W1.npz
├── S30_W1.npz
└── S33_W1.npz
```

Output directories are created automatically at runtime:
```
<trial_name>/
└── Pass<N>/
    ├── <trial>_Pass<N>_image.png        # Max-pressure heatmap image
    ├── <trial>_Pass<N>_predictions.csv  # Step bounding box predictions
    └── predict/                         # YOLO annotated output images
```

---

## Data Format

Miraculous reads trial data from `.npz` files. Each file should contain a single array (accessed as `arr_0`) with shape:

```
(num_frames, rows, cols)
```

where each frame is a 2D pressure matrix (pressure values in kPa). The application assumes a sample rate of **100 Hz** and a sensor tile size of **0.5 cm**.

Sample `.npz` files for several subjects are included in the repository root.

---

## Installation

**Requirements:** Python 3.8+

Install dependencies with pip:

```bash
pip install dash plotly numpy pandas matplotlib scipy scikit-learn \
            ultralytics reportlab svglib flask kaleido
```

> **Note:** `kaleido` is required for static image export used in PDF generation. Install it separately if needed:
> ```bash
> pip install kaleido
> ```

---

## Usage

1. **Configure the trial file** — In `miraculous_app.py`, set the `trial_file` variable near the top of the script to point to your `.npz` file:

   ```python
   trial_file = "S145_W1.npz"
   ```

2. **Run the application:**

   ```bash
   python miraculous_app.py
   ```

3. **Open your browser** and navigate to `http://127.0.0.1:8050`

---

## Workflow

The application is organized into five tabs:

| Tab | Description |
|---|---|
| **Pass Selection** | View the animated pressure heatmap, define walking passes by frame range, enter patient information, and save/process passes |
| **Step Identification** | Review and edit AI-generated step bounding boxes for each pass; classify steps as Left, Right, or Incomplete |
| **Single Step Analysis** | Inspect pressure data for an individual step *(in development)* |
| **Average Step Analysis** | View averaged pressure heatmaps, force magnitude curves, and summary biomechanical metrics across a trial |
| **Report Generation** | Preview and download a formatted PDF clinical report |

### Typical Analysis Flow

1. Load a trial, scrub the heatmap to identify walking passes, and define pass frame ranges in the **Pass Selection** tab
2. Fill in patient demographic and clinical information, then click **Save and Process Passes**
3. In the **Step Identification** tab, review the AI-detected bounding boxes; add, remove, or adjust them as needed
4. Click **Analyze Selected Step** for individual step inspection, or **Compute Average Metrics** for trial-level analysis
5. Navigate to **Report Generation** to preview and export the PDF report

---

## Model Weights

The step identification model (`model_weights/yolov8l_best.pt`) is a YOLOv8-Large network fine-tuned to detect plantar pressure footprints. It classifies each detected region as:

| Class ID | Label |
|---|---|
| 0 | Incomplete |
| 1 | Left |
| 2 | Right |

The model runs at an input size of 736 px with a confidence threshold of 0.25.

---

## Configuration

Pathology and project dropdown options are stored in `dropdown_values.csv` and loaded at startup. New options can be added directly from the UI and are automatically appended to this file.

```csv
Type,Value
Pathology,Cerebral Palsy
Pathology,Club Foot
Project,ACL Rehab Project
...
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `dash` | Web application framework |
| `plotly` | Interactive visualizations |
| `numpy` | Numerical array processing |
| `pandas` | Data handling |
| `matplotlib` | Heatmap image rendering |
| `scipy` | Image processing |
| `scikit-learn` | PCA and other analyses |
| `ultralytics` | YOLOv8 step identification |
| `reportlab` | PDF report generation |
| `svglib` | SVG-to-PDF conversion |
| `flask` | File serving |
| `kaleido` | Static Plotly image export |

---

## License

*Add your license here.*

---

## Acknowledgements

Developed at Phoenix Children's Hospital. Logo and branding assets located in the `assets/` directory.
