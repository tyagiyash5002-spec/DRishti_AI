# DRishti AI — Implementation Plan

## Stack
- **Backend**: Python + FastAPI
- **Frontend**: Single-page HTML/CSS/JS (no framework needed for demo)
- **AI Pipeline**: Simulated with real-looking outputs (demo mode)
- **Visualization**: OpenCV + Matplotlib for heatmaps/overlays
- **Report**: HTML → PDF via browser print / jsPDF

## Project Structure
```
DRishti_AI/
├── main.py                  # FastAPI app entry point
├── pipeline/
│   ├── iqa.py               # Image Quality Assessment
│   ├── anatomy.py           # Anatomy Detection (U-Net style)
│   ├── lesion.py            # Lesion Detection (FPN style)
│   ├── neovascularization.py# Vessel/NV Analysis
│   ├── grading.py           # ResNet-50 + EfficientNet-B4
│   ├── gradcam.py           # Grad-CAM visualization
│   └── ecde.py              # Evidence-Consistency Decision Engine
├── report/
│   └── generator.py         # Final report generation
├── static/
│   ├── index.html           # Main UI
│   ├── style.css            # Medical UI styling
│   └── app.js               # Frontend logic
├── uploads/                 # Temp uploaded images
├── outputs/                 # Generated overlays/heatmaps
└── requirements.txt
```
