"""
DRishti AI — FastAPI Backend
Entry point: runs the full 10-stage pipeline on uploaded retinal images.
"""

import os
import uuid
import base64
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Pipeline modules ──────────────────────────────────────────────────────────
from pipeline.iqa             import assess_quality
from pipeline.anatomy         import detect_anatomy
from pipeline.lesion          import detect_lesions
from pipeline.neovascularization import analyze_neovascularization
from pipeline.grading         import grade_image
from pipeline.gradcam         import generate_gradcam
from ecde.engine              import run_ecde
from report.generator         import generate_report

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DRishti AI",
    description="AI-Assisted Diabetic Retinopathy Screening",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).parent
STATIC_DIR  = BASE_DIR / "static"
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main analysis endpoint.
    Accepts a retinal fundus image, runs the full pipeline, returns JSON results.
    """
    # ── Validate file type ─────────────────────────────────────────────────
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(allowed)}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img_bgr = _read_image(data)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid retinal photograph.")

    case_id     = str(uuid.uuid4())[:8].upper()
    original_b64 = _bgr_to_b64(img_bgr)

    results = {
        "case_id":      case_id,
        "filename":     file.filename,
        "original_b64": original_b64,
    }

    try:
        # ── Stage 1: Image Quality Assessment ─────────────────────────────
        iqa_result = assess_quality(img_bgr)
        results["iqa"] = iqa_result

        if not iqa_result["gradable"]:
            results["pipeline_stopped"] = True
            results["stop_reason"]      = iqa_result["reason"]
            # Still run ECDE to produce the "not gradable" decision
            results["ecde"] = run_ecde(iqa_result, {}, {}, {})
            return JSONResponse(results)

        results["pipeline_stopped"] = False

        # ── Stage 2: Anatomy Detection ────────────────────────────────────
        anatomy_result = detect_anatomy(img_bgr)
        results["anatomy"] = {
            "overlay_b64": anatomy_result["overlay_b64"],
            "findings":    anatomy_result["findings"],
            "optic_disc":  {
                "center": anatomy_result["optic_disc"]["center"],
                "radius": anatomy_result["optic_disc"]["radius"],
            },
            "fovea": {"center": anatomy_result["fovea"]["center"]},
        }
        od_center = anatomy_result["optic_disc"]["center"]
        od_radius = anatomy_result["optic_disc"]["radius"]

        # ── Stage 3: Lesion Detection ──────────────────────────────────────
        lesion_result = detect_lesions(img_bgr)
        results["lesion"] = {
            "overlay_b64":          lesion_result["overlay_b64"],
            "counts":               lesion_result["counts"],
            "findings":             lesion_result["findings"],
            "quadrant_hemorrhages": lesion_result["quadrant_hemorrhages"],
            "coordinates":          lesion_result.get("coordinates", {}),
        }

        # ── Stage 4: Neovascularization Analysis ──────────────────────────
        # Re-decode vessel mask for NV analysis
        import base64 as b64lib
        vessel_mask_data = anatomy_result["blood_vessels"]["mask_b64"]
        vessel_bytes     = b64lib.b64decode(vessel_mask_data)
        vessel_arr       = np.frombuffer(vessel_bytes, np.uint8)
        vessel_mask      = cv2.imdecode(vessel_arr, cv2.IMREAD_GRAYSCALE)

        nv_result = analyze_neovascularization(vessel_mask, od_center, od_radius or 30)
        results["neovascularization"] = nv_result

        # ── Stage 5: Disease Grading (ResNet-50 + EfficientNet-B4) ────────
        grading_result = grade_image(img_bgr)
        results["grading"] = {
            "resnet50": {
                "pred_label":  grading_result["resnet50"]["pred_label"],
                "confidence":  grading_result["resnet50"]["confidence"],
                "uncertainty": grading_result["resnet50"]["uncertainty"],
                "probs":       grading_result["resnet50"]["probs"],
                "mc_runs":     grading_result["resnet50"]["mc_runs"],
            },
            "efficientnet": {
                "pred_label":  grading_result["efficientnet"]["pred_label"],
                "confidence":  grading_result["efficientnet"]["confidence"],
                "uncertainty": grading_result["efficientnet"]["uncertainty"],
                "probs":       grading_result["efficientnet"]["probs"],
                "mc_runs":     grading_result["efficientnet"]["mc_runs"],
            },
            "ensemble_probs":      grading_result["ensemble_probs"],
            "ensemble_label":      grading_result["ensemble_label"],
            "ensemble_confidence": grading_result["ensemble_confidence"],
        }

        # ── Stage 6: ECDE ──────────────────────────────────────────────────
        ecde_result = run_ecde(iqa_result, grading_result, lesion_result, nv_result)
        results["ecde"] = ecde_result

        # ── Stage 7: Grad-CAM ──────────────────────────────────────────────
        try:
            gradcam_result = generate_gradcam(img_bgr, class_idx=ecde_result["final_grade_idx"] if ecde_result["final_grade_idx"] >= 0 else None)
            results["gradcam"] = {
                "overlay_b64": gradcam_result["overlay_b64"],
                "heatmap_b64": gradcam_result["heatmap_b64"],
                "class_label": gradcam_result["class_label"],
            }
        except Exception as gc_err:
            results["gradcam"] = {"error": str(gc_err)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(e)}",
        )

    return JSONResponse(results)


@app.post("/report")
async def get_report(file: UploadFile = File(...)):
    """Run analysis and return a printable HTML report."""
    data    = await file.read()
    img_bgr = _read_image(data)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    case_id      = str(uuid.uuid4())[:8].upper()
    original_b64 = _bgr_to_b64(img_bgr)

    iqa      = assess_quality(img_bgr)
    anatomy  = detect_anatomy(img_bgr)
    lesion   = detect_lesions(img_bgr)

    import base64 as b64lib
    vessel_bytes = b64lib.b64decode(anatomy["blood_vessels"]["mask_b64"])
    vessel_arr   = np.frombuffer(vessel_bytes, np.uint8)
    vessel_mask  = cv2.imdecode(vessel_arr, cv2.IMREAD_GRAYSCALE)

    nv       = analyze_neovascularization(vessel_mask, anatomy["optic_disc"]["center"])
    grading  = grade_image(img_bgr)
    ecde     = run_ecde(iqa, grading, lesion, nv)

    try:
        gradcam = generate_gradcam(img_bgr)
    except Exception:
        gradcam = {}

    html = generate_report(
        case_id, iqa, anatomy, lesion, nv, grading, ecde, gradcam, original_b64
    )
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "DRishti AI"}
