"""
DRishti AI — Report Generator
Produces a styled HTML report that can be printed to PDF from the browser.
"""

from datetime import datetime
import base64
import uuid


DR_GRADE_COLORS = {
    "No DR":             "#27ae60",
    "Mild NPDR":         "#f39c12",
    "Moderate NPDR":     "#e67e22",
    "Severe NPDR":       "#c0392b",
    "Proliferative DR":  "#8e44ad",
    "Image Not Gradable":"#7f8c8d",
    "Human Review Recommended": "#2980b9",
}

DECISION_COLORS = {
    "AUTO_GRADE":   "#27ae60",
    "HUMAN_REVIEW": "#e74c3c",
}

CONFIDENCE_COLORS = {
    "High":     "#27ae60",
    "Moderate": "#f39c12",
    "Low":      "#e74c3c",
}


def _img_tag(b64: str, alt: str = "", width: str = "100%") -> str:
    if not b64:
        return f'<div class="no-image">Image not available</div>'
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};border-radius:8px;">'


def generate_report(
    case_id:       str,
    iqa_result:    dict,
    anatomy_result: dict,
    lesion_result:  dict,
    nv_result:      dict,
    grading_result: dict,
    ecde_result:    dict,
    gradcam_result: dict,
    original_b64:   str,
) -> str:
    """
    Generate a complete HTML report.
    Returns HTML string.
    """
    now       = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    grade     = ecde_result.get("final_grade", "—")
    grade_col = DR_GRADE_COLORS.get(grade, "#2c3e50")
    decision  = ecde_result.get("decision", "—")
    dec_col   = DECISION_COLORS.get(decision, "#2c3e50")
    conf_col  = CONFIDENCE_COLORS.get(ecde_result.get("confidence_level", ""), "#2c3e50")

    rn = grading_result.get("resnet50",     {})
    en = grading_result.get("efficientnet", {})

    # Lesion counts
    counts = lesion_result.get("counts", {})
    findings = lesion_result.get("findings", {})

    review_reasons_html = ""
    for r in ecde_result.get("review_reasons", []):
        review_reasons_html += f'<li>{r}</li>'
    if not review_reasons_html:
        review_reasons_html = "<li>None — all consistency checks passed.</li>"

    rule421 = ecde_result.get("rule_421", {}) or {}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRishti AI — Screening Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f0f4f8; color:#2c3e50; font-size:14px; }}
  .report-wrapper {{ max-width:900px; margin:30px auto; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 4px 24px rgba(0,0,0,.12); }}
  .header {{ background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); color:#fff; padding:32px 40px; }}
  .header h1 {{ font-size:26px; font-weight:700; letter-spacing:1px; }}
  .header .subtitle {{ font-size:13px; opacity:.8; margin-top:4px; }}
  .header .meta {{ margin-top:18px; display:flex; gap:40px; flex-wrap:wrap; font-size:12px; opacity:.9; }}
  .header .meta span {{ display:flex; flex-direction:column; }}
  .header .meta b {{ font-size:14px; font-weight:600; margin-top:2px; }}
  section {{ padding:28px 40px; border-bottom:1px solid #ecf0f1; }}
  section:last-child {{ border-bottom:none; }}
  h2 {{ font-size:16px; font-weight:700; color:#1a1a2e; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #3498db; }}
  .grade-badge {{ display:inline-block; padding:10px 24px; border-radius:30px; font-size:20px; font-weight:700; color:#fff; background:{grade_col}; }}
  .decision-badge {{ display:inline-block; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:600; color:#fff; background:{dec_col}; margin-top:8px; }}
  .conf-badge {{ display:inline-block; padding:4px 12px; border-radius:14px; font-size:12px; font-weight:600; color:#fff; background:{conf_col}; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#f8f9fa; text-align:left; padding:10px 14px; font-weight:600; color:#555; border-bottom:2px solid #dee2e6; }}
  td {{ padding:9px 14px; border-bottom:1px solid #f0f0f0; }}
  tr:hover td {{ background:#fafafa; }}
  .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; }}
  .metric-card {{ background:#f8f9fa; border-radius:8px; padding:14px; text-align:center; }}
  .metric-card .label {{ font-size:11px; color:#888; text-transform:uppercase; letter-spacing:.5px; }}
  .metric-card .value {{ font-size:20px; font-weight:700; color:#2c3e50; margin-top:4px; }}
  .metric-card .unit  {{ font-size:11px; color:#aaa; }}
  .finding-tag {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; margin:3px; }}
  .found     {{ background:#d5f5e3; color:#1e8449; border:1px solid #a9dfbf; }}
  .not-found {{ background:#f8f9fa; color:#aaa; border:1px solid #dee2e6; }}
  .review-list {{ background:#fef9e7; border-left:4px solid #f39c12; padding:14px 18px; border-radius:6px; }}
  .review-list li {{ margin:4px 0; font-size:13px; }}
  .img-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .img-block {{ text-align:center; }}
  .img-block .cap {{ font-size:11px; color:#888; margin-top:6px; }}
  .no-image {{ background:#f8f9fa; border:2px dashed #dee2e6; border-radius:8px; padding:20px; text-align:center; color:#aaa; font-size:12px; }}
  .disclaimer {{ background:#eaf4ff; border:1px solid #aed6f1; border-radius:8px; padding:14px 18px; font-size:12px; color:#2471a3; }}
  .disclaimer strong {{ display:block; margin-bottom:4px; }}
  .footer {{ background:#1a1a2e; color:#aaa; text-align:center; padding:16px; font-size:11px; }}
  @media print {{
    body {{ background:#fff; }}
    .report-wrapper {{ box-shadow:none; margin:0; }}
  }}
</style>
</head>
<body>
<div class="report-wrapper">

  <!-- HEADER -->
  <div class="header">
    <h1>👁  DRishti AI — Diabetic Retinopathy Screening Report</h1>
    <div class="subtitle">AI-Assisted Clinical Decision Support System</div>
    <div class="meta">
      <span>Case ID<b>{case_id}</b></span>
      <span>Analysis Date/Time<b>{now}</b></span>
      <span>IQA Result<b>{"✓ Gradable" if iqa_result.get("gradable") else "✗ Not Gradable"}</b></span>
      <span>IQA Confidence<b>{iqa_result.get("confidence", 0):.1%}</b></span>
    </div>
  </div>

  <!-- OVERALL RESULT -->
  <section>
    <h2>Overall Result</h2>
    <div class="grade-badge">{grade}</div><br>
    <div class="decision-badge">
      {"⚠ Human Review Recommended" if decision=="HUMAN_REVIEW" else "✓ Automated Grade"}
    </div>&nbsp;
    <span class="conf-badge">{ecde_result.get("confidence_level","—")} Confidence</span>
    <p style="margin-top:14px;font-size:13px;color:#555;">{ecde_result.get("clinical_note","")}</p>
  </section>

  <!-- MODEL PREDICTIONS -->
  <section>
    <h2>Model Predictions</h2>
    <table>
      <tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>MC Dropout σ</th><th>MC Runs</th></tr>
      <tr>
        <td><strong>ResNet-50</strong></td>
        <td>{rn.get("pred_label","—")}</td>
        <td>{rn.get("confidence",0):.1%}</td>
        <td>{rn.get("uncertainty",0):.4f}</td>
        <td>{rn.get("mc_runs","—")}</td>
      </tr>
      <tr>
        <td><strong>EfficientNet-B4</strong></td>
        <td>{en.get("pred_label","—")}</td>
        <td>{en.get("confidence",0):.1%}</td>
        <td>{en.get("uncertainty",0):.4f}</td>
        <td>{en.get("mc_runs","—")}</td>
      </tr>
    </table>
  </section>

  <!-- RELIABILITY METRICS -->
  <section>
    <h2>Reliability Metrics (ECDE)</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="label">Jensen-Shannon Divergence</div>
        <div class="value">{ecde_result.get("jsd",0):.4f}</div>
        <div class="unit">Model Agreement: {ecde_result.get("model_agreement","—")}</div>
      </div>
      <div class="metric-card">
        <div class="label">ResNet Confidence</div>
        <div class="value">{(ecde_result.get("resnet_confidence") or 0):.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">EfficientNet Confidence</div>
        <div class="value">{(ecde_result.get("effnet_confidence") or 0):.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">ResNet Uncertainty (σ)</div>
        <div class="value">{(ecde_result.get("resnet_uncertainty") or 0):.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">EfficientNet Uncertainty (σ)</div>
        <div class="value">{(ecde_result.get("effnet_uncertainty") or 0):.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Evidence Grade (Lesions)</div>
        <div class="value" style="font-size:14px;">{ecde_result.get("evidence_grade_label","—")}</div>
        <div class="unit">Mismatch: {"Yes" if ecde_result.get("evidence_mismatch") else "No"}</div>
      </div>
    </div>
  </section>

  <!-- 4-2-1 RULE -->
  <section>
    <h2>4-2-1 Clinical Evidence Check (Severe NPDR)</h2>
    <table>
      <tr><th>Check</th><th>Result</th></tr>
      <tr><td>Quadrant hemorrhage counts</td><td>{rule421.get("quadrant_hemorrhages","—")}</td></tr>
      <tr><td>Quadrants with &gt;20 hemorrhages</td><td>{rule421.get("quadrants_with_20plus","—")}</td></tr>
      <tr><td>Venous beading proxy</td><td>{"Yes" if rule421.get("venous_beading_proxy") else "No"}</td></tr>
      <tr><td>IRMA proxy</td><td>{"Yes" if rule421.get("irma_proxy") else "No"}</td></tr>
      <tr><td><strong>Severe NPDR evidence triggered</strong></td>
          <td><strong>{"Yes — " + (rule421.get("rule_triggered") or "") if rule421.get("severe_evidence_present") else "No"}</strong></td></tr>
    </table>
  </section>

  <!-- LESION FINDINGS -->
  <section>
    <h2>Detected Lesions</h2>
    <table>
      <tr><th>Lesion Type</th><th>Detected</th><th>Count</th></tr>
      <tr><td>Microaneurysms</td>
          <td>{"✓" if findings.get("microaneurysms_detected") else "—"}</td>
          <td>{counts.get("microaneurysms",0)}</td></tr>
      <tr><td>Hemorrhages</td>
          <td>{"✓" if findings.get("hemorrhages_detected") else "—"}</td>
          <td>{counts.get("hemorrhages",0)}</td></tr>
      <tr><td>Hard Exudates</td>
          <td>{"✓" if findings.get("hard_exudates_detected") else "—"}</td>
          <td>{counts.get("hard_exudates",0)}</td></tr>
      <tr><td>Soft Exudates (Cotton-Wool Spots)</td>
          <td>{"✓" if findings.get("soft_exudates_detected") else "—"}</td>
          <td>{counts.get("soft_exudates",0)}</td></tr>
    </table>
  </section>

  <!-- ANATOMICAL FINDINGS -->
  <section>
    <h2>Anatomical Findings</h2>
    <div>
      {"".join([
        f'<span class="finding-tag found">Optic Disc Detected</span>'
        if anatomy_result.get("findings",{}).get("optic_disc_detected") else
        '<span class="finding-tag not-found">Optic Disc Not Detected</span>',
        '<span class="finding-tag found">Fovea Estimated</span>'
        if anatomy_result.get("findings",{}).get("fovea_estimated") else "",
        f'<span class="finding-tag found">Vessel Density {anatomy_result.get("findings",{}).get("vessel_density_percent",0):.1f}%</span>',
      ])}
    </div>
  </section>

  <!-- NEOVASCULARIZATION -->
  <section>
    <h2>Neovascularization Analysis</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="label">NV Detected</div>
        <div class="value" style="color:{'#e74c3c' if nv_result.get('neovascularization_detected') else '#27ae60'}">
          {"Yes" if nv_result.get("neovascularization_detected") else "No"}
        </div>
      </div>
      <div class="metric-card">
        <div class="label">NV Probability</div>
        <div class="value">{nv_result.get("nv_probability",0):.1%}</div>
      </div>
      <div class="metric-card">
        <div class="label">NVE Score</div>
        <div class="value">{nv_result.get("nve_score",0):.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">NVD Score</div>
        <div class="value">{nv_result.get("nvd_score",0):.4f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Vessel Tortuosity</div>
        <div class="value">{nv_result.get("tortuosity",0):.2f}</div>
      </div>
      <div class="metric-card">
        <div class="label">Vessel Density</div>
        <div class="value">{nv_result.get("vessel_density",0):.1f}%</div>
      </div>
    </div>
  </section>

  <!-- REVIEW REASONS -->
  <section>
    <h2>ECDE Review Reasons</h2>
    <div class="review-list">
      <ul>{review_reasons_html}</ul>
    </div>
  </section>

  <!-- VISUALIZATIONS -->
  <section>
    <h2>Visualizations</h2>
    <div class="img-grid">
      <div class="img-block">
        {_img_tag(original_b64, "Original Retinal Image")}
        <div class="cap">Original Retinal Image</div>
      </div>
      <div class="img-block">
        {_img_tag(gradcam_result.get("overlay_b64","") if gradcam_result else "", "Grad-CAM Overlay")}
        <div class="cap">Grad-CAM Attention Heatmap (ResNet-50 — {gradcam_result.get("class_label","") if gradcam_result else "N/A"})</div>
      </div>
      <div class="img-block">
        {_img_tag(anatomy_result.get("overlay_b64",""), "Anatomy Segmentation")}
        <div class="cap">Anatomy Segmentation (Optic Disc · Vessels · Fovea)</div>
      </div>
      <div class="img-block">
        {_img_tag(lesion_result.get("overlay_b64",""), "Lesion Detection Overlay")}
        <div class="cap">Lesion Detection (MA · HE · EX · SE)</div>
      </div>
    </div>
  </section>

  <!-- DISCLAIMER -->
  <section>
    <div class="disclaimer">
      <strong>⚠ Clinical Disclaimer</strong>
      DRishti AI is an <em>AI-assisted screening and clinical decision-support system</em>.
      It is intended to support, not replace, the judgment of a qualified ophthalmologist.
      All diagnoses and treatment decisions must be made by a licensed medical professional.
      This report should not be used as a standalone clinical document.
    </div>
  </section>

  <div class="footer">DRishti AI — AI-Assisted Diabetic Retinopathy Screening &nbsp;|&nbsp; {now}</div>

</div>
</body>
</html>"""

    return html
