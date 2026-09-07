"""
DRishti AI — Evidence-Consistency Decision Engine (ECDE)

Combines:
  • Image Quality Assessment
  • ResNet-50 prediction + MC Dropout uncertainty
  • EfficientNet-B4 prediction + MC Dropout uncertainty
  • Jensen-Shannon Divergence between model distributions
  • Lesion findings
  • Neovascularization findings
  • 4-2-1 Clinical Evidence Rule

Outputs a final ECDE decision: Auto-Grade / Human Review Required
"""

import numpy as np
from scipy.spatial.distance import jensenshannon


# ── DR Grade definitions ──────────────────────────────────────────────────────
DR_CLASSES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# ── Thresholds ────────────────────────────────────────────────────────────────
UNCERTAINTY_THRESHOLD = 0.15    # MC Dropout std > this → high uncertainty
JSD_THRESHOLD         = 0.10    # Jensen-Shannon divergence > this → model disagreement
CONFIDENCE_THRESHOLD  = 0.55    # model confidence < this → low confidence
EVIDENCE_MISMATCH_GAP = 2       # |predicted_grade - evidence_grade| > this → mismatch


# ── Jensen-Shannon Divergence ─────────────────────────────────────────────────

def compute_jsd(p: list, q: list) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    JSD ∈ [0, 1]; 0 = identical, 1 = maximally different.
    """
    p_arr = np.array(p, dtype=np.float64)
    q_arr = np.array(q, dtype=np.float64)

    # Normalise (guard against floating-point drift)
    p_arr = p_arr / (p_arr.sum() + 1e-12)
    q_arr = q_arr / (q_arr.sum() + 1e-12)

    jsd = float(jensenshannon(p_arr, q_arr) ** 2)   # scipy returns sqrt(JSD)
    return round(jsd, 6)


# ── 4-2-1 Clinical Evidence Rule ─────────────────────────────────────────────

def check_421_rule(lesion_result: dict) -> dict:
    """
    4-2-1 Severe NPDR evidence rule:
      • > 20 hemorrhages in each of 4 quadrants, OR
      • Venous beading in ≥ 2 areas (approximated by severe hemorrhage clustering), OR
      • Prominent IRMA in ≥ 1 area (approximated by soft exudate + hemorrhage density)

    Returns:
        {
          "severe_evidence_present": bool,
          "quadrant_hemorrhages":    list[int],
          "quadrants_with_20plus":   int,
          "venous_beading_proxy":    bool,
          "irma_proxy":              bool,
          "rule_triggered":          str or None
        }
    """
    quad_he = lesion_result.get("quadrant_hemorrhages", [0, 0, 0, 0])
    he_cnt  = lesion_result.get("counts", {}).get("hemorrhages", 0)
    se_cnt  = lesion_result.get("counts", {}).get("soft_exudates", 0)
    ma_cnt  = lesion_result.get("counts", {}).get("microaneurysms", 0)

    quads_20plus = sum(1 for q in quad_he if q > 20)
    # Proxy: venous beading ↔ severe hemorrhage asymmetry
    venous_beading = (max(quad_he) - min(quad_he) > 15 and he_cnt > 30)
    # Proxy: IRMA ↔ soft exudates + high MA count in same region
    irma           = (se_cnt > 5 and ma_cnt > 20)

    rule_4 = quads_20plus >= 4
    rule_2 = venous_beading
    rule_1 = irma

    severe_evidence = rule_4 or rule_2 or rule_1
    rule_triggered  = None
    if rule_4:
        rule_triggered = "4-quadrant hemorrhage rule"
    elif rule_2:
        rule_triggered = "Venous beading proxy"
    elif rule_1:
        rule_triggered = "IRMA proxy (soft exudate + MA)"

    return {
        "severe_evidence_present": severe_evidence,
        "quadrant_hemorrhages":    quad_he,
        "quadrants_with_20plus":   quads_20plus,
        "venous_beading_proxy":    venous_beading,
        "irma_proxy":              irma,
        "rule_triggered":          rule_triggered,
    }


# ── Evidence-grade estimation ─────────────────────────────────────────────────

def estimate_evidence_grade(lesion_result: dict, nv_result: dict) -> int:
    """
    Estimate a DR grade (0-4) purely from lesion/vessel evidence.
    Used to check consistency with the model-predicted grade.
    """
    ma   = lesion_result.get("counts", {}).get("microaneurysms", 0)
    he   = lesion_result.get("counts", {}).get("hemorrhages",    0)
    ex   = lesion_result.get("counts", {}).get("hard_exudates",  0)
    se   = lesion_result.get("counts", {}).get("soft_exudates",  0)
    nv   = nv_result.get("neovascularization_detected", False)

    # Proliferative DR: any NV
    if nv:
        return 4

    # Severe NPDR: many hemorrhages / soft exudates
    if he > 40 or se > 10:
        return 3

    # Moderate NPDR: moderate lesion load
    if he > 10 or ex > 15 or ma > 20:
        return 2

    # Mild NPDR: few microaneurysms
    if ma > 5 or he > 2:
        return 1

    return 0    # No DR


# ── Main ECDE ─────────────────────────────────────────────────────────────────

def run_ecde(iqa_result:    dict,
             grading_result: dict,
             lesion_result:  dict,
             nv_result:      dict) -> dict:
    """
    Run the Evidence-Consistency Decision Engine.

    Returns:
        {
          "final_grade":          str,
          "final_grade_idx":      int,
          "decision":             "AUTO_GRADE" | "HUMAN_REVIEW",
          "confidence_level":     "High" | "Moderate" | "Low",
          "review_reasons":       list[str],
          "jsd":                  float,
          "model_agreement":      str,
          "resnet_confidence":    float,
          "effnet_confidence":    float,
          "resnet_uncertainty":   float,
          "effnet_uncertainty":   float,
          "evidence_grade":       int,
          "evidence_grade_label": str,
          "evidence_mismatch":    bool,
          "rule_421":             dict,
          "clinical_note":        str,
        }
    """
    review_reasons = []

    # ── 1. Quality gate ───────────────────────────────────────────────────────
    if not iqa_result.get("gradable", True):
        return {
            "final_grade":          "Image Not Gradable",
            "final_grade_idx":      -1,
            "decision":             "HUMAN_REVIEW",
            "confidence_level":     "Low",
            "review_reasons":       ["Image quality insufficient for automated grading."],
            "jsd":                  None,
            "model_agreement":      "N/A",
            "resnet_confidence":    None,
            "effnet_confidence":    None,
            "resnet_uncertainty":   None,
            "effnet_uncertainty":   None,
            "evidence_grade":       None,
            "evidence_grade_label": None,
            "evidence_mismatch":    None,
            "rule_421":             None,
            "clinical_note":        "Image quality gate failed. Repeat acquisition recommended.",
        }

    # ── 2. Extract grading outputs ────────────────────────────────────────────
    rn   = grading_result["resnet50"]
    en   = grading_result["efficientnet"]

    rn_probs     = rn["probs"]
    en_probs     = en["probs"]
    rn_conf      = rn["confidence"]
    en_conf      = en["confidence"]
    rn_unc       = rn["uncertainty"]
    en_unc       = en["uncertainty"]
    rn_class     = rn["pred_class"]
    en_class     = en["pred_class"]

    ens_probs    = grading_result["ensemble_probs"]
    ens_class    = int(np.array(ens_probs).argmax())

    # ── 3. Jensen-Shannon Divergence ──────────────────────────────────────────
    jsd = compute_jsd(rn_probs, en_probs)
    model_agreement = (
        "High"     if jsd < 0.05 else
        "Moderate" if jsd < JSD_THRESHOLD else
        "Low"
    )

    if jsd >= JSD_THRESHOLD:
        review_reasons.append(
            f"Model disagreement detected (JSD={jsd:.4f}). "
            "ResNet-50 and EfficientNet-B4 predictions differ significantly."
        )

    # ── 4. Uncertainty check ──────────────────────────────────────────────────
    if rn_unc > UNCERTAINTY_THRESHOLD:
        review_reasons.append(
            f"ResNet-50 MC Dropout uncertainty high (σ={rn_unc:.4f}, "
            f"{rn['mc_runs']} runs)."
        )
    if en_unc > UNCERTAINTY_THRESHOLD:
        review_reasons.append(
            f"EfficientNet-B4 MC Dropout uncertainty high (σ={en_unc:.4f}, "
            f"{en['mc_runs']} runs)."
        )

    # ── 5. Confidence check ───────────────────────────────────────────────────
    min_conf = min(rn_conf, en_conf)
    if min_conf < CONFIDENCE_THRESHOLD:
        review_reasons.append(
            f"Low model confidence (min={min_conf:.2%}). Predictions unreliable."
        )

    # ── 6. 4-2-1 rule ────────────────────────────────────────────────────────
    rule_421 = check_421_rule(lesion_result)

    # ── 7. Evidence-consistency check ────────────────────────────────────────
    ev_grade     = estimate_evidence_grade(lesion_result, nv_result)
    ev_label     = DR_CLASSES[ev_grade]
    mismatch_gap = abs(ens_class - ev_grade)
    ev_mismatch  = mismatch_gap > EVIDENCE_MISMATCH_GAP

    if ev_mismatch:
        review_reasons.append(
            f"Evidence-prediction mismatch: model predicts {DR_CLASSES[ens_class]} "
            f"but lesion evidence suggests {ev_label} (gap={mismatch_gap})."
        )

    # ── 8. NV check for PDR ───────────────────────────────────────────────────
    nv_detected = nv_result.get("neovascularization_detected", False)
    if nv_detected and ens_class < 3:
        review_reasons.append(
            "Neovascularization indicators detected but grading model predicts "
            f"< Severe NPDR ({DR_CLASSES[ens_class]}). Clinical review recommended."
        )

    # ── 9. 4-2-1 vs predicted grade ──────────────────────────────────────────
    if rule_421["severe_evidence_present"] and ens_class < 3:
        review_reasons.append(
            f"4-2-1 rule evidence of Severe NPDR/PDR conflicts with predicted grade "
            f"({DR_CLASSES[ens_class]}). Review required."
        )

    # ── 10. Final decision ────────────────────────────────────────────────────
    decision = "HUMAN_REVIEW" if review_reasons else "AUTO_GRADE"

    confidence_level = (
        "High"     if (jsd < 0.05 and rn_unc < 0.08 and en_unc < 0.08
                       and min_conf > 0.70 and not ev_mismatch) else
        "Moderate" if not review_reasons else
        "Low"
    )

    # Clinical note
    clinical_note = (
        "AI grading is consistent and evidence-supported. "
        "Final clinical decision must be made by a qualified ophthalmologist."
        if decision == "AUTO_GRADE" else
        "One or more consistency checks failed. "
        "Human ophthalmological review is recommended before any clinical action."
    )

    return {
        "final_grade":          DR_CLASSES[ens_class],
        "final_grade_idx":      ens_class,
        "decision":             decision,
        "confidence_level":     confidence_level,
        "review_reasons":       review_reasons,
        "jsd":                  jsd,
        "model_agreement":      model_agreement,
        "resnet_confidence":    round(rn_conf, 4),
        "effnet_confidence":    round(en_conf, 4),
        "resnet_uncertainty":   round(rn_unc, 4),
        "effnet_uncertainty":   round(en_unc, 4),
        "evidence_grade":       ev_grade,
        "evidence_grade_label": ev_label,
        "evidence_mismatch":    ev_mismatch,
        "rule_421":             rule_421,
        "clinical_note":        clinical_note,
    }
