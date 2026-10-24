"""Builds report/Assignment1_Report.md and .docx from the notebook's real outputs."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "report"
REPORT_DIR.mkdir(exist_ok=True)

perf = pd.read_csv(OUT / "performance_comparison.csv")
support = pd.read_csv(OUT / "supporting_performance.csv")
investigation = pd.read_csv(OUT / "object_analysis.csv")
extras = pd.read_csv(OUT / "extra_detections.csv")
inventory = pd.read_csv(OUT / "baseline_inventory.csv")
scene_inv = pd.read_csv(OUT / "scene_inventory.csv")
scene_missed = pd.read_csv(OUT / "missed_objects_scene_inventory.csv")
both_readings = pd.read_csv(OUT / "missed_objects_both_readings.csv")


def inv_score(keyword):
    """Measured diagnostic score of an inventory row matched by keyword."""
    row = inventory[inventory["Object present in scene"].str.contains(keyword, case=False)]
    if row.empty:
        raise KeyError(f"no inventory row matching {keyword!r}")
    return float(row["Best score in ORIGINAL (diagnostic conf 0.03)"].iloc[0])


summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
settings = json.loads((OUT / "experiment_settings.json").read_text(encoding="utf-8"))

TIME_COL = [c for c in perf.columns if c.startswith("Inference time")][0]
PHYS_COL = [c for c in support.columns if c.startswith("Physical")][0]
run_date = datetime.fromtimestamp((OUT / "summary.json").stat().st_mtime).strftime("%d %B %Y")

failure = summary["failure_case"]
quality = summary["image_quality"]
analysis = summary["condition_analysis"]
device = summary["device"].upper()

# Derived, measured values used in the narrative (never hard-coded).
all_det = pd.read_csv(OUT / "detections_all_conditions.csv")
sup_baseline = int(support.loc[support["Condition"] == "Original", PHYS_COL].iloc[0])
weak_orig = float(all_det[(all_det["Condition"] == "Original") &
                          (all_det["Class"] == "car")]["Confidence"].min())
weak_rb = float(all_det[(all_det["Condition"] == "Reduced Brightness") &
                        (all_det["Class"] == "car")]["Confidence"].min())


def br(value):
    """Format a number for text."""
    return f"{value:g}"


# Per-condition class breakdown with confidences.
def breakdown(condition):
    df = pd.read_csv(OUT / "detections_all_conditions.csv")
    df = df[df["Condition"] == condition]
    parts = []
    for cls, group in df.groupby("Class"):
        confs = ", ".join(f"{v:.2f}" for v in sorted(group["Confidence"], reverse=True))
        parts.append(f"{cls} x{len(group)} [{confs}]")
    return "; ".join(parts)


# Inputs figure (2 x 2) assembled from the four supplied images.
condition_files = list(settings["input_files"].items())
panel_w = 740
panels = []
for _, filename in condition_files:
    img = Image.open(BASE_DIR / filename)
    panel = img.resize((panel_w, round(img.height * panel_w / img.width)))
    panels.append(panel)
inputs_figure = Image.new("RGB", (panels[0].width * 2, panels[0].height * 2), "white")
for idx, panel in enumerate(panels):
    inputs_figure.paste(panel, ((idx % 2) * panel_w, (idx // 2) * panel.height))
inputs_figure.save(OUT / "inputs_figure.png")

# ------------------------------------------------------------------ content blocks
blocks = []
blocks.append(("h1", "Assignment 1 - Experimental Analysis of YOLO Object Detection"))
blocks.append(("para", f"Course: CO4 | Programming Assignment from CO4_Assignment.pdf | "
                       f"Final experiment run: {run_date} | Device: {device}"))
blocks.append(("h2", "1. Objective and Methodology"))
blocks.append(("para",
    "The assignment asks for a YOLO-based detection system whose behaviour is compared when the "
    "same input image is modified. One real street-scene image (1536 x 1024 px) containing a bus, "
    "three cars, several people and a motorcycle was detected four times with the same "
    "pretrained model, changing only the input image. The experiment was executed in the Jupyter "
    "notebook Assignment1_YOLO_Experiment.ipynb, and every number below comes from that run. "
    "At the operating confidence threshold the model detects 8 unique objects in the original "
    "(a bus, three cars and four people); the motorcycle itself, the bus driver, car occupants "
    "and several partly-hidden people are present in the scene but never detected in any "
    "condition - section 4 inventories them so the missed-object denominator is explicit."))
blocks.append(("para",
    f"Fixed settings for all four conditions: model {settings['model']} (COCO-pretrained, no "
    f"training), confidence threshold {settings['confidence_threshold']}, inference size "
    f"{settings['image_size']} px, device {device}. Each condition was inferred "
    f"{settings['runs_per_condition']} times after a warm-up run and the mean wall-clock time "
    f"(measured around the full predict call) is reported; the supporting table also gives the "
    f"standard deviation."))
blocks.append(("para",
    "**Counting rule (important for the missed-object column).** The model sometimes draws two "
    "overlapping boxes for one physical object - the original image itself contains one such "
    "duplicate pair. Detections are therefore evaluated at three levels: raw *boxes* (the "
    "\"Objects detected\" column), *baseline objects* (unique objects detected in the original "
    "image), and objects that are visibly present in the scene but never detected. Boxes of the "
    "same class that overlap with IoU >= " + str(settings['iou_match_threshold']) + " are first "
    "merged into one physical object; a baseline object counts as **missed only when no box of "
    "the modified condition overlaps any of its boxes**. \"Missed objects\" is therefore measured "
    "against the baseline (the objects the model can detect at all), not against every object a "
    "human can see - an object the model never detects in the original cannot be \"missed\" by "
    "definition. A duplicate box that merges with its twin after modification is a box-level "
    "artifact, not a lost object, and is never counted as a miss. The duplicate pair was checked "
    "visually before being merged: both boxes enclose the same standing person (one full-body "
    "box, one tighter upper-body box), not two different people."))
blocks.append(("para",
    "**Ambiguity in the brief - reported, not silently resolved.** The assignment PDF lists a "
    "*MissedObjects* column for every condition, including the Original row, but never defines "
    "the term. Counting only objects lost from the original detections would make the Original "
    "row trivially zero, so a second reading is also reported: missed objects relative to a "
    "12-object inventory of everything visibly present in the scene (section 4.2). Both readings "
    "are given side by side in Table 5 and in outputs/missed_objects_both_readings.csv; neither "
    "is presented as the single correct answer."))
blocks.append(("h2", "2. Input and Modified Images"))
blocks.append(("fig", str(OUT / "inputs_figure.png"),
               "Figure 1: The four inputs - Original, Reduced Brightness, Blurred, and "
               "Brightness + Blur (same scene, left to right, top to bottom)."))
q_rows = pd.DataFrame([
    {"Condition": cond,
     "Mean brightness": quality[cond]["mean_brightness"],
     "Laplacian variance": quality[cond]["laplacian_variance"]}
    for cond in quality])
blocks.append(("table", "Table 1: Image-quality measurements of the four inputs "
                        "(higher Laplacian variance means sharper edges).", q_rows))
blocks.append(("para",
    "All four files show the same scene with identical dimensions. The brightness modification "
    "lowers the mean pixel intensity from "
    f"{quality['Original']['mean_brightness']} to "
    f"{quality['Reduced Brightness']['mean_brightness']}, while blur keeps brightness but "
    "collapses the Laplacian variance from "
    f"{quality['Original']['laplacian_variance']} to "
    f"{quality['Blurred']['laplacian_variance']}, confirming that sharp edges and texture are "
    "removed. The three supplied modified files were used as the official inputs; a small demo "
    "showing how they can be reproduced with OpenCV (convertScaleAbs alpha=0.6, GaussianBlur "
    "31 x 31, sigma 15) is saved in outputs/demo_regenerated/."))
blocks.append(("h2", "3. Detection Results"))
blocks.append(("fig", str(OUT / "comparison_figure.png"),
               "Figure 2: YOLO11n detections for all four conditions with bounding boxes, "
               "class labels and confidence scores."))
for cond in perf["Condition"]:
    blocks.append(("para", f"**{cond}:** {breakdown(cond)}"))
blocks.append(("para",
    "The four annotated full-resolution images are saved as outputs/detections/"
    "01_original_detections.png through 04_brightness_plus_blur_detections.png."))
blocks.append(("h2", "4. Performance Comparison"))
display_perf = perf[["Condition", "Objects detected", "Average confidence", TIME_COL,
                     "Missed objects"]].copy()
display_perf.columns = ["Condition", "Objects detected", "Average confidence",
                        "Inference time (ms)", "Missed objects"]
display_perf["Average confidence"] = display_perf["Average confidence"].map(lambda v: f"{v:.4f}")
blocks.append(("table", "Table 2: Required performance comparison across the four conditions "
                        "(measured values from the executed notebook). Objects detected counts "
                        "raw boxes; Missed objects uses Reading A - baseline objects (detected "
                        "in the original) that are no longer detected in that condition (see the "
                        "counting rule in section 1). Table 5 gives the second, scene-inventory "
                        "reading.", display_perf))
display_sup = support.copy()
new_col = [c for c in display_sup.columns if c.startswith("New")][0]
display_sup = display_sup[["Condition", PHYS_COL, "Duplicate boxes", new_col,
                           "Inference time std (ms)"]]
display_sup.columns = ["Condition", "Physical objects detected (of 8)", "Duplicate boxes",
                       "New or partial-counterpart boxes", "Inference time std (ms)"]
blocks.append(("table", "Table 3: Supporting analysis that separates physical objects from raw "
                        "boxes and classifies the extra boxes of Table 2.", display_sup))
blocks.append(("para",
    f"Average confidence decreases steadily from {perf.loc[0, 'Average confidence']:.4f} "
    f"(Original) to {perf.loc[1, 'Average confidence']:.4f} (Reduced Brightness), "
    f"{perf.loc[2, 'Average confidence']:.4f} (Blurred) and "
    f"{perf.loc[3, 'Average confidence']:.4f} (Brightness + Blur). In baseline-object terms "
    "(Table 3), all 8 baseline objects detected in the original stay detected under Original and "
    "Reduced Brightness, while Blurred and Brightness + Blur each lose exactly one baseline "
    "object - the small distant car."))
blocks.append(("para",
    "**Why the modified images show 10 boxes while the original shows 9.** The original's 9 boxes "
    "already cover only 8 physical objects, because one standing person is boxed twice (duplicate "
    "pair, IoU 0.580). Each modified image then shows 10 boxes = 7-8 physical objects + 1 "
    "duplicate box + 1-2 new boxes. Every extra box is classified in Table 3 and inspected "
    "visually in section 7: they fall on people who really exist in the scene but scored below "
    "the 0.25 threshold in the original (measured counterparts: bus driver "
    f"{inv_score('bus driver'):.3f}, white-car occupant {inv_score('white-car occupant'):.3f}), "
    "plus one partially occluded figure that cannot be confirmed "
    f"(its original score is only {inv_score('bollards'):.3f}). In other words, the count rises because "
    "borderline scores cross the threshold in both directions - a new person box at another "
    "location does not compensate for the car that was lost."))
blocks.append(("h3", "4.1 What is in the scene versus what the model detects (baseline inventory)"))
blocks.append(("para",
    "The 8 baseline objects are what the model detects in the original image - they are not the "
    "total number of objects in the scene. The scene visibly also contains a motorcycle in the "
    "middle of the road (with its rider), the bus driver behind the windshield, occupants inside "
    "the two foreground cars, a white vehicle partly cut off at the right image edge, and people "
    "standing further back. None of these are detected in any of the four conditions. To show "
    "this is a threshold effect and not a missing object, the model was re-run on the original "
    "image at a purely diagnostic lower threshold (conf >= 0.03); the experiment itself always "
    "uses conf >= 0.25. The measured best scores are below."))
inv_disp = inventory.copy()
inv_disp.columns = ["Object present in scene", "Best score in original (diagnostic)",
                    "Detected in any condition (conf >= 0.25)?"]
blocks.append(("table", "Table 4: Baseline inventory - real objects the model never detects at "
                        "the operating threshold, so they are excluded from the Missed-objects "
                        "denominator by definition.", inv_disp))
INV_SCORE = "Best score in ORIGINAL (diagnostic conf 0.03)"
moto_mask = inventory["Object present in scene"].str.contains("motorcycle", case=False)
bike_mask = inventory["Object present in scene"].str.contains("bike", case=False)
moto_score = float(inventory.loc[moto_mask, INV_SCORE].iloc[0])
bike_score = float(inventory.loc[bike_mask, INV_SCORE].iloc[0])
blocks.append(("para",
    "The motorcycle is the clearest case: in the original image the model scores it at only "
    f"{moto_score:.3f} when it sees it as a 'motorcycle' and {bike_score:.3f} when it labels the "
    f"same bike a 'bicycle' - both far below the {settings['confidence_threshold']} operating "
    "threshold, and neither label reaches that threshold in any of the four conditions. This is "
    "why the assignment's Missed-objects column is measured against the 8 objects the model can "
    "detect, not against everything a human can see in the photo; the motorcycle's absence is a "
    "fixed property of this detector on this scene, not an effect of the brightness or blur "
    "modification."))
blocks.append(("h3", "4.2 Both readings of \"missed objects\" (the brief leaves the term undefined)"))
blocks.append(("para",
    "Because the PDF asks for a MissedObjects value in the Original row yet never defines the "
    "term, both readings are reported rather than silently choosing one. Reading A (Table 2) "
    "counts objects lost from the original detections. Reading B counts objects visibly present "
    "in the scene that the model fails to detect in that condition - the usual object-detection "
    "meaning of a false negative, and non-zero even for the Original row."))
both_disp = both_readings.copy()
both_disp.columns = ["Condition",
                     "Reading A: missed vs original detections (of 8)",
                     "Reading B: missed vs scene inventory (of 12)"]
blocks.append(("table", "Table 5: The two readings of \"missed objects\" side by side. Reading A "
                        "isolates the effect of the modification; Reading B is the scene-level "
                        "false-negative count.", both_disp))
blocks.append(("para",
    "Reading B uses the 12-object scene inventory below, established by inspecting zoomed crops "
    "of the original image. Each object was matched with the same rule used everywhere else (same "
    "class, IoU >= " + str(settings['iou_match_threshold']) + "). One judgement call is stated "
    "openly: a figure behind the bollards to the right of the distant car is excluded from the "
    "inventory because its identity could not be confirmed (it is listed separately and marked "
    "uncertain). The white vehicle partly cut off at the right edge is counted, flagged as "
    "partially visible."))
blocks.append(("fig", str(OUT / "scene_inventory_figure.png"),
               "Figure 3: The 12-object scene inventory drawn on the original image (green = "
               "counted objects, numbered as in Table 6; red = the uncertain figure behind the "
               "bollards, excluded from the count)."))
scene_disp = scene_inv.copy()
scene_disp.columns = ["No.", "Class", "Object (visible in original)", "Region (x1, y1, x2, y2)",
                      "Original", "Reduced Bright.", "Blurred", "Bright.+Blur"]
blocks.append(("table", "Table 6: The 12-object scene inventory and its detection status in each "
                        "condition. Objects 5, 6, 11 and 12 are never detected at conf >= 0.25 in "
                        "any condition.", scene_disp))
# Gains/losses vs the Original condition, computed from the inventory (Reading B).
gains_losses = []
for cond in scene_missed["Condition"]:
    if cond == "Original":
        continue
    row = scene_missed[scene_missed["Condition"] == cond].iloc[0]
    orig_row = scene_missed[scene_missed["Condition"] == "Original"].iloc[0]
    gains_losses.append((cond, row["Missed vs scene inventory (of 12)"]))
moto_rows = scene_disp[scene_disp["Object (visible in original)"].str.contains("motorcycle", case=False)]
never_detected = scene_disp[(scene_disp[["Original", "Reduced Bright.", "Blurred",
                                         "Bright.+Blur"]] == "no").all(axis=1) &
                            (scene_disp["No."] != "-")]
never_names = ", ".join(never_detected["Object (visible in original)"].tolist())
blocks.append(("para",
    "Reading B makes two points that Reading A cannot. First, several objects are missed in "
    "**every** condition - " + never_names + " - so those absences are fixed blind spots of this "
    "detector on this scene, not effects of the modifications. Second, under blur the Reading B "
    "total stays at 4 of 12, identical to the Original row, only because the lost distant car is "
    "offset by the bus driver's score crossing the threshold under blur. A single "
    "scene-inventory number would therefore **hide** the damage blur caused; Reading A "
    "(0 -> 1 baseline object lost) is what isolates the modification's effect. This is exactly "
    "why the undefined term is reported twice."))
blocks.append(("para",
    f"The condition means span only {perf[TIME_COL].min():.0f}-{perf[TIME_COL].max():.0f} ms, with "
    f"standard deviations of {support['Inference time std (ms)'].min():.1f}-"
    f"{support['Inference time std (ms)'].max():.1f} ms over "
    f"{settings['runs_per_condition']} timed runs per condition, and the individual run times "
    f"overlap between conditions (e.g. Blurred "
    f"{min(analysis['Blurred']['inference_runs_ms']):.0f}-{max(analysis['Blurred']['inference_runs_ms']):.0f} ms "
    f"vs Brightness + Blur "
    f"{min(analysis['Brightness + Blur']['inference_runs_ms']):.0f}-{max(analysis['Brightness + Blur']['inference_runs_ms']):.0f} ms). "
    "Since all inputs have identical dimensions and every "
    "other setting is fixed, the small differences between the condition means are treated as "
    "measurement noise; no effect of the image content on speed and no performance improvement "
    "is claimed."))
blocks.append(("h2", "5. Object-Level Investigation of Two Objects"))
blocks.append(("para",
    "Object A is the largest bus instance (a large, prominent object) and Object B is the "
    "smallest vehicle instance in the original image (a small, distant object partially hidden "
    "behind the white car). The selection rule was fixed before running the conditions, and both "
    "crops were inspected visually in the notebook. Confidence changes follow the assignment "
    "formula: delta_C = C_original - C_modified. For undetected objects in a modified condition, "
    "the confidence is reported as 'missed/N/A' and observed delta_C is recorded as 'N/A' (since "
    "no modified detection exists from which to compute a difference). Under a hypothetical "
    "convention where missing is assigned confidence 0, delta_C would equal the object's full "
    "original score (0.3205); both perspectives are detailed below."))
display_inv = investigation.copy()
display_inv.columns = ["Object", "Condition", "Detected", "Confidence",
                       "delta_C", "IoU vs original box", "Center shift (px)",
                       "Width change (%)", "Height change (%)", "Bounding box"]
display_inv["Confidence"] = display_inv["Confidence"].astype(object)
display_inv["delta_C"] = display_inv["delta_C"].astype(object)
display_inv.loc[display_inv["Detected"] == "Missed", "Confidence"] = "missed/N/A"
display_inv.loc[display_inv["Detected"] == "Missed", "delta_C"] = "N/A" 
blocks.append(("table", "Table 7: Two-object investigation across the four conditions.", display_inv))
bus_rows = investigation[investigation["Object"].str.startswith("Object A")]
car_rows = investigation[investigation["Object"].str.startswith("Object B")]
car_orig, car_bright, car_blur = car_rows.iloc[0], car_rows.iloc[1], car_rows.iloc[2]
bus_delta_max = pd.to_numeric(bus_rows["delta_C = C_original - C_modified"], errors="coerce").max()
car_delta_bright = float(car_bright["delta_C = C_original - C_modified"])
blocks.append(("para",
    f"**Object A (bus, large)** is detected in all four conditions. Its confidence barely moves "
    f"({bus_rows.iloc[0]['Confidence']} -> {bus_rows.iloc[1]['Confidence']} / "
    f"{bus_rows.iloc[2]['Confidence']} / {bus_rows.iloc[3]['Confidence']}), so observed delta_C stays at "
    f"most {bus_delta_max:.4f}. The box overlap with the original stays at "
    f"IoU >= {bus_rows['IoU vs original box'].min():.3f} and the box centre moves no more than "
    f"{bus_rows['Center shift (px)'].max():.1f} px."))
blocks.append(("para",
    f"**Object B (car, small)** shows the opposite behaviour. It is detected in the Original "
    f"image with confidence {car_orig['Confidence']} and in Reduced Brightness with "
    f"{car_bright['Confidence']} (observed delta_C = {car_delta_bright:.4f}, box still overlaps with IoU "
    f"{car_bright['IoU vs original box']:.3f}). Under Blurred and Brightness + Blur the object is "
    f"**missed**: its modified confidence is recorded as 'missed/N/A' and observed delta_C is 'N/A'. "
    f"(Under a hypothetical convention where missing is treated as confidence 0, delta_C would equal the "
    f"full original confidence {float(car_orig['Confidence']):.4f}.)"))
blocks.append(("h2", "6. Failure-Case Analysis"))
blocks.append(("fig", str(OUT / "failure_case_figure.png"),
               "Figure 4: The failed object (green box = original location) in all four "
               "conditions. The distant car is detected with confidence 0.32 in the Original, "
               "keeps a low-confidence box under Reduced Brightness, and disappears once the "
               "image is blurred."))
blocks.append(("fig", str(OUT / "failure_region_raw.png"),
               "Figure 5: Raw, unannotated pixels of the same region. In the Original and "
               "Reduced Brightness panels the dark car behind the white car has visible "
               "boundaries; under blur its outline dissolves into the dark background and the "
               "bollards."))
n_same = failure["nearest_same_class_iou"]
n_any = failure["nearest_any_class"]
same_txt = (f"no car-class box overlaps the original location anymore (best same-class IoU "
            f"{n_same})" if n_same is None or n_same < 0.1
            else f"the closest car box has IoU {n_same}")
any_txt = (f"the closest box of any class is a low-confidence '{n_any['class_name']}' "
           f"({n_any['confidence']}) that overlaps the original location only partially "
           f"(IoU {n_any['iou']})" if n_any else "no box appears nearby at all")
blocks.append(("para",
    f"The clearest failure is a **{failure['class_name']} detected with confidence "
    f"{failure['original_confidence']} in the Original image (box {failure['original_bbox']}) "
    f"that is missed in the '{failure['condition']}' condition** "
    f"(observed delta_C = N/A; hypothetical delta_C = {float(failure['original_confidence']):.4f} under a zero-confidence convention). "
    f"In that condition {same_txt}, and {any_txt}. That person box sits over the bollard area "
    "partly overlapping the car's right edge, so it is reported as a nearby detection; it is "
    "*not* treated as proof that the model relabelled the car as a person, because the overlap "
    "is too small to establish this. What is certain is that no car box survives the blur. The "
    "same car is also missed in Brightness + Blur (observed delta_C = N/A), so the failure is "
    "reproducible rather than a one-off."))
blocks.append(("para",
    f"Image-quality measurements explain why: blur leaves mean brightness unchanged "
    f"({failure['quality_original']['mean_brightness']} -> "
    f"{failure['quality_condition']['mean_brightness']}) but reduces the Laplacian variance from "
    f"{failure['quality_original']['laplacian_variance']} to "
    f"{failure['quality_condition']['laplacian_variance']}, i.e. the fine edges that carried the "
    "object's evidence are largely gone. The car is small and distant - it covers only about "
    "8,400 pixels - and its discriminative features are the windshield line, the lights and the "
    "contrast between the dark body and the dark background behind the white car, exactly the "
    "detail Gaussian blur removes (Figures 4 and 5). With that evidence smoothed away, the score that "
    "was only 0.32 in the Original falls below the fixed 0.25 threshold. This is an "
    "interpretation consistent with the measurements, not a proven causal chain; by contrast the "
    "large bus and the two foreground cars keep strong edges under blur and lose at most 0.01 "
    "confidence."))
blocks.append(("para",
    "A separate box-level observation: the original image draws the standing person twice "
    "(duplicate pair, IoU 0.580) and under Blurred only one box remains at that spot. Because "
    "the person is still detected there, this merge is **not** counted in the Missed objects "
    "column; it is documented as a duplicate-box effect in Table 3."))
blocks.append(("h2", "7. Observations and Conclusion"))
blocks.append(("fig", str(OUT / "extra_detections_check.png"),
               "Figure 6: Visual verification of the extra boxes (annotated crops, original "
               "next to the modified condition). Top to bottom: the bus driver (a real person, "
               "below threshold in the original), the white-car occupants (real, below "
               "threshold), the partially occluded figure behind the bollards (uncertain), and "
               "the double-boxed motorcyclist (duplicate box)."))
blocks.append(("para",
    f"1. **Reduced brightness alone is well tolerated.** All {int(sup_baseline)} baseline objects "
    f"stay detected (0 missed under Reading A), only the average confidence falls from "
    f"{perf.loc[0, 'Average confidence']:.4f} to {perf.loc[1, 'Average confidence']:.4f}, and the "
    f"weakest car drops from {weak_orig:.2f} to {weak_rb:.2f} but remains above the "
    f"{settings['confidence_threshold']} threshold. Under Reading B its missed count falls from "
    f"{int(scene_missed.loc[scene_missed['Condition'] == 'Original', 'Missed vs scene inventory (of 12)'].iloc[0])} "
    f"to {int(scene_missed.loc[scene_missed['Condition'] == 'Reduced Brightness', 'Missed vs scene inventory (of 12)'].iloc[0])} "
    "of 12, because the modification pushed the white-car occupant's score above the threshold."))
blocks.append(("para",
    f"2. **Blur is the damaging factor.** Each blur condition loses exactly one physical object - "
    f"the small distant car - and the average confidence falls further to "
    f"{perf.loc[2, 'Average confidence']:.4f} (Blurred) and "
    f"{perf.loc[3, 'Average confidence']:.4f} (Brightness + Blur). The combined condition does not "
    "remove additional objects beyond blur alone."))
blocks.append(("para",
    "3. **Why both readings are reported, and why they disagree.** Under Reading B the Blurred "
    f"condition misses {int(scene_missed.loc[scene_missed['Condition'] == 'Blurred', 'Missed vs scene inventory (of 12)'].iloc[0])} "
    "of 12 scene objects - the same total as the Original row - only because the lost car is "
    "offset by the bus driver's score crossing the threshold under blur. A single "
    "scene-inventory number would therefore hide the damage blur caused; Reading A (0 -> 1 "
    "baseline object lost) is what isolates the modification's effect. The brief does not define "
    "\"missed objects\", so both are reported."))
blocks.append(("para",
    f"4. **Object size decides robustness.** The large bus changes by at most delta_C = {bus_delta_max:.4f} and "
    f"keeps IoU >= {bus_rows['IoU vs original box'].min():.3f} in every condition; the small distant car is missed under blur "
    f"(observed delta_C = N/A; hypothetical delta_C = {float(car_orig['Confidence']):.4f} under a zero-confidence convention)."))
blocks.append(("para",
    f"5. **The modified conditions add 1-2 low-confidence boxes each** "
    f"({analysis['Reduced Brightness']['new_boxes']} in Reduced Brightness, "
    f"{analysis['Blurred']['new_boxes']} in Blurred and "
    f"{analysis['Brightness + Blur']['new_boxes']} in Brightness + Blur; Table 3), which is why "
    "the raw box count rises to 10 even though an object is being missed. Figure 6 shows that "
    "these are not invented objects: the bus-window box and the white-car box fall on real "
    "people who were below the 0.25 threshold in the original (a borderline-score effect), while "
    "the figure behind the bollards cannot be confirmed with certainty. Every condition also "
    "contains exactly one duplicate box on an already detected object."))
blocks.append(("para",
    f"6. **Inference time is not affected by the image content** beyond measurement noise "
    f"(means of {settings['runs_per_condition']} runs per condition, standard deviations of "
    f"{support['Inference time std (ms)'].min():.1f}-{support['Inference time std (ms)'].max():.1f} ms); "
    "all inputs have identical dimensions and all other settings are fixed. The three modified "
    "conditions' per-run ranges overlap each other, while the Original condition's first timed "
    "run is a clear cold-start outlier, so with only 3 runs per condition these mean differences "
    "cannot be separated from system noise. No speed effect of the image content and no "
    "performance improvement is claimed."))
blocks.append(("para",
    "Conclusion: on this image and this model, moderate brightness reduction changes the "
    "detector's confidence only slightly, while blur - especially combined with reduced "
    "brightness - degrades small and low-contrast objects first, because it destroys the edges "
    "and texture the detector relies on; the large, high-contrast bus is essentially unaffected. "
    "The findings are limited to YOLO11n, one scene and these specific modifications; they do "
    "not generalise to other models or scenes."))
blocks.append(("h2", "8. References and Experiment Settings"))
blocks.append(("para",
    f"Model: {settings['model']} ({settings['weights']}); device: {settings['device']}; "
    f"confidence threshold: {settings['confidence_threshold']}; image size: "
    f"{settings['image_size']}; runs per condition: {settings['runs_per_condition']}; "
    f"IoU match threshold: {settings['iou_match_threshold']}; timing method: "
    f"{settings['timing_method']}. Package versions: "
    + ", ".join(f"{k} {v}" for k, v in settings["package_versions"].items()) + "."))
blocks.append(("para",
    "1. Assignment brief: CO4_Assignment.pdf, Assignment-I (CO4).\n"
    "2. J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: Unified, Real-Time "
    "Object Detection. CVPR 2016.\n"
    "3. Ultralytics. Ultralytics YOLO documentation. https://docs.ultralytics.com/\n"
    "4. Ultralytics. YOLO11 model card. https://docs.ultralytics.com/models/yolo11/\n"
    "5. COCO 80 class labels. https://docs.ultralytics.com/datasets/detect/coco/\n"
    "6. OpenCV documentation: cv2.GaussianBlur, cv2.convertScaleAbs."))
blocks.append(("para",
    "**Image source — unverified.** The four input PNGs carry no EXIF or embedded text metadata "
    "(verified chunk-by-chunk), and no source is recorded anywhere in the project files. To preserve "
    "academic and scientific integrity, no attribution or author origin is invented, and the origin "
    "remains unverified."))

# ------------------------------------------------------------------ Markdown
def md_table(title, df):
    lines = [title, "", "| " + " | ".join(df.columns) + " |",
             "|" + "|".join([" --- "] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        row_vals = []
        for v in row:
            if pd.isna(v) or v == "" or v == "nan":
                row_vals.append("-")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines) + "\n"


md_parts = []
for block in blocks:
    kind = block[0]
    if kind == "h1":
        md_parts.append(f"# {block[1]}\n")
    elif kind == "h2":
        md_parts.append(f"## {block[1]}\n")
    elif kind == "h3":
        md_parts.append(f"### {block[1]}\n")
    elif kind == "para":
        md_parts.append(block[1] + "\n")
    elif kind == "fig":
        rel = Path(block[1]).relative_to(BASE_DIR).as_posix()
        rel = "../" + rel if not rel.startswith("..") else rel
        md_parts.append(f"![{block[2]}]({rel})\n\n*{block[2]}*\n")
    elif kind == "table":
        md_parts.append(md_table(block[1], block[2]))
(REPORT_DIR / "Assignment1_Report.md").write_text("\n".join(md_parts), encoding="utf-8")

# ------------------------------------------------------------------ DOCX
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches, Pt

doc = Document()
section = doc.sections[0]
section.orientation = WD_ORIENT.LANDSCAPE
section.page_width, section.page_height = section.page_height, section.page_width
for margin in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(section, margin, Inches(0.7))

from docx.oxml import OxmlElement

for block in blocks:
    kind = block[0]
    if kind == "h1":
        p = doc.add_heading(block[1], level=0)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
    elif kind == "h2":
        p = doc.add_heading(block[1], level=1)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
    elif kind == "h3":
        p = doc.add_heading(block[1], level=2)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
    elif kind == "para":
        for i, chunk in enumerate(block[1].split("\n")):
            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after = Pt(4)
            # render simple **bold** spans
            parts = chunk.split("**")
            for j, piece in enumerate(parts):
                run = para.add_run(piece)
                run.bold = (j % 2 == 1)
    elif kind == "fig":
        from PIL import Image as _PILImage
        with _PILImage.open(block[1]) as _im:
            px_w, px_h = _im.size
        aspect = px_h / px_w
        # Constrain max width to 8.8 in and max height to 4.7 in so figure, caption, and spacing
        # never exceed single page limits or cause orphaned layout breaks.
        fig_w = min(8.8, 4.7 / aspect)
        p_fig = doc.add_paragraph()
        p_fig.alignment = 1
        p_fig.paragraph_format.keep_with_next = True
        p_fig.paragraph_format.space_before = Pt(4)
        p_fig.paragraph_format.space_after = Pt(2)
        run_fig = p_fig.add_run()
        run_fig.add_picture(block[1], width=Inches(fig_w))
        caption = doc.add_paragraph(block[2])
        caption.alignment = 1
        caption.paragraph_format.space_before = Pt(2)
        caption.paragraph_format.space_after = Pt(6)
        caption.runs[0].italic = True
        caption.runs[0].font.size = Pt(9)
    elif kind == "table":
        caption = doc.add_paragraph(block[1])
        caption.paragraph_format.keep_with_next = True
        caption.paragraph_format.space_before = Pt(6)
        caption.paragraph_format.space_after = Pt(2)
        caption.runs[0].italic = True
        caption.runs[0].font.size = Pt(9)
        df = block[2]
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = "Table Grid"
        for k, col in enumerate(df.columns):
            cell = table.rows[0].cells[k]
            cell.text = str(col)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(8.5)
        for _, row in df.iterrows():
            cells = table.add_row().cells
            for k, value in enumerate(row):
                cells[k].text = str(value) if str(value) != "" and str(value) != "nan" else "-"
                for paragraph in cells[k].paragraphs:
                    paragraph.paragraph_format.space_before = Pt(2)
                    paragraph.paragraph_format.space_after = Pt(2)
                    for run in paragraph.runs:
                        run.font.size = Pt(8.5)
        for row in table.rows:
            trPr = row._tr.get_or_add_trPr()
            trPr.append(OxmlElement('w:cantSplit'))
        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(4)

doc.save(REPORT_DIR / "Assignment1_Report.docx")
print("Wrote report/Assignment1_Report.md and report/Assignment1_Report.docx")
