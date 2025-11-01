# Synthetic Counterfactual Border Audit (SCBA) — Enhanced Implementation Plan

**Owner:** AI for Medical Imaging Lab\
**Audience:** Coding Agent (talented software engineer)\
**Scope:** Chest X‑rays (CXR) only (2D).\
**Goal:** Build the first systematic **counterfactual border audit** for **segmentation XAI** in chest imaging, with reproducible code, a curated dataset, a wide baseline comparison, rigorous robustness tests, a radiologist user study, and a clinical‑deployment‑ready viewer (3‑column: original / perturbed / repaired). Deliverables should support a MICCAI/TMI submission.

---

## 0) What’s new vs. the earlier plan

- **Wider XAI baselines:** add Grad‑CAM++, HiResCAM, Guided Grad‑CAM, SmoothGrad, Occlusion, LIME, SHAP, RISE, and (optional) D‑RISE for detection heads.
- **Robustness suite:** sanity checks, stability under noise/blur/compression, OOD/domain shift, adversarial edits, and CF‑stress (our border/lesion edits).
- **Clinical deployment:** DICOM SEG/SR overlays, OHIF/PACS hooks, logging/audit and “explanation stability” indicator.
- **Radiologist user study:** application‑grounded evaluation (interpretability & trust calibration) with counterfactual pairs.
- **Sharper metrics:** formalize **AM‑ROI**, **CoA‑Δ**, **Directional Consistency**, **faithfulness (deletion/insertion)** with in‑distribution reveals, **pointing‑game**, **localization IoU**, and **stability (SSIM/Pearson)**.
- **Repro kit:** fully scripted sweeps, CI, seed control, run cards, and paper figures.

---

## 1) Problem statement & contribution

We will **synthesize realistic, controlled perturbations** at **organ borders** and **lesion surrogates** in chest X‑rays to audit whether segmentation explanations **follow the causal edit**. We propose a measurement suite that jointly evaluates **what moved** (saliency) and **what changed** (predictions/segmentations). Results are packaged as a **benchmark + toolkit + triptych UI** for human review.

**Core questions**

1. **Border sensitivity:** If we dilate/erode/jitter lung borders (or pathology borders), do explanations shift to the **new contour band**?
2. **Lesion sensitivity:** If we insert **Gaussian nodule surrogates**, do explanations accumulate there proportionally to effect size?
3. **Repair reversibility:** After inpainting/removal, do explanations **return** to baseline?
4. **Robustness:** Are explanations **stable** under benign perturbations and **honest** under adversarial or OOD shifts?

---

## 2) Datasets (primary + optional)

**Primary CXR (with lung masks):**

1. **Montgomery** & **Shenzhen** (Jaeger et al., 2014): public PA CXRs with expert lung masks. Good for border fidelity.
2. **JSRT**: 247 PA CXRs; community masks available; 2048² native resolution. Good for fine border analysis.

**Optional (feature‑flag):** 3) **SIIM‑ACR Pneumothorax** (segmentation): to test generalization beyond lungs (pleural border pathology).\
4\) **CheXmask** (2024): large‑scale auto‑generated masks (anatomical); used only for **scale/stability** analyses (label‑noise caveats).\
5\) **LIDC‑IDRI (CT)**: not a benchmark target (3D), but use 2D axial slices **only** to calibrate nodule surrogate realism.

**Data prep & policy**

- Scripts `scripts/prep_{montgomery,shenzhen,jsrt}.py` download and verify checksums, deduplicate patients, and create `{train,val,test}.csv`.
- Normalize to `float32`, keep pixel spacing if available; produce 1024×1024 and native‑res pipelines.
- Respect licenses: redistribute **recipes/IDs**, not images, where restricted.

---

## 3) Baseline segmentation models

- **U‑Net** (baseline), **Attention U‑Net**, optional **nnU‑Net wrapper** (if time). Binary lung masks; optional pneumothorax head.
- **Loss:** Dice + BCE (lungs), Dice only for single class; optional focal for pneumothorax.
- **Augmentations:** flips, ±5–10° rotate, mild brightness/contrast, CLAHE, small Gaussian noise; no geometry that breaks anatomy.
- **Training:** AMP enabled, AdamW, cosine LR, early stopping on val Dice/BF‑score. Export TorchScript for viewer.

---

## 4) XAI methods (segmentation‑aware, broad)

**Gradient‑based**

- **Seg‑Grad‑CAM** (Vinogradova et al., AAAI’20).
- **Seg‑XRes‑CAM** (CVPRW’23): spatially weighted variant; better for **region‑within‑mask** explanation.
- **HiResCAM** (medical imaging‑friendly high‑res CAM).
- **Grad‑CAM++** (multi‑instance sensitivity).
- **Guided Backprop** & **Guided Grad‑CAM** (detail sharpening).
- **Integrated Gradients (IG)**; **LRP** (zennit).

**Perturbation/black‑box**

- **Occlusion** (patch sliding).
- **RISE** (random mask sampling).
- **LIME** (superpixels).
- **SHAP** (kernel SHAP on superpixels).
- **(Optional)** **D‑RISE** for detectors.

**Common API**

```python
explain(image, model, target_mask=None, class_id=None, mode="seg", method="seg_grad_cam", **kwargs) -> SaliencyMap  # HxW, float32
```

- All maps normalized to [0,1]; keep raw scores; expose seed.
- Enforce **sanity checks** utilities (parameter randomization, input randomization) as test helpers.

---

## 5) Synthetic counterfactual generators (SCBA core)

### 5.1 Border edits (mask‑aware, in‑distribution)

- **Binary morphology:** dilation/erosion/open/close with radius r∈{1,2,4,8,12} px.
- **Contour jitter:** random Fourier shape noise (band‑limited) on the SDF isocontour.
- **Warping:** thin‑plate spline (TPS) guided by control points from original vs. perturbed contour samples.
- **Blending:** Poisson blending within a narrow band (8–16 px) around the new border to avoid seams.
- **Budgets:** constrain Δarea to {±5%, ±10%, ±20%}; stratify Dice(mask₀, mask₁) into {0.8, 0.6, 0.4} bins.
- **Outputs:** (perturbed\_image, perturbed\_mask, **ROI\_band**) where ROI\_band is the symmetric contour band used for metrics.

### 5.2 Gaussian nodule surrogates (parenchyma only by default)

- **Shape:** 2D anisotropic Gaussian; optional DoG halo (spiculated look). σx,σy∈[2,12] px; rotation uniform.
- **Intensity:** sample ΔI from local lung window (μ±k·σ), clamp to plausible range; optional rib‑shadow modulation via steerable filters.
- **Placement:** within lung mask, margin ≥ 20 px from pleura/hilum unless border‑adjacent scenario is toggled.
- **Blending:** multi‑scale Laplacian or Poisson; add subtle noise to match grain.
- **Outputs:** (image\_with\_lesion, lesion\_mask, lesion\_bbox).

### 5.3 “Repair” scenes

- **Undo border edits** (restore mask and TPS‑warp back) or **inpaint lesion** (Telea/Navier‑Stokes; optional diffusion‑inpainting hook). Report SSIM(image\_repaired, image\_original).

**Implementation notes**

- All generators deterministic under seed; expose config dataclasses; assert **idempotence** at r=0 and ΔI=0.

---

## 6) Metrics (definitions & formulas)

Let `S(x)` be a normalized saliency map (∑ S = 1) for image `x`.

### 6.1 Faithfulness (deletion/insertion with in‑distribution reveals)

- **Deletion AUC:** progressively fade top‑k% salient pixels (alpha‑blend to locally in‑distribution mean) and measure drop in **segmentation confidence** or **soft‑Dice** to GT; lower AUC = more faithful.
- **Insertion AUC:** progressively reveal top‑k% and measure recovery; higher AUC = more faithful.

### 6.2 Counterfactual consistency (proposed)

- **AM‑ROI:** attribution mass inside edited ROI (border band or lesion):\
  `AM_ROI = ∑_{p∈ROI} S(p)`.
- **ΔAM‑ROI:** `AM_ROI(perturbed) − AM_ROI(original)`; want positive when ROI is the causal edit.
- **CoA (center of attribution):** centroid of S; **CoA‑Δ** = Euclidean shift toward ROI center.
- **Directional Consistency (DC):** fraction of samples where CoA moves **closer** to ROI after perturbation **and** returns after repair.

### 6.3 Segmentation stability

- **ΔDice/ΔIoU** vs. GT across conditions; **Boundary F‑score (BF‑score)**; **Hausdorff 95** (optional) on borders.

### 6.4 Localization & pointing

- **Localization IoU:** saliency thresholded to top‑τ% vs. pathology/organ mask (if a localized target exists).
- **Pointing game:** whether argmax(S) falls within ROI/lesion.

### 6.5 Stability & calibration

- **Stability:** SSIM / Pearson(S₀, Snoise) under benign noise/blur/compression; **saliency entropy** (compactness).
- **Sanity checks:** parameter randomization / input randomization pass‑rates.
- **Agreement:** pairwise IoU between methods; report ensemble consensus.

**Reporting:** per‑image → per‑dataset → overall with **BCa bootstrap CIs**; publish stratified by edit strength bins.

---

## 7) Robustness test suite (beyond SCBA)

1. **Benign perturbations:** Gaussian noise σ∈{0.5,1.0,2.0}, JPEG q∈{100,80,60}, Gaussian blur r∈{1,2,3}. Expect small saliency drift.
2. **Domain shift:** train on Montgomery+Shenzhen, test on JSRT (and vice versa). Track **explanation drift** (mean CoA‑Δ w\.r.t. lung centroid) and localization IoU.
3. **Adversarial edits (white/black‑box):** generate perturbations that **preserve label** but attempt to move saliency off‑ROI; measure drop in AM‑ROI without prediction change.
4. **Sanity checks:** Adebayo randomization (layers/labels).
5. **Counterfactual stress:** our border/lesion edits across budgets; assert ΔAM‑ROI monotonicity with edit strength.

---

## 8) Reviewer UI (3‑column + study mode)

- **Layout:** Original | Perturbed | Repaired triptych with synchronized pan/zoom.
- **Overlays:** GT mask, prediction, and saliency (method dropdown). Opacity sliders; ROI band toggle; edge emphasis.
- **Controls:** border radius r, Δarea, Gaussian σ, contrast ΔI, seed; live preview of **AM‑ROI / CoA‑Δ**.
- **Export:** triptych PNG, JSON config, per‑case metrics row (CSV/Parquet).
- **Study mode:** presents randomized cases & counterfactual pairs; collects Likert ratings (relevance, coherence, trust), free‑text, and decisions.
- **Clinical hooks:** DICOM SEG/SR export; Secondary Capture for heatmaps; OHIF viewer link; syslog audit.

Tech stack: Streamlit or Gradio (dev), FastAPI backend, TorchScript models; optional OHIF + DICOMweb for PACS sandbox.

---

## 9) Repository layout

```
scba/
  README.md
  env/environment.yml  # CUDA & pinned libs
  scba/
    data/
      loaders/{montgomery,shenzhen,jsrt,siim,chexmask}.py
      transforms/*.py   # standardize, CLAHE, etc.
    models/
      unet.py  attention_unet.py  nnunet_wrapper.py
    xai/
      cam/
        seg_grad_cam.py  seg_xres_cam.py  grad_cam_pp.py  hires_cam.py  guided_backprop.py
      perturb/
        occlusion.py  rise.py  lime.py  shap_kernel.py  drise.py
      ig.py  lrp.py  common.py
    cf/
      borders.py  gaussian_nodules.py  tps.py  poisson_blend.py  inpaint.py  utils.py
    metrics/
      faithfulness_auc.py  cf_consistency.py  stability.py  localization.py  bootstrap.py  sanity.py
    robustness/
      benign.py  domain_shift.py  adversarial.py
    ui/
      app.py  study_mode.py  exporters.py  dicom_export.py
    train/
      train_seg.py  eval_seg.py  export_torchscript.py
    scripts/
      prep_*.py  run_cf_sweep.py  run_robustness.py  make_tables.py  export_triptych.py
  experiments/
    configs/*.yaml
    results/
  tests/
    test_*.py
  LICENSE
```

---

## 10) Implementation plan (do it task‑by‑task)

> **Rule:** Complete each task end‑to‑end (code → unit tests → sample run → commit) before moving on. Use GPU; enable AMP.

### Phase A — Scaffolding & data (Week 1)

A1. **Repo bootstrap**: pre‑commit (black/isort/flake8), `pytest`, GitHub Actions (CPU smoke), release drafter.\
A2. **Environment**: `env/environment.yml` with PyTorch+CUDA, torchvision, captum (IG), zennit (LRP), shap, lime, scikit‑image, kornia, opencv‑python, albumentations, pydicom, pydicom‑seg, pynetdicom, fastapi, streamlit/gradio, wandb/mlflow.\
A3. **Data loaders**: implement `{montgomery,shenzhen,jsrt}.py` with deterministic splits, metadata CSV, checksum tests.\
A4. **Sanity dataset script**: tiny subset export for CI and UI demo.

### Phase B — Baseline models & training (Week 2)

B1. **U‑Net train** per dataset; log Dice/BF‑score; save best weights & TorchScript.\
B2. **Eval CLI**: `python -m scba.train.eval_seg --ckpt path --data jsrt` (prints Dice/IoU/BF; saves confusion figures).\
B3. **Docs**: run cards with seeds, hyperparams, compute time.

### Phase C — XAI interface & baselines (Week 3)

C1. **XAI API** (`xai/common.py`); implement Seg‑Grad‑CAM, Seg‑XRes‑CAM, HiResCAM, Grad‑CAM++, Guided Backprop, IG, LRP; write **shape/range/determinism** unit tests.\
C2. **Perturbation explainers**: Occlusion, RISE, LIME, SHAP; cache model calls; unit tests for reproducibility.\
C3. **Sanity checks** helpers (layer randomization) + tests.

### Phase D — Counterfactual generation (Week 4)

D1. **Border edits** (`cf/borders.py`, `cf/tps.py`, `cf/poisson_blend.py`): implement ops & ROI bands; tests (idempotence at r=0; Δarea bins).\
D2. **Gaussian nodules** (`cf/gaussian_nodules.py`): placement rules; blending; lesion masks; tests (ΔI=0 ⇒ no‑op).\
D3. **Repair ops** (`cf/inpaint.py`): Telea/Navier‑Stokes; SSIM checks; hooks for diffusion inpainting (optional).

### Phase E — Metrics & sweeps (Week 5)

E1. **Faithfulness** (deletion/insertion) with alpha‑blend reveals; configurable reveal schedule.\
E2. **CF consistency** (AM‑ROI, ΔAM‑ROI, CoA‑Δ, DC).\
E3. **Stability/localization**: SSIM/Pearson; IoU/pointing game.\
E4. **Sweep script** `run_cf_sweep.py`: loops datasets×methods×edits; writes per‑case CSV + aggregates with BCa CIs; logs to W&B/MLflow; saves plots.

### Phase F — Robustness suite (Week 6)

F1. **Benign** (`robustness/benign.py`): noise/blur/JPEG; stability curves.\
F2. **Domain shift** (`robustness/domain_shift.py`): train↔test swaps; drift metrics.\
F3. **Adversarial** (`robustness/adversarial.py`): PGD/FGSM to shift saliency with minimal output change; guardrails.

### Phase G — UI & DICOM export (Week 7)

G1. **Triptych UI** (`ui/app.py`): synchronized views, overlays, method switcher, live metrics, export buttons.\
G2. **Study mode** (`ui/study_mode.py`): case randomizer, Likert forms, result store; admin dashboard for stats.\
G3. **DICOM export** (`ui/dicom_export.py`): SEG for masks, SR for metrics, Secondary Capture for heatmaps; OHIF linkout.

### Phase H — User study pack (Week 8)

H1. **IRB pack** templates; consent; anonymization; compensation guidelines.\
H2. **Study protocol** JSON (case sets, counterfactual pairs, tasks).\
H3. **Analysis notebook** (trust calibration, preference stats, qualitative coding skeleton).

### Phase I — Paper kit & release (Week 9)

I1. **make\_tables.py / figures** (triptychs, AUC curves, Δ metrics barplots).\
I2. **README & docs site** (Quickstart, Method Card, Limitations, Ethics).\
I3. **Release** weights, configs, and synthetic generation scripts.

---

## 11) Detailed specs & pseudo‑code

### 11.1 Border edit generator (TPS + Poisson blend)

```python
@dataclass
class BorderEditCfg:
    radius_px: int  # morphology radius
    contour_noise: float  # Fourier noise amplitude
    band_px: int = 12
    area_budget: float = 0.10  # |Δarea| ≤ 10%
    seed: int = 0

def apply_border_edit(img, mask, cfg: BorderEditCfg):
    m_pert = morph_edit(mask, cfg.radius_px, cfg.area_budget, cfg.seed)
    c0, c1 = sample_correspondence(mask, m_pert, seed=cfg.seed)
    tps = fit_tps(c0, c1)
    img_warp = warp_tps(img, tps)
    band = contour_band(m_pert, width=cfg.band_px)
    img_cf = poisson_blend(src=img_warp, dst=img, mask=band)
    return img_cf, m_pert, band
```

### 11.2 Gaussian nodule surrogate

```python
@dataclass
class NoduleCfg:
    sigma: Tuple[float,float]  # (σx, σy)
    delta_I: float
    margin_px: int = 20
    use_border_adjacent: bool = False
    seed: int = 0

def insert_gaussian_nodule(img, lung_mask, cfg: NoduleCfg):
    center = sample_parenchyma(lung_mask, margin=cfg.margin_px, seed=cfg.seed)
    blob = anisotropic_gaussian(img.shape, center, cfg.sigma, seed=cfg.seed)
    lesion = blend_poisson(img, blob * cfg.delta_I)
    lesion_mask = blob > tau_from_sigma(cfg.sigma)
    return lesion, lesion_mask
```

### 11.3 AM‑ROI & CoA‑Δ

```python
def attribution_mass_roi(S, ROI):
    S = S / (S.sum() + 1e-8)
    return float((S * ROI).sum())

def center_of_attribution(S):
    S = S / (S.sum() + 1e-8)
    ys, xs = np.indices(S.shape)
    return np.array([ (S*ys).sum(), (S*xs).sum() ])
```

---

## 12) Quality gates & tests

- **Unit tests**: XAI shapes/dtypes; determinism with seed; border edits idempotent at r=0; lesion ΔI=0 ⇒ no‑op; Δarea binning; metrics monotonicity on toy images; sanity‑check pass/fail logic.
- **Golden cases**: fixed‑seed triptychs for regression testing; hash triptych PNGs.
- **Performance**: Seg‑Grad‑CAM ≤ 120 ms @ 1024² on A100 (AMP); cache conv features; crop to lung bbox.
- **Numerical**: mixed precision whitelist; clamp/normalize; test NaN‑guards.

---

## 13) Clinical deployment checklist (prototype → PACS)

- **Packaging**: containerized FastAPI + TorchScript; GPU optional.
- **Standards**: export segmentation as **DICOM SEG**; metrics as **DICOM SR**; heatmaps as **Secondary Capture**; DICOMweb endpoints for OHIF.
- **Shadow mode**: vendor‑agnostic PACS integration (OHIF) for retrospective validation; log model/XAI versions & hashes; **audit trail** for each explanation shown.
- **Safety rails**: show **explanation stability badge** (green/yellow/red) based on stability & sanity; warn if out‑of‑distribution.
- **Governance**: PHI‑safe pipeline; role‑based access; retention policies.

---

## 14) Radiologist user study (application‑grounded)

- **Participants**: ≥ 15 radiologists/residents (power as available).
- **Design**: within‑subjects; tasks include diagnosis decision w/ and w/o explanations; **counterfactual pairs** for border/lesion edits.
- **Measures**: interpretability (relevance/coherence Likert), **trust** & **trust calibration** (confidence vs. correctness), error detection rate, time‑to‑decision, preference (Grad‑CAM vs. LIME/SHAP/RISE), qualitative feedback.
- **Data**: de‑identified; randomized order; IRB templates provided; compensation via institutional norms.
- **Analysis**: paired tests (t/Wilcoxon), effect sizes, bootstrap CIs; thematic coding for free text.
- **Outcomes**: Which XAI methods aid decisions? Do counterfactual audits reveal misleading saliency? Are explanations appropriately trusted?

---

## 15) Paper kit (MICCAI/TMI)

- **Title (working):** *Synthetic Counterfactual Border Audit for Segmentation Explainability in Chest X‑rays.*
- **Key Figures:** triptych panels; faithfulness curves; ΔAM‑ROI bars; stability plots; study preference chart; PACS overlay screenshot.
- **Ablations:** Δarea bins, σ/ΔI sweep, blend radius, method comparison, domain shift.
- **Limitations:** surrogate realism; saliency method variance; generalization beyond lungs.

---

## 16) Timeline (ambitious, 9 weeks)

- **W1:** Scaffolding & data.
- **W2:** Train baselines.
- **W3:** XAI API + gradient methods.
- **W4:** Perturbation explainers + sanity checks.
- **W5:** CF generators + metrics.
- **W6:** Robustness suite.
- **W7:** UI + DICOM export.
- **W8:** User study pack & pilot.
- **W9:** Sweeps, figures, docs & release.

---

## 17) Definition of Done (DoD)

- **Repro sweep** produces paper tables/plots; CI green; unit tests ≥ 85% for cf+metrics.
- **Models** released; **UI** runs locally with GPU; DICOM export verified in OHIF.
- **User study** pilot complete; summary stats included.
- **Docs** include Quickstart, Method Card, Limitations, Ethics, and Governance.

---

## 18) Quickstart for the Coding Agent

```bash
# 1) Create env
conda env create -f env/environment.yml && conda activate scba

# 2) Prep data (example)
python -m scba.scripts.prep_montgomery --out data/montgomery
python -m scba.scripts.prep_shenzhen  --out data/shenzhen
python -m scba.scripts.prep_jsrt      --out data/jsrt

# 3) Train a baseline
python -m scba.train.train_seg --data jsrt --arch unet --epochs 80 --amp --save runs/jsrt_unet.pt

# 4) Run counterfactual sweep
python -m scba.scripts.run_cf_sweep --data jsrt --ckpt runs/jsrt_unet.pt \
  --methods seg_grad_cam,seg_xres_cam,hires_cam,ig,lrp,rise,lime,shap \
  --border_radii 1,2,4,8,12 --lesion_sigmas 2,4,8,12 --deltaI 0.2,0.4 \
  --out results/jsrt_cf

# 5) Launch UI
python -m scba.ui.app --demo assets/demo_jsrt

# 6) Export to DICOM + figures
python -m scba.ui.exporters --triptych results/jsrt_cf --dicom-out exports/dicom
python -m scba.scripts.make_tables --in results --paper-figs figs/
```

---

## 19) Risks & mitigations

- **Unrealistic edits** → spurious XAI: use Poisson blending + TPS warp + realism checks (SSIM back to baseline for repairs).
- **AUC artifacts** under hard occlusion: use **in‑distribution alpha reveals** and pair with ΔAM‑ROI & CoA‑Δ.
- **Compute overhead**: cache features & saliency; crop to lung bbox; AMP; parallel RISE masks.

---

## 20) References (selected, non‑exhaustive)

- Vinogradova et al., 2020 — Seg‑Grad‑CAM (AAAI).
- Hasany et al., 2023 — Seg‑XRes‑CAM (CVPRW).
- Draelos et al., 2020 — HiResCAM (medical imaging).
- Petsiuk et al., 2018 — RISE (BMVC). Ribeiro et al., 2016 — LIME (KDD). Lundberg & Lee, 2017 — SHAP (NeurIPS).
- Saporta et al., 2022 — Benchmarking saliency in CXR (Nat. Mach. Intell.).
- Adebayo et al., 2018 — Sanity Checks (NeurIPS). Ghorbani et al., 2019 — Interpretation Fragility (AAAI).
- Datasets: Jaeger et al., 2014 (Montgomery/Shenzhen); JSRT; SIIM‑ACR; CheXmask (2024); LIDC‑IDRI (TCIA).
- Poisson Image Editing (Pérez et al., 2003) and follow‑ups for seamless blending.
- PACS/DICOM integration: OHIF + DICOM SEG/SR best practices.

---

### Final note to the Coding Agent

Execute **Phase A → I** sequentially. After completing all tasks:

1. Run the **full sweep** on the three primary datasets.
2. Verify the **UI** on 50 random cases per dataset.
3. Confirm **ΔAM‑ROI increases** on perturbed images and **reverses** on repairs for ≥70% of cases.
4. Generate the **paper figures** and archive all configs/results for reproducibility.

> We have GPU access—use AMP, caching, and lung‑box cropping to keep runtimes snappy. Good luck!

