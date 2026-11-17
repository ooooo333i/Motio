# Handoff: GVHMR + HaMeR wrist merge & 2D refinement (DeVI-style)

This document hands an **already-completed design** to an implementation agent
(Claude Code / Codex). The paper reading and design decisions are DONE — do not
re-derive them. Your job is to wire up, run, and **debug via visual overlays**,
following the step-gated checklist at the bottom. Do not skip overlay gates.

Repo: https://github.com/ooooo333i/capstone_deploy
Skeleton already written: `cap_wrist_refine.py` (treat as the spec; fill TODOs).
Reference paper: DeVI (arXiv 2604.20841), Sec 4.2 + Appendix A.5/A.6. You do NOT
need to read it — all relevant equations are reproduced below.


---

## 0. Current confirmed state and scope correction

This section supersedes any older assumptions in this document that conflict with it.
Do not delete the older DeVI-style design notes; keep them as background, but follow
this section for the current implementation scope.

### 0.1 Context reset warning

A previous Claude session reached **100% CONTEXT USED**. Do not rely on previous
conversation memory. Use this HANDOFF document and the current repository files as
the source of truth. Before changing code, inspect the current files directly.

### 0.2 Root cause already found

The main HaMeR failure was not primarily caused by Stage A, Stage B, DeVI math,
SMPL-X wrist optimization, or rendering.

The root cause was the **hand bbox passed into HaMeR**. The debug pipeline used
GVHMR's `vitpose_path`, which contains **COCO17 body keypoints only**. COCO17 has
wrist and elbow keypoints, but no hand/finger keypoints. Because of that, HaMeR
hand bboxes were estimated from wrist/elbow via `estimate_hand_boxes_from_coco17()`.

Confirmed bbox issue:

```text
Total detections: 2989
box_size min = 768.0
box_size max ≈ 779.0
box_size mean ≈ 768.0
box_size median = 768.0
box_size >= 768: 2989 / 2989, 100%

Frame 200:
left box_size = 768.0
right box_size = 768.0
```

This means HaMeR was receiving a very large, nearly fixed crop that covered a large
arm region instead of a tight hand crop.

### 0.3 Correct bbox solution already verified

ViTPose-H WholeBody 133-keypoint based hand bbox was tested and produced very good
HaMeR MANO projection. This is now the correct bbox direction.

WholeBody keypoint layout used here:

```text
Total keypoints: 133
left hand:  kp[91:112]
right hand: kp[112:133]
```

Important scope rule:

```text
GVHMR remains unchanged.
WholeBody ViTPose is used only for HaMeR hand bbox generation.
Do not feed WholeBody keypoints into GVHMR.
Do not replace GVHMR's COCO17 vitpose_path.
Do not modify GVHMR body inference.
```

### 0.4 Correct current pipeline direction

```text
GVHMR:
- keep existing COCO17-based body pipeline
- keep existing GVHMR outputs

HaMeR:
- replace wrist/elbow COCO17 hand bbox with WholeBody133 hand bbox
- use ViTPose-H WholeBody only as a HaMeR bbox helper
- use --hamer-rescale-factor 2.0 for WholeBody133 bbox, matching official HaMeR demo behavior

After HaMeR projection is stable:
- regenerate full HaMeR npz results
- merge HaMeR hand pose into SMPL-X
- run Stage A wrist replacement
- run Stage B 2D reprojection refinement
```

### 0.5 Stage B is required, but its variants must be compared

Stage B is still needed. Stage A only inserts a wrist rotation using a frame-sensitive
formula, and `G_w` from HaMeR crop-camera space is not guaranteed to perfectly match
GVHMR incam space. The remaining mismatch should be absorbed by Stage B.

However, Stage B should not be treated as a single fixed policy. Compare two variants:

#### Variant B1: arm-chain only

```text
Optimize:
- wrist
- elbow
- shoulder
- optional upper spine if needed

Freeze:
- left_hand_pose
- right_hand_pose
```

Purpose: preserve HaMeR finger detail and adjust only the SMPL-X/GVHMR arm chain.

#### Variant B2: arm-chain + regularized hand pose

```text
Optimize:
- wrist
- elbow
- shoulder
- optional upper spine if needed
- left_hand_pose
- right_hand_pose

Add regularization:
- keep optimized hand_pose close to the original HaMeR hand_pose
```

Recommended added term:

```text
L = w_b·L_b + w_h·L_h + w_tc·L_tc + w_pose·L_pose
L_pose = || left_hand_pose_opt  - left_hand_pose_hamer  ||²
       + || right_hand_pose_opt - right_hand_pose_hamer ||²
```

Purpose: allow small finger-pose corrections when wrist alignment is difficult, while
preventing the optimizer from destroying HaMeR's hand detail.

Final decision should be based on overlay videos, wrist-hand junction quality, and
temporal jitter, not only loss values.

### 0.6 Next implementation target

First implement and stabilize WholeBody133 bbox in `entry_point_hamer.py`. Then rerun
HaMeR only. Do not run GVHMR again. Do not run Stage A/B until the full HaMeR output
with WholeBody133 bbox is saved and projection samples look correct.

Required code direction:

```text
Primary target:
- entry_point_hamer.py

Reference only unless absolutely necessary:
- entry_point.py
- cap_wrist_refine.py
- external/hamer demo code
- external/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth
```

Add CLI option:

```text
--hand-bbox-source coco17 | wholebody133 | auto
```

Meaning:

```text
coco17: existing wrist/elbow bbox
wholebody133: ViTPose-H WholeBody hand bbox
auto: try wholebody133 first, fallback to coco17 if invalid
```

Add bbox overlay debug output on the original frame image:

```text
red: coco17 bbox
green: wholebody133 bbox
blue: fallback/reference if needed
```

After this step, suggest a full HaMeR rerun command using:

```text
--hand-bbox-source wholebody133 or auto
--hamer-rescale-factor 2.0
```

Do not overwrite the baseline `.pt` files.

---

## 1. What we are building (and what already exists)

**Two entry-point tracks in this repo** (do not confuse them):
- `entry_point.py` — production entry, called by `service.py` and
  `batch_entry_point.py`. Outputs `smplx_merged_hamer.pt`. Plain finger-only merge.
- `entry_point_hamer.py` — debug entry the user is iterating on. Has its own
  `main()` and CLI. `import entry_point as cap` to reuse the production functions,
  then runs `merge_hamer_hands_into_gvhmr_postprocessed` (finger-only merge +
  temporal smoothing / outlier handling on the hand sequence, via
  `postprocess_hand_sequence` + `HandPostprocessConfig`). Outputs
  `smplx_merged_hamer_post.pt`.

**We integrate into `entry_point_hamer.py` ONLY.** Leave `entry_point.py` untouched
until the debug pipeline is validated end-to-end.

The repo ALREADY merges GVHMR (body) + HaMeR (hand) but only swaps **finger pose**
(`left_hand_pose` / `right_hand_pose`), keeping GVHMR's wrist. See
`entry_point.py: merge_hamer_hands_into_gvhmr` (production) and
`entry_point_hamer.py: merge_hamer_hands_into_gvhmr_postprocessed` (debug). Both
keep `smpl_params_global` and `smpl_params_incam` in sync.

We are ADDING two stages. **Integration pattern = minimal change** (do not refactor
`merge_hamer_hands_into_gvhmr_postprocessed`):

```
GVHMR
 → HaMeR (with extended npz save — see §4)
 → merge_hamer_hands_into_gvhmr_postprocessed()       [finger-only merge + temporal post-process]
 → smplx_merged_hamer_post.pt                          [BASELINE, do not overwrite]
 → run_wrist_refine(merged_input_path=smplx_merged_hamer_post.pt,
                    output_path=smplx_merged_hamer_wrist_refined.pt)
 → render the refined .pt
```

The user's "hand smoothing / penetration" step (`postprocess_hand_sequence`) is
already inside `merge_hamer_hands_into_gvhmr_postprocessed`, so wrist refine runs
**on its output**, not before it. The Stage A (wrist) + Stage B (2D refine) inside
`run_wrist_refine`:

- **Stage A — wrist replacement** (DeVI Appendix A.5):
  replace GVHMR's local wrist rotation with one derived from HaMeR's wrist.
- **Stage B — 2D reprojection refinement** (DeVI eq.10, HOI term dropped):
  optimize upper-body + hand pose so projected joints match 2D observations.

The 3D-object / HOI-contact parts of DeVI are intentionally OUT OF SCOPE.

---

## 2. Locked design decisions (do not revisit)

1. **Coordinate system = GVHMR incam.** Use `pred["smpl_params_incam"]` and
   `pred["K_fullimg"]` (already stored). All projection happens in incam.
2. **Reuse GVHMR's SMPL-X.** `make_smplx("supermotion_fullhands")` →
   `BodyModelSMPLX`. It is standard SMPL-X (`use_pca=False`, `flat_hand_mean=False`),
   so `body_pose` is `(F, 21, 3)` axis-angle and is differentiable. Do NOT build a
   new SMPL-X layer.
3. **Wrist transform (Stage A):**  `G_e = ∏_{r=0..elbow} R_{p_r}` ,
   `R_w = G_e^{-1} G_w`, then axis-angle → write into `body_pose` wrist row.
   `G_w` = HaMeR wrist global rotation (`mano_params["global_orient"]`).
4. **Refinement loss (Stage B):**  `L = w_b·L_b + w_h·L_h + w_tc·L_tc`
   - `L_b`, `L_h`: squared 2D reprojection error of body / hand joints.
   - `L_tc`: mean geodesic distance between adjacent-frame rotations.
   - Weights from DeVI A.6 can be used as starting defaults: `w_b=1.0, w_h=1.0, w_tc=1.0e4`. Adam, lr `2e-2`, no decay.
   - Current project scope requires comparing **two Stage B variants**:
     - **B1 arm-chain only:** optimize wrist/elbow/shoulder/optional upper spine; freeze `left_hand_pose` and `right_hand_pose`.
     - **B2 hand-regularized:** optimize wrist/elbow/shoulder/optional upper spine + `left_hand_pose`/`right_hand_pose`, but add `L_pose` so hand pose stays close to the original HaMeR hand pose.
   - Lower body and unrelated occluded joints remain out of scope.
5. **Hand 2D target source — start with Option 1, fall back to Option 2 if needed.**
   - **Option 1 (start here):** reproject HaMeR's own 3D hand joints into incam and
     use those pixels as the target. Cheapest; no new model. Known weakness: bakes
     HaMeR's crop→incam error into the target (limited ceiling).
   - **Option 2 (fallback):** real image hand keypoints (MediaPipe Hands OR
     ViTPose-wholebody) as target. Higher fidelity, no circularity. Adds a detector.
   - The target builder is isolated in ONE function so switching is a single swap;
     the optimization loop, losses, SMPL-X, projection, and Stage A stay identical.
6. **body 2D target:** GVHMR is already fitted to its VitPose COCO-17, so re-fitting
   body to VitPose is largely redundant. Keep `body_targets=None` (or a weak anchor
   on shoulders/elbows only). The live data term is `L_h`.
7. **Hand post-processing (`postprocess_hand_sequence` inside
   `merge_hamer_hands_into_gvhmr_postprocessed`) is untouched and runs FIRST.**
   It is already part of the debug merge that produces `smplx_merged_hamer_post.pt`.
   Wrist refine runs ON THAT OUTPUT — it does NOT come before the post-process and
   does NOT modify it. The text "wrist refine before post-process" is incorrect; the
   correct pipeline is in §1.
8. **The baseline to beat is the existing finger-only merge, NOT DeVI.** This project
   is a *reduced* application of DeVI's body+hand unification, deliberately without
   object/HOI/physics. Success = measurably better hand/wrist quality than the repo's
   current merge. Do not frame results as "matching DeVI" — that is a different
   (physics-based HOI) problem and out of scope.
9. **Integration point = `entry_point_hamer.py` (debug pipeline), NOT `entry_point.py`.**
   In `entry_point_hamer.py`'s `main()`, after `merge_hamer_hands_into_gvhmr_postprocessed`
   has produced `smplx_merged_hamer_post.pt`, call
   `run_wrist_refine(merged_input_path=<that .pt>, output_path=<refined .pt>)` and then
   render the refined `.pt`. **Do not overwrite** `smplx_merged_hamer_post.pt` —
   STEP 6 needs it as the baseline (`smplx_merged_hamer_post.pt`,
   `smplx_merged_hamer_wrist_only.pt`, `smplx_merged_hamer_wrist_refined.pt`).
   Leave `entry_point.py` untouched until the debug pipeline is validated.
10. **Sync local-pose changes incam → global, with the right scope per stage.**
    The repo's existing merge writes both `smpl_params_global` and `smpl_params_incam`
    consistently. We extend that contract:
    - **After Stage A:** call `sync_wrist_rows(incam, global)` — only the two wrist
      body_pose rows changed.
    - **After Stage B B1:** sync only the optimized body rows: wrist/elbow/shoulder/optional upper spine. Do not copy hand poses if they were frozen.
    - **After Stage B B2:** sync the optimized body rows AND both `left_hand_pose`/`right_hand_pose`, because B2 allows regularized hand-pose updates.
    All copied quantities are LOCAL rotations, so values transfer verbatim between incam and global (only `global_orient`/`transl` differ between the two frames).

---

## 3. Coordinate-frame cautions (this is where silent bugs live)

- **G_w is in HaMeR's crop-cam frame; G_e is in GVHMR's incam frame. They are NOT
  the same frame.** Therefore `R_w = G_e^{-1} G_w` only sets a good *initial* wrist
  orientation; the residual is expected and is absorbed by Stage B. Do not expect
  Stage A alone to be perfect. (If you later add a crop→incam alignment rotation
  `R_align`, apply `G_w ← R_align @ G_w` before the product.)
- **Left-hand mirror.** HaMeR treats the left hand as a flipped right hand. The repo
  mirrors finger `hand_pose` by negating y,z of each axis-angle. For the wrist
  rotation MATRIX the equivalent x-mirror is `M R M`, `M = diag(-1,1,1)`. VERIFY this
  matches by overlaying a left-hand frame — a wrong mirror puts the hand on the wrong
  side / flipped.
- **body_pose indexing:** `body_pose[i]` = full SMPL-X joint `(i+1)` (root excluded).
  Left wrist = full 20 → body index 19; right wrist = full 21 → body index 20.
  CONFIRM by printing `smplx_model.bm.parents` and deriving the chain, not by trusting
  hardcoded ids.
- **Are `out.joints` in incam?** Must verify by overlay (STEP 1). If GVHMR's wrist
  joint doesn't project onto the wrist in the raw frame, the projection/frame
  assumption is wrong and everything downstream is meaningless.

---

## 4. Data availability

- HaMeR npz (`hamer_out_dir/<frame>_<pid>.npz`; in the current debug pipeline this is usually `hamer_out_clean`) currently stores, per detection:
  `mano_params` (dict incl. `global_orient`, `hand_pose`, `betas`), `cam_t_full`,
  `cam_t`, `pred_cam`, `box_center`, `box_size`, `img_size`, `focal_length`,
  `is_right`, `vertices` (NOTE: saved as `vertices`, not `pred_vertices`).
  → `global_orient` (for `G_w`) and the camera-transform inputs ARE present.

- **OPEN ITEM RESOLVED — HaMeR 3D keypoints confirmed in current RunPod build:**
  HaMeR `out.keys()` was checked in the current RunPod environment.

  Confirmed keys:
  - `pred_keypoints_3d` exists, shape = `torch.Size([1, 21, 3])`
  - `pred_keypoints_2d` exists, shape = `torch.Size([1, 21, 2])`

  Therefore, extend the HaMeR save step in `entry_point_hamer.py` only, around the
  `np.savez(...)` call:

  ```python
  keypoints_3d=out["pred_keypoints_3d"][n].detach().cpu().numpy(),
  keypoints_2d=out["pred_keypoints_2d"][n].detach().cpu().numpy(),
  ```

  Then re-run HaMeR once to regenerate the npz files, and update
  `load_hamer_detections_full()` to read `keypoints_3d`.
  MANO forward fallback is not needed in the current RunPod environment unless this key disappears in another HaMeR build.


---

## 4.5 WholeBody133 HaMeR bbox implementation task

Before running any wrist refinement, fix HaMeR bbox generation in `entry_point_hamer.py`.

Required behavior:

```text
--hand-bbox-source coco17 | wholebody133 | auto
```

- `coco17`: use the existing wrist/elbow-based bbox.
- `wholebody133`: run ViTPose-H WholeBody and use hand keypoints.
- `auto`: use WholeBody133 when valid; fallback to COCO17 only when hand keypoints are invalid.

WholeBody hand extraction:

```text
left hand  = kp[91:112]
right hand = kp[112:133]
```

BBox creation:

```text
1. filter hand keypoints by confidence
2. require enough valid points
3. create tight min/max bbox
4. clamp to image bounds
5. reject boxes that are too small or invalid
6. use rescale_factor=2.0 for HaMeR ViTDetDataset
```

Debug output must include a bbox overlay drawn on the original image, not a blank canvas.

```text
red: coco17 bbox
green: wholebody133 bbox
blue: fallback/reference if needed
```

Only after WholeBody133 bbox and HaMeR projection are verified should the pipeline
continue to merge, Stage A, and Stage B.
---

## 5. STEP-GATED CHECKLIST (do in order; do NOT advance past a failing overlay)

> The pass/fail criterion at each gate is the **overlay image**, not the loss value.
> "Runs without error" is NOT "correct". A wrong coordinate/index produces a clean
> run with wrong-looking hands. Look at the pixels.

- **STEP 0 — indexing.** Print `smplx_model.bm.parents`. Derive wrist parent chains
  programmatically. Confirm `body_pose` is `(F,21,3)` and `full = body_idx + 1`.

- **STEP 1 — projection sanity (BEFORE any change).** Run `debug_overlay` on raw
  GVHMR params. GVHMR wrist dots MUST land on the wrists in the frame. If not, fix
  `project_incam` / the incam-frame assumption / K BEFORE anything else.

- **STEP 2 — wrist replacement.** Run `replace_wrists`, overlay `after_wrist`.
  Wrist *position* should be ~unchanged (only rotation changed). If the hand
  teleports, `G_w` frame handling or the mirror is wrong. Test a left-hand frame
  specifically for the mirror.

- **STEP 3 — targets.** Implement the Option-1 target builder (resolve the OPEN ITEM
  in §4 first). Overlay the TARGET uv (not the prediction) and confirm targets sit on
  the actual hand. Bad targets → refinement pulls hands to the wrong place.

- **STEP 4 — refine.** Run `refine_2d` with `w_tc=0` first (pure data fit); confirm
  `L_h` drops and hands snap toward targets in `after_refine`. THEN restore
  `w_tc=1e4`. Jitter → w_tc too low; stiff/frozen → too high.

  **Early stopping (required — do NOT rely on a fixed 300-iter loop).**
  Keep `iters=300` as the MAX, but stop early based on **hand reprojection RMS pixel
  error (`hand_rms_px`)**, NOT total loss. Total loss is not a reliable stop signal:
  `w_tc=1e4` dominates the objective, so a falling total loss can just mean the
  temporal term is shrinking while hand alignment stalls. Compute `hand_rms_px`
  separately from the (detached) hand reprojection error each iteration.
  Defaults: `iters=300, min_iter=60, patience=30, target_px=12.0, min_delta=0.1,
  restore_best=True`.
  Stop when:
  - `it >= min_iter` AND `hand_rms_px <= target_px`, OR
  - `it >= min_iter` AND best `hand_rms_px` has not improved by >= `min_delta` for
    `patience` iterations (plateau).
  **Save and RESTORE the best-`hand_rms_px` parameter state** (`body_free`, `lhand`,
  `rhand`), not the last iteration's state. Log per checkpoint: `loss, L_h, L_b,
  L_tc, hand_rms_px, best_hand_rms_px, bad_count, stop_reason`. Persist the final
  `hand_rms_px`, stop_reason, and iter count into refinement metadata on the `.pt`.
  Caveats:
  - `target_px=12.0` is resolution/hand-size dependent — it's a starting default.
    Consider normalizing the threshold by hand-box size (e.g. % of box diagonal)
    rather than an absolute pixel count, especially across clips where the hand is
    large vs small.
  - Under Option 1 the target IS HaMeR reprojection, so `hand_rms_px` is partly
    self-referential (target and metric share HaMeR's error) — use it as a stopping
    convenience, NOT as a quality claim. Under Option 2 the detector keypoints are
    an independent reference, so there `hand_rms_px` is a meaningful quality number
    (and is what STEP 6 should report).

- **STEP 5 — render the refined .pt.** Hand post-process (`postprocess_hand_sequence`)
  already ran INSIDE `merge_hamer_hands_into_gvhmr_postprocessed` to produce the
  baseline `smplx_merged_hamer_post.pt`; wrist refine ran on top of that. After
  STEP 4 overlays look right, render the refined `.pt`
  (`smplx_merged_hamer_wrist_refined.pt`). Do NOT overwrite the baseline — STEP 6
  needs it.

- **STEP 6 — comparison & evaluation (the deliverable).** The point of this project
  is to BEAT the existing finger-only merge, NOT to match DeVI. So the baseline you
  must out-perform is the repo's current `merge_hamer_hands_into_gvhmr`, not DeVI.
  Produce at least four variants on the SAME clip and compare them on the SAME frames:
    1. `baseline`              — existing finger-only merge (GVHMR wrist kept)
    2. `wrist_only`            — Stage A only (wrist replacement, no refine)
    3. `wrist_opt_arm_only`    — Stage A + Stage B1 arm-chain-only refine
    4. `wrist_opt_hand_reg`    — Stage A + Stage B2 hand-pose-regularized refine
    5. `wrist_opt_detector`    — optional: Stage B using Option-2 detector targets, if implemented
  Save each as its own `.pt` and render each to video.

  **Quantitative metric (guards against "looks good" self-deception):**
  report **2D hand-keypoint reprojection error in pixels** for each variant, measured
  on the same frames. Project each variant's SMPL-X hand joints with `K_fullimg` and
  compare to a reference 2D hand keypoint set. If Option 2 is implemented, its detector
  keypoints ARE the natural ground-truth for this metric (so Option 2 doubles as both
  target source and evaluator). If only Option 1 exists, this metric is weaker (the
  target and the evaluator share HaMeR's error) — note that caveat explicitly and lean
  on the qualitative overlays instead.
  Also report a simple **temporal jitter** number per variant: mean per-frame change of
  hand-joint position (or wrist rotation geodesic) across adjacent frames — lower = smoother.

  **Qualitative pass criteria** (vs the `baseline`, not vs DeVI): the wrist–finger
  junction bends less unnaturally, fingers continue naturally from the wrist direction,
  and frame-to-frame hand jitter is visibly reduced. Put the four renders side by side.
  Do NOT declare success from loss curves — use the renders + the two numbers above.

- **If Option-1 overlays stay biased after honest debugging:** set
  `target_mode="option2"`, add the detector (prefer ViTPose-wholebody to reuse the
  existing VitPose stack; MediaPipe Hands is the lighter alternative), implement the
  Option-2 target builder, and rerun from STEP 3. Nothing else changes.

---

## 6. TODO[verify] / TODO[implement] markers in the skeleton

- `[verify]` STEP 0 gate: call `verify_wrist_chains(smplx_model, smpl_params_incam)`
  (already auto-called at the top of `run_wrist_refine`). It prints `bm.parents`,
  derived wrist parent chains (LEFT/RIGHT), and asserts `body_pose` is (F,21,3) or
  (F,63). If chains differ from the hardcoded `LEFT/RIGHT_CHAIN_TO_ELBOW`, the
  function warns; investigate before proceeding.
- `[verify]` `out.joints` are in incam frame (STEP 1 overlay).
- `[verify]` left-hand mirror convention `mirror_rotmat_right_to_left` (STEP 2,
  test on a frame where only the LEFT hand is detected).
- `[verify]` `project_incam` correctly handles both `K.shape == (3,3)` and
  `K.shape == (F,3,3)`. Already implemented; just confirm the STEP 1 overlay lands
  on wrists across MULTIPLE frames (a buggy K-handling would only show up on some
  frames if intrinsics vary).
- `[implement]` HaMeR keypoint save step:
  HaMeR `out.keys()` was already checked in the current RunPod environment.
  Confirmed keys:
  - `pred_keypoints_3d` exists, shape = `torch.Size([1, 21, 3])`
  - `pred_keypoints_2d` exists, shape = `torch.Size([1, 21, 2])`
  Do NOT repeat the debug-key inspection unless the HaMeR build/environment changes.
  Extend the `np.savez(...)` call in `entry_point_hamer.py` only:
  ```python
  keypoints_3d=out["pred_keypoints_3d"][n].detach().cpu().numpy(),
  keypoints_2d=out["pred_keypoints_2d"][n].detach().cpu().numpy(),
  ```
  Then re-run HaMeR once to regenerate the npz files, and update
  `load_hamer_detections_full()` to read `keypoints_3d`.
  MANO forward fallback is not needed in the current RunPod environment unless this
  key disappears in another HaMeR build.
- `[implement]` `build_hand_targets_option1_hamer`: read `keypoints_3d` from the
  extended npz (or derive from `mano_params` if joints aren't saved), lift to incam
  using `cam_t_full`, project with `K_fullimg`.
- `[implement]` integrate `run_wrist_refine` into `entry_point_hamer.py`'s
  `main()`: AFTER `merge_hamer_hands_into_gvhmr_postprocessed` produces
  `smplx_merged_hamer_post.pt`, call `run_wrist_refine(merged_input_path=<that>,
  output_path=<smplx_merged_hamer_wrist_refined.pt>)`, then render the refined
  file. **Do NOT overwrite the baseline `smplx_merged_hamer_post.pt`** — STEP 6
  needs all three variants on disk. Leave `entry_point.py` untouched.
- `[verify]` `run_wrist_refine`'s first arg is `merged_input_path` (a merged `.pt`,
  NOT a raw GVHMR result). Reading the wrong file silently disables the wrist
  swap.
- `[verify]` sync scope per stage:
  - After Stage A → `sync_wrist_rows(incam, global)` (wrist rows only).
  - After Stage B → `sync_refined_local_pose(incam, global)` (UPPER_BODY_OPT_IDS
    rows of `body_pose` PLUS both hand poses).
  Both are auto-called inside `run_wrist_refine`. Confirm by inspecting
  `smpl_params_global["body_pose"][:, i]` for an UPPER_BODY_OPT_IDS index AND
  `smpl_params_global["left_hand_pose"]` before/after the refine call.
- `[implement]` early stopping in `refine_2d`: compute `hand_rms_px` (detached) each
  iter, stop by `target_px` / plateau (`min_iter`/`patience`/`min_delta`), RESTORE
  best-`hand_rms_px` state (`body_free`/`lhand`/`rhand`), log the full metric set, and
  persist refinement metadata (`hand_rms_px`, `stop_reason`, iters) onto the `.pt`.
  NOTE: measure `hand_rms_px`, capture `best_state`, and decide to stop ALL **before**
  `opt.step()`, then step only if not stopping — otherwise the recorded metric and the
  saved state are one update apart. `refine_2d` also raises if hand targets are empty.
  `refine_2d` returns `(refined_params, refine_meta)`: `refined_params` is pure-tensor
  (NOT carrying the meta dict inside, since downstream render/post-process may assume
  every value in `smpl_params_incam` is a tensor); the caller stores `refine_meta` under
  `cap_wrist_refine_meta["refine"]`. Meta includes both `best_hand_rms_px` and
  `final_hand_rms_px`. `pack_targets` asserts the joint-id set is identical every frame
  (uv/conf are stacked against one id list, so a per-frame change would desync indices).
- `[verify]` STEP 3 uses `debug_target_overlay` (draws TARGET uv), not `debug_overlay`
  (draws predictions). It is wired right after target building in `run_wrist_refine`.
- `[implement]` (if needed) `build_hand_targets_option2_detector` + detector in preproc.
- `[implement]` STEP 6 comparison harness. A small helper inside
  `entry_point_hamer.py` (or a new `compare_variants.py`) that on the same input:
  1. Saves baseline as-is (the existing `smplx_merged_hamer_post.pt`).
  2. Runs `run_wrist_refine(do_refine=False)` → `smplx_merged_hamer_wrist_only.pt`.
  3. Runs `run_wrist_refine(do_refine=True, target_mode="option1")` →
     `smplx_merged_hamer_wrist_refined.pt`.
  4. (optional) `run_wrist_refine(do_refine=True, target_mode="option2")`.
  5. Renders each `.pt` to its own mp4.
  6. Computes 2D hand reprojection error (px) + temporal jitter per variant on
     identical frames, writes a small JSON report.

---

## 7. Escalate to the human (don't guess) when:

- Option-1 overlays remain biased after the mirror/frame checks pass — this is a
  design call (switch to Option 2), not a code bug.
- A coordinate transform "works" numerically but you're unsure it matches DeVI's
  intent — the frame semantics are subtle; confirm rather than assume.
- `bm.parents` / SMPL-X variant disagrees with the assumptions in §2/§3.
