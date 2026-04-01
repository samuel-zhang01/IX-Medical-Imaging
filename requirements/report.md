# Report Plan: Uncertainty-Aware Deep MRI Reconstruction

## Proposed Title

**"Uncertainty-Aware Deep MRI Reconstruction: A Multi-Faceted Trustworthiness Analysis with Downstream Task Evaluation"**

**Keywords:** MRI reconstruction, uncertainty quantification, MC Dropout, Deep Ensembles, trustworthy AI, cardiac imaging, data consistency

---

## Paper Metadata

- **Format:** Springer LNCS, 8 pages total (max 1 page references)
- **Content pages:** ~7 pages of text, figures, and tables
- **Deadline:** 3 April 2026, 12:00 midday
- **Dataset:** MM-WHS cardiac dataset (MR: 1738 train / 254 val / 236 test slices; CT: 3389 train / 382 val / 484 test slices; all 256x256 with segmentation labels)
- **Evaluation weighting:** Literature Review 20%, Methods + Results 40% (must include trustworthiness), Discussion 20%, Bonus innovation 20%

---

## Figure and Table Plan (Total: 4 figures, 3 tables)

| ID | Type | Content | Section | Size |
|----|------|---------|---------|------|
| Fig. 1 | Diagram | System architecture: U-Net with DC layer, MC Dropout paths, Deep Ensemble overview, downstream segmentation pipeline | Methods (Sec 3) | Full width |
| Fig. 2 | Grid | Reconstruction visual comparison: ground truth, zero-filled, U-Net, U-Net+DC at 4x/8x, with error maps and uncertainty maps | Results (Sec 4) | Full width |
| Fig. 3 | Multi-panel | Uncertainty analysis: (a) calibration plot, (b) uncertainty vs error scatter, (c) uncertainty maps at different acceleration factors, (d) MR vs CT domain shift uncertainty | Results (Sec 4) | Full width |
| Fig. 4 | Side-by-side | Downstream segmentation: reconstructed images + predicted segmentations + ground truth labels, for different reconstruction methods and acceleration factors | Results (Sec 4) | Full width |
| Tab. 1 | Table | Reconstruction quality metrics (PSNR, SSIM, NMSE) across methods and acceleration factors | Results (Sec 4) | Single col |
| Tab. 2 | Table | Uncertainty quality metrics (ECE, UCE, AUSE, Pearson/Spearman correlation with error) for MC Dropout vs Deep Ensemble | Results (Sec 4) | Single col |
| Tab. 3 | Table | Downstream segmentation Dice scores on reconstructed images vs on ground truth, across methods and acceleration factors | Results (Sec 4) | Single col |

---

## Section-by-Section Detailed Plan

---

### ABSTRACT (~150-200 words)

**Target word count:** 180 words

**Content:**
- Opening: MRI reconstruction from undersampled k-space is a critical clinical task, but most deep learning methods provide point estimates without uncertainty quantification, undermining clinical trust.
- Gap: Current DL-based MRI reconstruction lacks systematic evaluation of trustworthiness properties -- uncertainty calibration, robustness to distribution shifts, and downstream task reliability.
- What we do: We present a comprehensive trustworthiness analysis of U-Net-based MRI reconstruction with data consistency, comparing MC Dropout and Deep Ensemble uncertainty quantification on the MM-WHS cardiac dataset.
- Key findings (3 points):
  1. Both uncertainty methods produce well-calibrated uncertainty estimates that correlate strongly with reconstruction error.
  2. Reconstruction quality and uncertainty calibration degrade predictably under increasing acceleration and cross-domain (MR-to-CT) distribution shift, enabling reliable failure detection.
  3. Downstream cardiac segmentation performance is preserved at moderate acceleration factors but degrades at aggressive undersampling, where uncertainty estimates serve as effective quality gatekeepers.
- Closing: Our multi-faceted evaluation framework provides a template for trustworthiness assessment in clinical MRI reconstruction pipelines.

**Evaluation criteria addressed:** Sets up the trustworthiness framing (mandatory requirement), highlights the multi-faceted evaluation approach (bonus innovation).

---

### 1. INTRODUCTION (~0.75 pages, ~550 words)

**Target word count:** 550

#### Paragraph 1 -- Clinical motivation and inverse problem (120 words)
- MRI is indispensable for soft-tissue imaging (cardiac, brain, abdominal) but suffers from long acquisition times.
- Prolonged scans cause patient discomfort, motion artifacts, and limit throughput [cite Safari et al. 2025: estimated $364,000/year cost from motion artifacts].
- Accelerated MRI via k-space undersampling is clinically important but poses an ill-posed inverse problem: infinitely many images are consistent with the acquired measurements.
- Compressed sensing (CS) and deep learning (DL) approaches have been developed to solve this inverse problem [cite Safari et al. 2025, Lustig et al. 2007].

#### Paragraph 2 -- DL reconstruction progress and trust gap (150 words)
- DL methods (U-Net, unrolled networks like VarNet, MoDL) have shown state-of-the-art performance on benchmarks like fastMRI [cite Zbontar et al. 2018].
- End-to-end approaches learn mappings from undersampled to fully-sampled images; unrolled optimization networks interleave learned regularization with data consistency steps [cite Hammernik et al. 2018, Aggarwal et al. 2019].
- However, most methods produce point estimates without quantifying reconstruction uncertainty.
- **Critical trust gap:** In clinical settings, clinicians need to know *when* a reconstruction can be trusted. A high-quality average reconstruction metric (PSNR/SSIM) across a test set does not guarantee reliability for individual patients.
- Recent work by Atalik et al. (2026) on Trust-Guided Variational Networks (TGVN) highlights the importance of disambiguating solutions in the null space of the forward operator, but focuses on side information rather than explicit uncertainty quantification.

#### Paragraph 3 -- What is trustworthiness in this context (120 words)
- Trustworthy AI in medical imaging encompasses multiple pillars: uncertainty quantification, robustness, fairness, explainability, and safety [cite Kaur et al. 2022, general TAI references].
- For MRI reconstruction specifically, we argue three pillars are most critical:
  1. **Uncertainty quantification** -- Does the model know what it does not know?
  2. **Robustness** -- How does performance degrade under distribution shift (acceleration factor changes, noise, cross-domain)?
  3. **Downstream reliability** -- Do reconstruction artifacts propagate into clinical tasks (e.g., cardiac segmentation)?
- A trustworthy reconstruction pipeline must address all three.

#### Paragraph 4 -- Our contributions (160 words)
- We present a systematic, multi-faceted trustworthiness evaluation of DL-based MRI reconstruction. Our contributions are:
  1. **Uncertainty-aware reconstruction:** We implement and compare two principled UQ methods -- MC Dropout and Deep Ensembles -- on top of a U-Net with data consistency baseline, quantifying both aleatoric and epistemic uncertainty in the reconstruction.
  2. **Robustness analysis:** We systematically evaluate how reconstruction quality and uncertainty calibration degrade across acceleration factors (4x, 8x, 12x), additive noise levels, and cross-domain transfer (MR to CT using the MM-WHS multi-modal dataset).
  3. **Downstream task evaluation:** We measure how reconstruction quality propagates to cardiac segmentation accuracy using the paired segmentation labels in MM-WHS, and show that uncertainty estimates can predict segmentation failures.
  4. **Practical trust framework:** We propose an uncertainty-guided quality gating mechanism that flags unreliable reconstructions before they reach downstream clinical tasks.

**Evaluation criteria addressed:** Literature Review (contextualizes within state-of-the-art), Methods (previews approach), Bonus (multi-faceted evaluation + quality gating is novel for a coursework-scale project).

**No figures or tables in this section.**

---

### 2. RELATED WORK (~1 page, ~750 words)

**Target word count:** 750

#### 2.1 DL-Based MRI Reconstruction (~250 words)

**Content to include:**
- Classical CS-MRI formulation: min ||y - Ax||_2 + lambda * R(x), where A is the forward operator (undersampling mask * Fourier transform), R is a regularizer [cite Lustig et al. 2007].
- End-to-end approaches: U-Net [Ronneberger et al. 2015] applied directly to zero-filled images; simple but effective baseline. Mention AUTOMAP [Zhu et al. 2018] as a fully learned approach.
- Data consistency (DC) layers: Enforcing fidelity to acquired k-space measurements by replacing acquired frequencies in the network output [cite Schlemper et al. 2017]. This is critical -- without DC, networks can hallucinate structures not supported by the data.
- Unrolled optimization networks: VarNet [Hammernik et al. 2018], MoDL [Aggarwal et al. 2019], FISTA-Net [Xiang et al. 2021] -- these interleave learned CNN blocks with DC steps, mimicking iterative optimization. Safari et al. (2025) provide a comprehensive taxonomy.
- TGVN [Atalik et al. 2026]: Trust-Guided Variational Network that leverages side information to disambiguate solutions in the null space. Introduces "ambiguous space consistency" constraint. Relevant as a trust-oriented reconstruction approach, though it focuses on side information rather than uncertainty quantification.

**Key argument:** DL methods have advanced rapidly but overwhelmingly focus on point estimation. Even the trust-oriented TGVN focuses on improving reconstruction reliability through side information, not on quantifying when the reconstruction should not be trusted.

#### 2.2 Trustworthy AI in Medical Imaging (~250 words)

**Content to include:**
- Pillars of trustworthy AI: robustness, uncertainty, fairness, explainability, privacy [cite EU AI Act framework, Kaur et al. 2022].
- Uncertainty quantification in medical image analysis: primarily studied in segmentation [cite Jungo et al. 2020, Mehrtash et al. 2020] and classification, far less in reconstruction.
- Types of uncertainty: aleatoric (data-inherent, e.g., noise in k-space) vs epistemic (model-related, reducible with more data) [cite Kendall and Gal 2017].
- MC Dropout for UQ: Gal and Ghahramani (2016) showed dropout at test time approximates Bayesian inference. Applied in medical imaging segmentation [cite Nair et al. 2020] but sparsely in MRI reconstruction.
- Deep Ensembles: Lakshminarayanan et al. (2017) proposed training multiple models with different initializations. Shown to be competitive or superior to Bayesian approaches for calibration [cite Ovadia et al. 2019].
- Calibration metrics: Expected Calibration Error (ECE), uncertainty-error correlation, Area Under Sparsification Error (AUSE) [cite references from UQ literature].
- Gap: Comprehensive comparison of UQ methods specifically for MRI reconstruction, with evaluation extending to calibration quality and downstream task impact, is largely absent.

**Key argument:** Trustworthy AI is an active area in medical imaging broadly, but uncertainty quantification for MRI reconstruction specifically is underexplored, and the connection between reconstruction uncertainty and downstream task reliability has not been systematically studied.

#### 2.3 Cross-Domain Robustness and Distribution Shift (~250 words)

**Content to include:**
- Distribution shift is a critical concern for DL in medical imaging: models trained on one scanner, protocol, or anatomy may fail on another [cite Stacke et al. 2020].
- In MRI reconstruction, distribution shift arises from: different acceleration factors at test time, different noise levels, different anatomies, different imaging modalities (MR vs CT).
- The MM-WHS dataset provides a natural cross-domain testbed: paired MR and CT cardiac images with segmentation labels. Training on MR and testing on CT (after appropriate preprocessing) provides a controlled domain shift experiment.
- Robustness to acceleration factor mismatch: training at 4x and testing at 8x or 12x is clinically relevant since radiologists may change protocols.
- Connection to uncertainty: a well-calibrated model should produce higher uncertainty when encountering out-of-distribution data. This is the "uncertainty as a trust signal" idea.
- Downstream evaluation: degraded reconstruction under domain shift should predictably reduce segmentation accuracy, and uncertainty should correlate with this degradation.

**Key argument:** Cross-domain robustness evaluation using the multi-modal MM-WHS dataset is a natural and underexploited experimental design that directly tests trustworthiness claims.

**Evaluation criteria addressed:** Literature Review (20%) -- This section is the core of the literature review grade. Depth is demonstrated through: (a) systematic coverage of DL reconstruction taxonomy, (b) clear mapping of trustworthy AI pillars to MRI reconstruction, (c) identification of specific gaps. Bonus -- The cross-domain framing using MM-WHS multi-modal data is an innovative experimental design.

**References to cite in this section (~12-15):**
- Safari et al. (2025) -- DL+CS MRI review
- Atalik et al. (2026) -- TGVN
- Lustig et al. (2007) -- CS-MRI
- Ronneberger et al. (2015) -- U-Net
- Schlemper et al. (2017) -- DC-CNN for MRI
- Hammernik et al. (2018) -- VarNet
- Aggarwal et al. (2019) -- MoDL
- Zbontar et al. (2018) -- fastMRI
- Gal and Ghahramani (2016) -- MC Dropout
- Lakshminarayanan et al. (2017) -- Deep Ensembles
- Kendall and Gal (2017) -- Aleatoric vs epistemic
- Ovadia et al. (2019) -- Ensemble calibration
- Nair et al. (2020) -- MC Dropout in medical imaging
- Zhuang (2018/2019) -- MM-WHS dataset/challenge

**No figures or tables in this section.**

---

### 3. METHODS (~2 pages, ~1500 words)

**Target word count:** 1500

**Figure in this section:** Fig. 1 (system architecture diagram) -- place near start of section

#### 3.1 Problem Formulation (~200 words)

**Content:**
- Formalize the MRI reconstruction inverse problem for single-coil magnitude images (matching the MM-WHS data format).
- Let x be the ground truth image (256 x 256). The forward model for single-coil acquisition is:
  - y = M * F * x + noise
  - where F is the 2D Fourier transform, M is the binary undersampling mask, y is the acquired k-space.
- The zero-filled reconstruction is: x_zf = F^H * M^H * y (inverse Fourier of zero-padded k-space).
- The goal is to learn a reconstruction function f_theta(x_zf) that approximates x.
- Note: the MM-WHS data provides real-valued magnitude images. We simulate the k-space acquisition pipeline: take FFT of ground truth, apply undersampling mask, add optional noise, take IFFT to get zero-filled input.
- Undersampling masks: random Cartesian with center fraction preserved (e.g., 8% center lines), at acceleration factors R = {4, 8, 12}.

**Key equations:**
```
y = M * F(x) + epsilon
x_zf = F^{-1}(y)
x_hat = f_theta(x_zf)
```

#### 3.2 U-Net Baseline with Data Consistency (~250 words)

**Content:**
- Architecture: Standard U-Net encoder-decoder with skip connections.
  - Encoder: 4 downsampling blocks, each with 2 conv layers (3x3, ReLU, BatchNorm) + 2x2 max pool.
  - Channel progression: 64 -> 128 -> 256 -> 512 -> 1024 (bottleneck).
  - Decoder: symmetric upsampling path with transposed convolutions and skip connections.
  - Final 1x1 conv to single-channel output.
  - Input: zero-filled reconstruction (single channel, 256x256).
  - Output: reconstructed image (single channel, 256x256).
- Data Consistency (DC) layer: After the U-Net output, enforce fidelity to acquired k-space measurements.
  - x_dc = F^{-1}( M * F(x_gt) + (1-M) * F(x_unet) )
  - Acquired k-space lines are kept unchanged; only unacquired lines come from the network prediction.
  - This is critical for trustworthiness: DC ensures the reconstruction is consistent with actual measurements and prevents hallucination of structures in the acquired frequency bands.
- Loss function: L1 loss + SSIM loss (weighted combination):
  - L_total = alpha * L1(x_hat, x_gt) + (1 - alpha) * (1 - SSIM(x_hat, x_gt))
  - alpha = 0.85 following common practice.

**Key argument for trustworthiness:** DC layer is a physics-informed constraint that provides a hard guarantee: the reconstruction is consistent with the measured data. This is the most basic form of trustworthiness in MRI reconstruction.

#### 3.3 MC Dropout Uncertainty Quantification (~250 words)

**Content:**
- Method: Add Dropout layers (p=0.1-0.2) after each encoder and decoder block in the U-Net. At inference time, keep dropout active and perform T forward passes (T=20) for each input.
- Mean reconstruction: x_mean = (1/T) * sum(x_t) for t=1..T
- Pixel-wise uncertainty (epistemic): sigma^2(i,j) = (1/T) * sum((x_t(i,j) - x_mean(i,j))^2) for t=1..T
- This approximates the predictive variance under a Bayesian interpretation [Gal and Ghahramani, 2016].
- Interpretation: High variance regions indicate where the model is uncertain -- typically edges, fine structures, and regions with fewer k-space measurements (high-frequency content lost by undersampling).
- Apply DC layer after each stochastic forward pass (not after averaging) to ensure each sample is data-consistent.
- Computational cost: T times the single forward pass cost, but trivially parallelizable.
- Hyperparameters to report: dropout rate p, number of MC samples T, and ablation showing convergence of uncertainty estimate with T.

**Key argument:** MC Dropout captures epistemic uncertainty -- the model's lack of knowledge about the correct reconstruction in undersampled regions. It is lightweight (requires only adding dropout layers to the existing architecture) and provides spatially resolved uncertainty maps.

#### 3.4 Deep Ensemble Uncertainty Quantification (~250 words)

**Content:**
- Method: Train N independently initialized U-Net+DC models (N=5) with different random seeds for weight initialization and data shuffling.
- Each ensemble member m produces a reconstruction x_m.
- Mean reconstruction: x_mean = (1/N) * sum(x_m) for m=1..N
- Pixel-wise uncertainty: sigma^2(i,j) = (1/N) * sum((x_m(i,j) - x_mean(i,j))^2) for m=1..N
- Deep Ensembles capture both epistemic uncertainty (disagreement between models) and, if each member is trained with a heteroscedastic loss, aleatoric uncertainty as well.
- For our experiments, we use the standard approach of training each member with the same loss function but different initialization, which primarily captures epistemic uncertainty through functional diversity.
- Computational cost: N times the training cost, but inference can be parallelized. Storage cost is N times a single model.
- Comparison with MC Dropout:
  - Ensembles explore different loss landscape modes; MC Dropout samples around a single mode.
  - Ensembles are generally better calibrated [Ovadia et al. 2019] but more expensive.
  - Both provide pixel-wise uncertainty maps suitable for spatial analysis.

**Key argument:** Deep Ensembles are the gold-standard baseline for UQ. Comparing MC Dropout (cheap, approximate) with Deep Ensembles (expensive, higher quality) characterizes the cost-quality tradeoff for trustworthiness in MRI reconstruction.

#### 3.5 Uncertainty Evaluation Framework (~250 words)

**Content:**
- Define how we evaluate the quality of uncertainty estimates (not just reconstruction quality):
  1. **Calibration:** For a well-calibrated model, predicted uncertainty should match observed error. Measure via Expected Calibration Error (ECE) adapted for regression: bin pixels by predicted uncertainty, compute mean absolute error in each bin, compare.
  2. **Uncertainty-Error Correlation:** Pearson and Spearman correlation between pixel-wise uncertainty sigma(i,j) and pixel-wise absolute error |x_hat(i,j) - x_gt(i,j)|. A trustworthy model should show strong positive correlation.
  3. **Sparsification/Oracle Error:** Remove pixels in order of decreasing uncertainty and plot remaining error vs fraction of pixels removed. Compare against the oracle (removing pixels in order of actual error). The closer the uncertainty curve to the oracle, the better the UQ. Quantify via Area Under the Sparsification Error (AUSE).
  4. **Out-of-Distribution Detection:** Use mean uncertainty as a score to distinguish in-distribution (MR) from out-of-distribution (CT) reconstructions. Measure AUROC.

**Key argument:** Evaluating uncertainty quality is as important as evaluating reconstruction quality for trustworthiness. These four metrics provide complementary views of UQ reliability.

#### 3.6 Downstream Segmentation Evaluation Protocol (~200 words)

**Content:**
- Motivation: The ultimate clinical value of MRI reconstruction is the diagnostic information it enables. Cardiac segmentation is a direct downstream task available in MM-WHS (paired images + labels).
- Protocol:
  1. Train a separate U-Net segmentation model on ground truth (fully sampled) MR images and their segmentation labels.
  2. At test time, feed reconstructed images (from different methods and acceleration factors) through the frozen segmentation model.
  3. Measure Dice score, Hausdorff distance for each cardiac structure (myocardium, blood pool, etc.).
  4. Correlate segmentation degradation with reconstruction uncertainty: do high-uncertainty reconstructions predict segmentation failures?
- This "reconstruction -> segmentation" pipeline is a proxy for clinical workflow, where reconstruction artifacts can lead to diagnostic errors.
- Innovation: Using reconstruction uncertainty as a *gating signal* -- if mean uncertainty exceeds a threshold, flag the reconstruction for manual review before it enters the segmentation pipeline.

**Key argument:** Connecting reconstruction to downstream task evaluation goes beyond standard MRI reconstruction benchmarking and directly addresses clinical trustworthiness.

**Evaluation criteria addressed:** Methods and Implementation Results (40%) -- This is the core of the 40% grade. Key elements: (a) clear mathematical formulation, (b) two distinct UQ methods (MC Dropout + Deep Ensembles), (c) DC layer for physics-informed trust, (d) rigorous UQ evaluation framework, (e) downstream task connection. Bonus -- Uncertainty-guided quality gating is an innovative contribution; the downstream segmentation evaluation adds clinical relevance beyond standard reconstruction metrics.

---

### 4. EXPERIMENTS AND RESULTS (~2.5 pages, ~1800 words)

**Target word count:** 1800

**Figures in this section:** Fig. 2, Fig. 3, Fig. 4
**Tables in this section:** Tab. 1, Tab. 2, Tab. 3

#### 4.1 Dataset and Preprocessing (~200 words)

**Content:**
- MM-WHS cardiac dataset [Zhuang 2018]: multi-modal whole heart segmentation challenge.
- MR data: 1738 training / 254 validation / 236 test slices, 256x256 pixels, with 7-class segmentation labels (LV myocardium, LV blood cavity, RV blood cavity, LA blood cavity, RA blood cavity, ascending aorta, pulmonary artery).
- CT data: 3389 training / 382 validation / 484 test slices, 256x256 pixels, same label classes. Used for cross-domain robustness evaluation only.
- k-space simulation: Apply 2D FFT to ground truth images to obtain fully-sampled k-space. Apply random Cartesian undersampling masks with center fraction = 0.08 at acceleration factors R = {4, 8, 12}. Optional additive complex Gaussian noise at SNR levels {30, 20, 10} dB.
- Normalization: images normalized to [0, 1] range per-slice.
- Data augmentation: random horizontal/vertical flips, random rotation (+/- 10 degrees) during training.

#### 4.2 Implementation Details (~200 words)

**Content:**
- Framework: PyTorch.
- U-Net architecture: encoder channels [64, 128, 256, 512], bottleneck 1024, decoder symmetric. Total parameters: ~31M.
- MC Dropout: dropout rate p = 0.1 after each conv block; T = 20 MC samples at inference.
- Deep Ensemble: N = 5 independently trained models with different random seeds.
- Optimizer: Adam, learning rate 1e-4, weight decay 1e-5.
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5).
- Batch size: 8.
- Training: 100 epochs, early stopping on validation SSIM (patience=10).
- Loss: 0.85 * L1 + 0.15 * (1 - SSIM).
- Segmentation model: Separate U-Net trained on ground truth images, 50 epochs, cross-entropy + Dice loss, frozen at evaluation time.
- Hardware: single GPU (note actual GPU used).
- Training time: ~X hours per model (to be filled after experiments).

#### 4.3 Reconstruction Quality Results (~350 words)

**Placement:** Tab. 1, Fig. 2

**Table 1 -- Reconstruction Quality Metrics:**

| Method | R=4x PSNR | R=4x SSIM | R=4x NMSE | R=8x PSNR | R=8x SSIM | R=8x NMSE | R=12x PSNR | R=12x SSIM | R=12x NMSE |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|
| Zero-filled | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| U-Net (no DC) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| U-Net + DC | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| MC Dropout (mean) | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Deep Ensemble (mean) | -- | -- | -- | -- | -- | -- | -- | -- | -- |

**Expected narrative:**
- DC layer provides consistent improvement over vanilla U-Net across all acceleration factors (validates physics-informed trustworthiness).
- MC Dropout mean is slightly below the deterministic U-Net+DC (due to dropout regularization).
- Deep Ensemble mean is expected to match or slightly exceed U-Net+DC (ensemble averaging smooths predictions).
- All DL methods substantially outperform zero-filled baseline.
- Performance degrades gracefully from 4x to 8x, with sharper decline at 12x.

**Figure 2 -- Visual Comparison:**
- Grid layout: rows = {Ground Truth, Zero-Filled, U-Net, U-Net+DC, MC Dropout mean, Ensemble mean}, columns = {R=4x, R=8x, R=12x}.
- Include error maps (absolute difference from GT) as a separate row or overlay.
- Include uncertainty maps for MC Dropout and Ensemble in a separate row.
- Point out: uncertainty is highest at edges and fine cardiac structures; uncertainty increases with acceleration factor.

#### 4.4 Uncertainty Quality Results (~350 words)

**Placement:** Tab. 2, Fig. 3

**Table 2 -- Uncertainty Quality Metrics:**

| Method | R=4x ECE | R=4x Corr | R=4x AUSE | R=8x ECE | R=8x Corr | R=8x AUSE | R=12x ECE | R=12x Corr | R=12x AUSE |
|--------|----------|-----------|-----------|----------|-----------|-----------|-----------|------------|------------|
| MC Dropout | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Deep Ensemble | -- | -- | -- | -- | -- | -- | -- | -- | -- |

**Expected narrative:**
- Deep Ensembles expected to show better calibration (lower ECE) and tighter uncertainty-error correlation than MC Dropout, consistent with literature [Ovadia et al. 2019].
- Both methods show increasing uncertainty at higher acceleration factors -- this is desirable behavior: the model "knows" that 12x reconstruction is less reliable.
- AUSE should be low for both methods, with Ensemble slightly better.
- Discuss where uncertainty is well-calibrated vs where it breaks down.

**Figure 3 -- Uncertainty Analysis (4-panel):**
- (a) Calibration plot: predicted uncertainty quantile vs observed error quantile. Perfect calibration = diagonal. Show MC Dropout and Ensemble curves.
- (b) Scatter plot: pixel-wise uncertainty (x-axis) vs pixel-wise absolute error (y-axis), with density coloring. Show Pearson r value.
- (c) Uncertainty maps at R = {4, 8, 12} for the same slice, showing how uncertainty grows with acceleration. Use a perceptually uniform colormap (viridis).
- (d) Uncertainty histogram for in-distribution (MR test) vs out-of-distribution (CT test), showing that OOD data triggers higher uncertainty.

#### 4.5 Robustness Analysis (~350 words)

**Content (no additional figure -- integrated into Tab. 1 and Fig. 3d):**

- **Acceleration factor robustness:** Already captured in Tab. 1. Discuss the rate of PSNR/SSIM degradation. A trustworthy model degrades gracefully, not catastrophically.
- **Noise robustness:** Test models trained at SNR=inf on inputs with SNR = {30, 20, 10} dB. Report PSNR drop and uncertainty increase. Expected: uncertainty maps correctly highlight noise-corrupted regions.
- **Cross-domain robustness (MR -> CT):** Apply the MR-trained reconstruction model to CT images (after preprocessing to match input range/format). Expected: substantial PSNR/SSIM drop (CT has very different intensity distributions and contrast). Crucially, uncertainty should spike, signaling untrustworthiness. Report AUROC for OOD detection using mean uncertainty.
- **Acceleration factor mismatch:** Train at R=4x, test at R=8x (without retraining). Expected: performance degradation, but well-calibrated uncertainty should increase proportionally.

**Key argument:** A trustworthy model should not silently fail. Across all robustness experiments, we expect uncertainty to increase when reliability decreases. This predictable behavior is the hallmark of trustworthiness.

#### 4.6 Downstream Segmentation Results (~350 words)

**Placement:** Tab. 3, Fig. 4

**Table 3 -- Downstream Segmentation Dice Scores:**

| Input Source | R | Mean Dice | LV Myo | LV Blood | RV Blood | LA Blood | RA Blood |
|-------------|---|-----------|---------|----------|----------|----------|----------|
| Ground Truth | -- | -- | -- | -- | -- | -- | -- |
| U-Net+DC | 4x | -- | -- | -- | -- | -- | -- |
| U-Net+DC | 8x | -- | -- | -- | -- | -- | -- |
| U-Net+DC | 12x | -- | -- | -- | -- | -- | -- |
| MC Dropout (mean) | 4x | -- | -- | -- | -- | -- | -- |
| MC Dropout (mean) | 8x | -- | -- | -- | -- | -- | -- |
| Ensemble (mean) | 4x | -- | -- | -- | -- | -- | -- |
| Ensemble (mean) | 8x | -- | -- | -- | -- | -- | -- |

**Expected narrative:**
- Segmentation on ground truth images establishes the upper bound.
- At R=4x, segmentation Dice on reconstructed images should be close to ground truth performance (small gap).
- At R=8x, noticeable Dice degradation, especially for small structures (ascending aorta, pulmonary artery).
- At R=12x, substantial degradation -- reconstruction artifacts corrupt segmentation.
- Key finding: reconstruction PSNR/SSIM alone does not perfectly predict segmentation impact. Some artifacts are "benign" (background noise) while others are "malignant" (structural distortion).
- **Uncertainty gating experiment:** Set a threshold on mean reconstruction uncertainty. Reject (flag) images above threshold. Among accepted images, segmentation Dice is maintained. This demonstrates uncertainty as a practical quality gatekeeper.

**Figure 4 -- Downstream Segmentation Visualization:**
- 3 rows: R=4x, R=8x, R=12x.
- 4 columns: Reconstructed Image, Predicted Segmentation, Ground Truth Segmentation, Uncertainty Map.
- Show how segmentation errors visually correspond to high-uncertainty regions in the reconstruction.

**Evaluation criteria addressed:** Methods and Implementation Results (40%) -- This is the bulk of the results grade. Comprehensive quantitative evaluation with appropriate metrics. Trustworthiness component is woven throughout (UQ metrics, robustness tests, downstream evaluation). Bonus -- The downstream segmentation evaluation and uncertainty gating mechanism go beyond standard reconstruction benchmarks.

---

### 5. DISCUSSION (~0.75 pages, ~550 words)

**Target word count:** 550

#### 5.1 Key Findings Summary (~150 words)

**Content:**
- Restate the three main findings:
  1. MC Dropout and Deep Ensembles both provide meaningful uncertainty estimates for MRI reconstruction, with Ensembles showing better calibration at higher computational cost. The cost-quality tradeoff is quantified: MC Dropout requires ~20x inference time but no additional training; Ensembles require ~5x training cost but provide superior calibration.
  2. Uncertainty behaves predictably under distribution shift: it increases with acceleration factor, noise level, and cross-domain transfer. This makes uncertainty a reliable trust signal.
  3. Reconstruction quality matters for downstream tasks, but uncertainty provides additional predictive value beyond PSNR/SSIM for identifying reconstructions that will cause segmentation failures.

#### 5.2 Clinical Implications (~150 words)

**Content:**
- In clinical MRI workflows, reconstruction happens before any downstream analysis (segmentation, diagnosis, treatment planning). A silent failure at the reconstruction stage propagates errors throughout.
- Our uncertainty-guided gating mechanism provides a practical safety net: reconstructions with uncertainty above a threshold are flagged for radiologist review or re-acquisition.
- The computational overhead of MC Dropout (20 forward passes) is acceptable for offline clinical workflows. For real-time applications, distillation of uncertainty estimates into a single forward pass is a promising direction.
- Cross-domain detection capability means the system can alert clinicians when input data deviates from training distribution (e.g., unusual anatomy, different scanner protocol).

#### 5.3 Limitations (~150 words)

**Content:**
- **Single-coil simplification:** The MM-WHS data is magnitude-only, and we simulate single-coil k-space. Real clinical MRI uses multi-coil acquisition with complex-valued data. Our findings may not directly transfer.
- **Simulated undersampling:** We retrospectively undersample fully-sampled data. Prospective undersampling with real noise characteristics may behave differently.
- **U-Net baseline only:** We do not evaluate unrolled networks (VarNet, MoDL) which are state-of-the-art. Our framework is architecture-agnostic and could be applied to any reconstruction network.
- **Limited dataset:** MM-WHS is a relatively small cardiac dataset. Larger datasets (fastMRI) with more anatomical diversity would strengthen conclusions.
- **Aleatoric vs epistemic decomposition:** Both MC Dropout and Deep Ensembles primarily capture epistemic uncertainty. Heteroscedastic extensions to capture aleatoric uncertainty are left for future work.

#### 5.4 Future Work (~100 words)

**Content:**
- **Integration with TGVN-style trust guidance:** Atalik et al. (2026) showed that side information can reduce ambiguity in the reconstruction. Combining their ambiguous space consistency constraint with our uncertainty quantification framework could yield a system that is both more accurate and better calibrated.
- **Heteroscedastic uncertainty:** Train each ensemble member (or the MC Dropout model) to output both a mean and a variance, capturing aleatoric uncertainty from noisy k-space measurements.
- **Uncertainty-guided active acquisition:** Use predicted uncertainty maps to design the next k-space sampling pattern, acquiring lines where uncertainty is highest. This closes the loop between reconstruction and acquisition.

**Evaluation criteria addressed:** Discussion (20%) -- Depth of analysis is demonstrated through: (a) honest limitation acknowledgment, (b) clinical implication analysis, (c) concrete future directions grounded in the reviewed literature (TGVN connection). Bonus -- The TGVN integration idea and uncertainty-guided active acquisition are forward-looking innovative proposals.

---

### 6. REFERENCES (~1 page, ~20-25 references)

**Target: 22-25 references, formatted in Springer LNCS style.**

**Core references by category:**

#### MRI Reconstruction (8-10 references)
1. Safari, M., Eidex, Z., Chang, C.-W., Qiu, R.L., Yang, X.: Advancing MRI reconstruction: A systematic review of deep learning and compressed sensing integration. Preprint arXiv:2501.14158 (2025)
2. Atalik, A., Chopra, S., Sodickson, D.K.: A trust-guided approach to MR image reconstruction with side information. IEEE Trans. Med. Imaging 45(1), 190--205 (2026)
3. Lustig, M., Donoho, D., Pauly, J.M.: Sparse MRI: The application of compressed sensing for rapid MR imaging. Magnetic Resonance in Medicine 58(6), 1182--1195 (2007)
4. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: MICCAI 2015, LNCS, vol. 9351, pp. 234--241 (2015)
5. Schlemper, J., Caballero, J., Hajnal, J.V., Price, A.N., Rueckert, D.: A deep cascade of convolutional neural networks for dynamic MR image reconstruction. IEEE Trans. Med. Imaging 37(2), 491--503 (2018)
6. Hammernik, K., Klatzer, T., Kobler, E., et al.: Learning a variational network for reconstruction of accelerated MRI data. Magnetic Resonance in Medicine 79(6), 3055--3071 (2018)
7. Aggarwal, H.K., Mani, M.P., Jacob, M.: MoDL: Model-based deep learning architecture for inverse problems. IEEE Trans. Med. Imaging 38(2), 394--405 (2019)
8. Zbontar, J., Knoll, F., Sriram, A., et al.: fastMRI: An open dataset and benchmarks for accelerated MRI. arXiv:1811.08839 (2018)
9. Zhu, B., Liu, J.Z., Cauley, S.F., Rosen, B.R., Rosen, M.S.: Image reconstruction by domain-transform manifold learning. Nature 555(7697), 487--492 (2018)

#### Uncertainty Quantification (6-8 references)
10. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In: ICML 2016, pp. 1050--1059 (2016)
11. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: NeurIPS 2017, pp. 6402--6413 (2017)
12. Kendall, A., Gal, Y.: What uncertainties do we need in Bayesian deep learning for computer vision? In: NeurIPS 2017, pp. 5574--5584 (2017)
13. Ovadia, Y., Fertig, E., Ren, J., et al.: Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In: NeurIPS 2019, pp. 13991--14002 (2019)
14. Nair, T., Precup, D., Arnold, D.L., Arbel, T.: Exploring uncertainty measures in deep networks for multiple sclerosis lesion detection and segmentation. Medical Image Analysis 59, 101557 (2020)

#### Trustworthy AI / Medical Imaging (4-5 references)
15. Kaur, D., Uslu, S., Rittichier, K.J., Durresi, A.: Trustworthy artificial intelligence: A review. ACM Computing Surveys 55(2), 1--38 (2022)
16. Jungo, A., Balsiger, F., Reyes, M.: Analyzing the quality and challenges of uncertainty estimations for brain tumor segmentation. Frontiers in Neuroscience 14, 282 (2020)
17. Mehrtash, A., Wells, W.M., Tempany, C.M., Abolmaesumi, P., Kapur, T.: Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Trans. Med. Imaging 39(12), 3868--3878 (2020)
18. Stacke, K., Eilertsen, G., Unger, J., Lundstrom, C.: Measuring domain shift for deep learning in histopathology. IEEE J. Biomed. Health Inform. 25(2), 325--336 (2021)

#### Dataset (2 references)
19. Zhuang, X.: Multivariate mixture model for myocardial segmentation combining multi-source images. IEEE Trans. Pattern Anal. Mach. Intell. 41(12), 2933--2946 (2019)
20. Zhuang, X., Shen, J.: Multi-scale patch and multi-modality atlases for whole heart segmentation of MRI. Medical Image Analysis 31, 77--87 (2016)

#### Additional (2-3 references as needed)
21. Edupuganti, V., Mardani, M., Vasanawala, S., Pauly, J.: Uncertainty quantification in deep MRI reconstruction. IEEE Trans. Med. Imaging 40(1), 239--250 (2021)
22. Xiang, J., Dong, Y., Yang, Y.: FISTA-Net: Learning a fast iterative shrinkage thresholding network for inverse problems in imaging. IEEE Trans. Med. Imaging 40(5), 1329--1339 (2021)

---

## Page Budget Summary

| Section | Pages | Words | Figures | Tables |
|---------|-------|-------|---------|--------|
| Abstract | 0.25 | 180 | 0 | 0 |
| 1. Introduction | 0.75 | 550 | 0 | 0 |
| 2. Related Work | 1.00 | 750 | 0 | 0 |
| 3. Methods | 2.00 | 1500 | 1 (Fig 1) | 0 |
| 4. Experiments & Results | 2.50 | 1800 | 3 (Fig 2-4) | 3 (Tab 1-3) |
| 5. Discussion | 0.75 | 550 | 0 | 0 |
| 6. References | 0.75 | -- | 0 | 0 |
| **Total** | **8.00** | **~5330** | **4** | **3** |

---

## Evaluation Criteria Mapping

### Literature Review (20%)
- **Section 2** provides comprehensive coverage of three distinct research areas.
- **20+ references** from top venues (IEEE TMI, MRM, NeurIPS, MICCAI, ICML).
- Clear gap identification: UQ for MRI reconstruction is underexplored; no existing work connects reconstruction uncertainty to downstream task reliability.
- The two assigned papers (Safari 2025, Atalik 2026) are deeply integrated.

### Methods and Implementation Results (40%)
- **Section 3** provides rigorous mathematical formulation and clear method description.
- **Trustworthiness component (mandatory):** MC Dropout and Deep Ensemble UQ, DC layer for physics consistency, uncertainty calibration evaluation, robustness analysis.
- **Section 4** provides comprehensive quantitative evaluation with 3 tables and 3 figures.
- Multiple evaluation axes: reconstruction quality, uncertainty quality, robustness, downstream impact.

### Discussion (20%)
- **Section 5** provides honest analysis of findings, limitations, and clinical implications.
- Concrete future work grounded in reviewed literature (TGVN connection).
- Acknowledges limitations (single-coil, simulated undersampling, dataset size).

### Bonus -- Innovative Ideas (20%)
- **Multi-faceted trustworthiness evaluation framework** -- goes beyond single UQ method.
- **Downstream segmentation evaluation** -- connects reconstruction to clinical utility.
- **Uncertainty-guided quality gating** -- practical mechanism for clinical deployment.
- **Cross-domain robustness using MM-WHS multi-modal data** -- creative use of dataset.
- **Future work on TGVN + UQ integration** -- shows forward thinking.
- **Uncertainty-guided active acquisition** -- ambitious future direction.

---

## Implementation Priority Order

For the experiments agent, execute in this order:

1. **Data pipeline:** k-space simulation, undersampling masks, data loaders for MM-WHS MR data.
2. **U-Net + DC baseline:** Train and evaluate at R=4x. Validate pipeline with PSNR/SSIM/NMSE.
3. **MC Dropout:** Add dropout layers, implement MC inference loop, compute uncertainty maps.
4. **Deep Ensemble:** Train 5 models with different seeds, implement ensemble inference.
5. **Uncertainty evaluation:** Implement ECE, correlation, AUSE, sparsification plots.
6. **Multi-acceleration:** Run all methods at R={4, 8, 12}.
7. **Robustness tests:** Noise injection, cross-domain (CT) evaluation.
8. **Downstream segmentation:** Train segmentation model, evaluate on reconstructed images.
9. **Uncertainty gating:** Implement threshold-based rejection and measure accepted-set Dice.
10. **Figure generation:** All publication-quality figures saved to latex/figures/.
