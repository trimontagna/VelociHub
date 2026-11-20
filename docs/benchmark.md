# Benchmark — analytic flows

The benchmark runs VelociHub on flows whose displacement field is known exactly, step by step, and checks each step
against that field. Run it after a change to catch regressions, and to see how a new feature behaves.

- **Module:** `velocihub/benchmark/`:
  - `flows.py`: the flows;
  - `cases.py`: the cases, their checks and the levels;
  - `post_level.py`, `regression_level.py`: the runs and the metrics;
  - `image_level.py`: the levels on particle images (pre-processing → processing, and the full chain);
  - `uncertainty.py`: the uncertainty of the PTV predicted from the flow;
  - `images.py`: the particle images, the test workspace and the background-removal test;
  - `measurement.py`: folders written as VelociHub's own;
  - `report.py`: the report.
- **Tests:** `tests/test_benchmark.py`.

## Levels

The checks are integrated progressively, from the last step backwards, so that a failure points to the step that
caused it.

| Level | What runs | Against |
|---|---|---|
| **Pre-processing** | Particle images of the cylinder flow with stationary reflections of known shape (the cylinder's rim, a streak, a wall line, a ring, a glare spot), then VelociHub's POD background removal (mask loaded) with 0–3 modes removed | What remains of each reflection, by a matched filter of its exact shape (% and grey levels) and by a Hough transform for lines and circles; the particles as a whole and one by one; the background left and the pixels clipped at 0 |
| **Post-processing** | The exact fields, without and with noise, written as PIV results (`Exact/vectors.npz`, through `PIVRun.vectors`), read by `load_vectors` with the camera and post-processed by `PostProcessing` with its default recipe. No PIV/PTV step | 1. The samples themselves: mean, standard deviation and covariance, computed by numpy from the same displacements; this verifies the arithmetic, units and orientation to the rounding. 2. The analytic values: the exact mean, the noise statistics, the analytic vorticity and gradient, and the turbulence statistics of the modes |
| **Processing → post** | Particle links and PIV vectors sampled from the exact field (no images), binned and regressed by VelociHub, saved by `PIVRun.save` (`vectors.npz`), read back and post-processed | The exact field; the post-processed mean against the fields read back (the chain, to the rounding); the error of the binning against its uncertainty predicted from the flow |
| **Pre-processing → processing** | Particle images of each case, pre-processed with several recipes (none, the POD background removal, the default of the interface, then each contrast filter), processed by the PIV (PaIRS-UniNa) and the PTV (TracTrac) through the API, as a user runs them. The same flows rendered harder compare all the recipes | The exact field: the error of the PIV and its outliers, valid vectors and peak ratio; the particles the PTV links, its wrong links, the error of the right ones and the lengths of its tracks |
| **Full chain** | The test images with the background removal (1 POD mode), the PIV and the PTV with and without SPICY, then the post-processing with its default recipe | The exact statistics: the mean, the vorticity of the mean, the fluctuations; for the turbulence, its variances, integral scales and dissipation |

The background removal has a test of its own (the pre-processing level). Its measurement, the cylinder with
reflections, is also its single case through the whole chain, with the background removal on and off.

## Running it

| Where | How |
|---|---|
| Interface | **Tests > Benchmark**: everything, one level for all the cases, or one case (post and processing levels). It runs in its own process while the interface stays usable. A window follows it: a bar per level (the runs of the PIV/PTV counted on the image levels), the current step, the time since the start, the log of the messages, **Stop** (the run and the PIV/PTV it started ended) and **Open the report** at the end. **Close** hides it while the run goes on; **Tests > Show the progress** shows it again. The report also opens in the browser at the end. Other entries: **Open the last report**, **Open the reports folder**. |
| Command line | `python -m velocihub.benchmark [cases ...] [--levels preprocessing post processing images chain] [--scenarios ...] [--out folder] [--compare previous] [--spicy key=value ...] [--no-open] [--json]`. Exit code 1 when a check fails. |
| Tests | `pytest tests/test_benchmark.py` (one test per check; the report written too). The checks of the levels on images are skipped unless `--benchmark-images` (they run the PIV/PTV for about 20 minutes). `--benchmark-report folder` writes the report elsewhere. |

**The reports** go to `benchmark_reports` in the folder of the repository, one folder per run named by its date and
time (`report_20261120_183015`), so that the runs can be gone through and compared. Git leaves the folder out
(`.gitignore`). When VelociHub is installed without its repository, they go to `~/VelociHub/benchmarks`. `--out`
writes one elsewhere.

**How long:** about a minute without the levels on images; about 20 minutes with them (12 cores; the turbulence and
its SPICY fits on 200 maps take the longest). The images are rendered once and kept in
`~/VelociHub/benchmarks/images` (about 100 MB), then reused while their parameters stay the same.
| Python | `from velocihub.benchmark import run_benchmark, report`; `result = run_benchmark(["jet"], levels=("post",))`; `report.write_report(result, "folder")`. |

Each report can be compared with a previous one: in the interface, the last report; on the command line, `--compare`.
The previous values are shown next to the current ones. `--spicy n_k=8 eps_l=0.5` changes the SPICY settings of every
case, for studies of the settings; the tolerances stay as they are.

## Test workspace (images)

**Tests > Test workspace > Open the test workspace** writes the test measurements the first time (a few minutes,
about 70 MB) and opens them as the workspace of the interface: test mode. **Back to my workspace** returns to the
user's own workspace. The equivalent command is `python -m velocihub.benchmark --test-workspace [folder] [cases ...]`.
The default folder is `~/VelociHub/benchmarks/test_workspace`.

The pictures of the documentation are those of the turbulence case, processed as a measurement: `python
tools/doc_screenshots.py` renders its images (`~/VelociHub/examples/Turbulence` by default, `--example` another
folder), runs the pre-processing, the PIV (PaIRS-UniNa), the PTV (TracTrac) and the post-processing with the API, then
opens the interface on it (scratch preferences) and captures each page and window into `docs/images` (Windows; keep
the window uncovered meanwhile).

Each case is a measurement folder, like real data:
- `img_0000.tif ...`: a time-resolved sequence. The steady flows have 41 frames and the turbulence 201. The pairs are
  consecutive frames, so the flow's displacement is per frame.
- `masks.json`: the mask of the body, for the cylinder and the Blasius wall. Load it in the pre-processing.
- `Exact/vectors.npz`: the exact fields of the pairs, written as PIV results. In the Projects panel it is a project of
  its own, `case/Exact`, which opens in the post-processing at once.
- `benchmark_case.json`: the flow, its parameters and the camera.

The camera of every case is 512 × 320 px, 0.2 mm/px (a 102.4 mm wide field of view), 1000 frames/s, so 1 px/pair is
0.2 m/s. Enter these in the post-processing for the physical space.

The images are rendered as follows:
- **Particles:** 0.015 per pixel over the images and a 48 px margin, advected by the flow (4th-order Runge–Kutta, 4
  steps per frame), drawn as Gaussian spots about 2.5 px across with random peak intensities.
- **Lighting and noise:** the laser energy varies by ±5 % from frame to frame, over a dark background with camera
  noise.
- **Masks:** the body of the flow is dark.
- **Reflections:** only the `background_removal` measurement has the stationary reflections.

## Cases

All the flows are divergence-free, in px per pair, with a 16 px grid.

| Case | Flow | What it tests |
|---|---|---|
| `uniform` | Uniform, oblique (4, 1) px | Bias, edge effects, spurious vorticity: anything that isn't constant is an artefact. |
| `cylinder` | Potential flow around a cylinder (R = 40 px), masked | A masked body, stagnation points, 2 U at the shoulders, strong gradients near the surface, zero vorticity. |
| `jet` | Planar jet: a uniform core between two tanh shear layers that thicken downstream (δ from 8 to 18 px) | Steep gradients barely resolved by the grid (close to the jet measurements); half-width, peak shear. |
| `modes` | A uniform stream plus Taylor–Green cells at λ = 320, 160, 80, 48 px | Spatial resolution: the amplitude each method keeps at each wavelength (its transfer function). |
| `blasius` | Thick laminar boundary layer (Blasius; δ99 from 60 to 120 px), wall masked | Wall vorticity, the small wall-normal displacement, δ*, θ and the gradient at the wall. |
| `turbulence` | Synthetic turbulence: 400 Fourier modes of a stream function (random directions and phases, spectrum E(k) ∝ k⁴ exp(−2(k/k_p)²), peak at 128 px, RMS 0.6 px), frozen and convected at U0 = 4 px/pair | The statistics of the post-processing. All are known from the modes: variances, Reynolds stress, autocorrelations, integral lengths to the first zero crossing (L_x^u = 45 px, L_y^u = 22 px), integral time (L_x^u / U0 = 11.4 frames), and dissipation, both exact and with the differences of the grid (each derivative k becomes sin(kh)/h). |

**Post-processing level**:
- **Exact fields:** 200 maps for the steady flows; for the turbulence, 2000 time-resolved maps.
- **With noise:** the same, with 0.1 px of white noise on each component. For the steady flows, the noise statistics
  are known:
  - RMS = 0.1 px;
  - no Reynolds stress;
  - skewness 0 and kurtosis 3;
  - the vorticity RMS of central differences of white noise, noise / grid spacing.

**Processing level**:
- **Data:** 3 pairs per scenario, 0.01 particles per pixel (about 1500 links per pair). For the turbulence, the pairs
  are independent snapshots.
- **Scenarios:** 0.1 px of noise on each component of every link and vector; and the same with 2 % of the links wrong,
  each given a random displacement within ±8 px (PTV only).
- **Methods compared:**
  - *Exact*: the exact field on the grid;
  - *SPICY (PTV)*: SPICY fitted on the particle links;
  - *Binned (PTV)*: the PTV without regression;
  - *SPICY (PIV)*: SPICY fitted on the vectors.

## The mean gradient: the grid and the smoothing

The post-processing computes the mean gradients (`gradient_fields`: dU/dY, the gradients of the turbulence intensity,
of the shear stress and of the vorticity) by smoothing the mean field over σ cells (`gradient_sigma`: a Gaussian
ignoring the holes), then central differences; or, with `gradient_smooth_after`, the differences first (one-sided next
to the holes and the edges), then the smoothing. The benchmark checks that it recomputes both to the rounding, then
splits the error of dU/dY (exact fields scenario):

- **The grid alone:** the exact field on the grid, central differences, no smoothing. Each wavelength λ keeps
  sin(kh)/kh of its gradient (k = 2π/λ, h the grid spacing).
- **The smoothing:** it multiplies that by the transfer of the Gaussian in X and in Y, about exp(−(kσh)²/2) each.
  At σ = 1.5 cells a structure of 10 cells keeps 38 % of its gradient, at 0.5 cell 90 %.
- **The edges and masks:** within about 3σ of them, smoothing the field averages only one side, and the central
  differences give no value at the points next to them. Smoothing after the differences avoids both.

| Case (RMS error, % of the largest dU/dY) | Grid alone | σ = 1.5 (before 1.2) | σ = 0.5 (default) | σ = 0.5, smoothed after |
|---|---|---|---|---|
| Cylinder | 0.7 | 4.3 | 0.7 | 0.6 |
| Blasius | 0.8 | 10.3 | 2.0 | 1.1 |
| Jet (δ = 0.5–1.1 h) | 9.7 | 16.6 | 11.0 | 11.0 |
| Modes (λ = 3–20 h) | 16.2 | 31.5 | 22.8 | 22.8 |

- On the cylinder and the Blasius layer, the error of σ = 1.5 came from the smoothing next to the body and the wall.
  With the differences first, the Blasius layer also gets the gradient at its first point above the wall (1.03 × the
  exact one), which has none otherwise.
- On the jet, the grid alone keeps 0.67 of the peak shear U/2δ; σ = 0.5 keeps 0.62 (σ = 1.5 kept 0.37).
- The measured amplitude per wavelength matches the theory to 10⁻⁴; this is a check of the modes case.
- The report plots the error against σ, in both orders, with a white noise added to the mean field. With 0.1–0.2 px
  of noise (that of a single map), the best smoothing is σ = 0.5–1 cell on the cylinder and the Blasius layer, and
  none on the jet and the modes.

## The uncertainty predicted from the flow

For the binned PTV, the uncertainty of each grid point follows from the flow and the links of its bin
(`uncertainty.py`): with N links in a bin of size h, measurement noise σ₀,

- **random error:** √((σ₀² + h²/12 |∇u|²) / N). This is the noise of the links, plus the variation of the displacement
  over the bin, sampled at N random positions;
- **bias:** h²/24 ∇²u, the curvature averaged by the bin, as a top-hat filter.

It is checked the way the uncertainty of a measurement is (Sciacchitano et al. 2015):
- the RMS uncertainty against the RMS error;
- (error − bias) / random part, which should have a standard deviation of 1;
- the errors within one and two uncertainties: 68 % and 95 % for a Gaussian error.

On every case it holds: ratio 0.97–1.09, 67–69 % within 1σ, 95–97 % within 2σ. For real data, the same formula
applies with the gradients and curvature of the measured field, and σ₀ from the dispersion of the links in each bin.

Not modelled yet:
- **SPICY:** its filter could come from the transfer function measured on the modes, and its random error from its
  residual.
- **The image level:**
  - the correlation of the PIV windows: random error growing with the gradient across the window;
  - the truncation of the time step along curved paths: a displacement error of about a Δt²/24 times the Lagrangian
    derivative, and a position error of about a Δt²/8;
  - the peak locking.

## The levels on images

**The images.** Each case's test images (`images.render`: 41 frames for the steady flows, 201 for the turbulence) are
linked into a measurement folder of the work folder, with the mask of the body, and run through `vh.Project` as a user
would: `set_preprocessing` with the recipe, `set_piv` with the engine, then the PIV/PTV (the pre-processing first, as
Run step > Singular does). Each run keeps its own folders (`PreProcessing_<recipe>`, `PIV_<recipe>`, `PTV_<recipe>`,
`..._spicy`). The PIV uses windows of 64 then 32 px with 50 % overlap (the 16 px grid of the cases); the PTV, TracTrac
with its defaults. The exact displacement of a pair is the flow at the middle of the pair.

**The recipes** (`image_level.RECIPES`; the filters applied after the background removal, with their defaults):

| Recipe | What it does |
|---|---|
| None | No pre-processing (the mask of the body only) |
| POD, 1 mode | The background removal: the first POD mode, about the mean image, removed |
| Default (POD 6 modes + capping) | The default of the interface |
| POD 1 + capping, + min-max, + CLAHE, + edges, + Gaussian matching | 1 mode removed, then one contrast filter |

The test images run the first three, the hard images all eight.

**The hard images** (`images.HARD`) render the cylinder, the jet and the turbulence (61 frames) as a difficult
recording: dim particles of varied brightness (peak 20 to 90 grey levels) and size (1.5 to 3.5 px across), camera
noise of 6 grey levels, a background of 30, a light sheet dimmer by 60 % at the top and bottom edges, ±10 % laser
energy, and the stationary reflections of the background removal test. The test images stay as they were.

**The measures.**
- PIV: the RMS error of the displacement (all the flow and away from the edges and masks), the outliers (more than
  0.5 px off), the valid vectors and the median correlation peak ratio.
- PTV: the particles linked in each pair (median), the wrong links (more than 1 px off), the RMS error of the right
  links, the median length of the tracks, the particles tracked of those detected, and the field binned on the grid.
- Chain: the post-processing's own measures (`post_level.post_metrics`) with no noise added, so the errors of the
  measurement count as fluctuations; and the RMS error of the instantaneous fields.

**What they show** (first run, VelociHub 1.2):

| Hard images | PIV error, none → best (px) | PTV wrong links, none → best (%) | Best for the PIV | Best for the PTV |
|---|---|---|---|---|
| Cylinder (reflections) | 0.74 → 0.15 | 6.8 → 3.7 | POD 1 mode (+ edges: about the same) | POD 1 + edges |
| Turbulence (reflections) | 0.60 → 0.19 | 7.2 → 4.6 | POD 1 + edges | POD 1 + edges |
| Jet (still fluid around it) | 0.30 (none) | 5.0 (none) | None | None |

- **The background removal (1 mode)** is what matters on images with reflections: it divides the error of the PIV by 3
  to 5 and the wrong links of the PTV by about 1.5. On clean images it costs nothing.
- **The contrast filters add little after it.** The edge enhancement is slightly the best, and capping changes
  nothing. Min-max and Gaussian matching make the PTV link more particles, but more wrongly: up to almost 4 times the wrong
  links and much shorter tracks. CLAHE is the worst: it doubles the error of the PIV and gives 4 to 5 times the wrong
  links.
- **The default recipe (6 modes + capping)** wipes the particles of single images in these 41-image sequences. The PIV
  error goes from 0.015 to 1.95 px on the uniform flow, and the PTV keeps a third of its links, with tracks of 6
  frames instead of 26. On longer recordings, the modes beyond the first should hold less of single images. Check it
  on yours: the PIV quality of the pairs shows the ones that fail.
- **Still fluid loses its particles to the background removal.** Around the jet, the fluid doesn't move, so its
  particles are part of the mean image and get removed. The PIV error goes from 0.15 to 1.4 px.

**The full chain** keeps the grid of the case, the units and the statistics of the samples to the rounding. The
error of the mean of the PIV is 0.003–0.09 px on the steady flows, except the jet (the still fluid) and the modes
(λ = 48 px isn't resolved by 32 px windows); 0.19 px for the turbulence, whose 200 maps (18 integral times) converge
no closer. For the turbulence, the PIV windows lower the fluctuations: RMS of u at 0.83 of the modes'
and dissipation at 0.52, against 0.93 and 0.70 for the PTV. With the background removal on, the cylinder with
reflections has 0.63 times the error of the mean of the PIV and 0.57 times that of the PTV.

## On a real recording

The same recipes can be compared on your own images, which have no exact field:

```bash
python -m velocihub.benchmark --recording D:/Jet/Run_30 [--frames 100] [--recipes none pod1 pod1_edge] [--work folder]
```

- **The runs:** each recipe, then the PIV and the PTV, one run at a time. The images are linked into a work folder
  (temporary, or `--work`, kept). The recording's folder is only read: the masks and the PIV windows come from its
  `VelociHub_project.json` (or its `masks.json`), and the pre-processed images are exported as PNG. Run again with
  the same `--work`, the runs that are up to date are skipped and only the measures are computed again.
- **The measures need no exact field:**
  - PIV: the valid vectors, the median peak ratio, and the spatial noise: the median distance of a vector from the
    median of its 8 neighbours. This is the random error, plus the curvature of the flow, which is the same for
    every recipe.
  - PTV: the particles linked per pair, and the suspect links: those failing the normalized median test against
    their 10 nearest links, an estimate of the wrong links. Also the track lengths, the particles detected and
    tracked, and the cells of the binned field with a value.
  - The median displacement is shown for reference. A binned PTV well below the PIV means fast particles that
    aren't tracked: a matter of the tracking settings, not of the pre-processing.
- **The report:** `benchmark_reports/recording_<name>_<date>_<time>`. It has the best recipe for each measure, a
  chart, the tables and the time of each run.
- **How long:** about 30 s per run for 41 frames of 1920 × 1200 px; 500 frames with the eight recipes, about 2
  hours (7 minutes per run).

**A jet recording** (`Test_Run_30_500f`: 500 frames of 1920 × 1200 px, 8-bit and very dark, 99 % of the pixels at 4
grey levels or less; its mask of the nozzle; PIV windows 128 64 32 px, 50 % overlap):

| Recipe | PIV valid (%) | PIV peak ratio | PIV spatial noise (px) | PTV links per pair | PTV suspect links (%) | PTV mean track (frames) |
|---|---|---|---|---|---|---|
| None | 96.8 | 2.99 | 0.238 | 9 462 | 3.28 | 10.7 |
| POD, 1 mode | 96.8 | 2.99 | 0.236 | 9 526 | 3.34 | 10.5 |
| Default (POD 6 modes + capping) | 96.7 | 3.03 | 0.230 | 9 701 | 3.55 | 9.9 |
| POD 1 + capping | 96.7 | 3.03 | 0.230 | 9 675 | 3.48 | 10.3 |
| POD 1 + min-max | 97.0 | 3.03 | 0.231 | 10 970 | 3.39 | 12.7 |
| POD 1 + CLAHE | 97.3 | 3.19 | 0.219 | 10 993 | 3.72 | 11.2 |
| POD 1 + edges | 95.9 | 2.86 | 0.233 | 10 994 | 3.37 | 10.8 |
| POD 1 + Gaussian matching | 97.1 | 3.01 | 0.229 | 11 482 | 3.43 | 13.4 |

- **The PIV barely changes:** valid vectors within 1.5 %, peak ratio within 11 %. The background removal changes
  nothing: this recording has little stationary background. CLAHE scores best on all three measures, yet on the
  hard synthetic images it doubles the error of the PIV. A higher peak ratio or fewer replaced vectors doesn't
  mean a more accurate field, so these measures rank the recipes, not their accuracy.
- **The default 6 modes cost nothing here**, unlike on the 41-image sequences of the benchmark: with 500 images,
  the modes beyond the first hold little of single images.
- **The PTV links 16–21 % more particles** with min-max, edges or Gaussian matching. Gaussian matching and min-max
  also give the longest tracks (mean 13.4 and 12.7 frames, against 10.7), at a slightly higher share of suspect
  links (3.4 % against 3.3 %). On the hard synthetic images, the same filters raised the actual wrong links 2 to
  4 times. The median test misses wrong links that agree with their neighbours, so check these on a known flow.
- **The binned PTV** covers 58 % of the cells, with a median displacement of 4.0 px against 6.4 px for the PIV:
  TracTrac with its defaults misses the fast particles of the jet's core (up to 20 px per frame). Its settings
  (search radius, motion model) matter more here than the pre-processing.

## Checks and known issues

A **check** compares a metric with a tolerance, or with another method's metric. The verification checks, against the
samples and through the chain, must hold to the rounding of the files (10⁻⁴ px). The others use VelociHub 1.2's own
values with a margin of about 30 %. The runs are deterministic: the seeds are fixed, and SPICY's clustering is too. So
a failed check is either a regression, or an improvement that should tighten its tolerance in `cases.py`.

A **known issue** is a check expected to fail, with its reason. It is reported apart from the failures and is an
`xfail` in the tests. When it starts passing it is reported, and its test fails, so that it becomes a normal check.

The known issues of VelociHub 1.2 (SPICY defaults), all at the processing level:

| Issue | Seen in |
|---|---|
| With 4 points per RBF, SPICY on particles is noisier than the binning (little averaging per RBF) | uniform, cylinder |
| Worse at the edges of the field of view and of the masks (RBFs constrained on one side) | uniform, cylinder, blasius, turbulence |
| SPICY on PIV vectors smooths out structures shorter than about 5 grid spacings: worse than the vectors it fits | jet, modes |
| The wavelength of 3 grid spacings isn't resolved by SPICY on particles | modes |

On particle images (levels images and chain):

| Issue | Seen in |
|---|---|
| The default recipe removes 6 POD modes: with 41 images, the modes beyond the first hold the particles of single images, whose pairs fail | uniform, cylinder, jet, blasius, background removal (PIV); every case (PTV wrong links) |
| The POD background removal takes the particles of still fluid for the background: the PIV/PTV find nothing there | jet |

SPICY on particles now runs an outlier test of the links before its fit: the normalized median test of scattered data
(`outlier_test` 3, `outlier_neighbours` 10; see `docs/processing.md`). With 2 % of the links wrong it catches 93–100 %
of them; the error of SPICY is 1.01–1.11 times that without wrong links, against 4–5 times without the test. It
leaves out at most 0.6 % of the good links, on the shortest wavelength of the modes, whose curvature over the
neighbourhood the median doesn't follow.

What the post-processing level shows (all checks pass):
- **Statistics:** VelociHub reproduces the sample statistics to the rounding. It recovers the noise RMS within 1 % and
  the turbulence variances within 1 %.
- **Integral scales:** within 4–10 % of the modes' values, slightly short: the 16 px lags and the finite field of view.
- **Dissipation:** within 2.5 % of the best the grid can resolve, which is 78 % of the exact value.
- **Mean gradients:** dU/dY is smoothed: RMS error of 4 % (cylinder), 10 % (Blasius), 17 % (jet) and 31 % (modes) of
  its largest value. See the mean gradient section above for how the grid, the smoothing and the edges split it.

## The report

The report is a folder:
- `index.html`: the summary, with:
  - the counts;
  - each case's checks by level and its key numbers;
  - a chart of the processing errors;
  - the checks not as expected;
  - the known issues;
  - the environment (versions, commit).
- A page per case, with a section per level:
  - the checks (value, tolerance, previous run);
  - the metrics;
  - the maps: the post-processed statistics against the expected ones and their difference; the fields of each
    method and their differences from the exact field;
  - the case's profiles, transfer function or autocorrelations.
- `background.html`: how what is left is measured; for each reflection, its % and grey levels left, and whether the
  Hough transform finds it; the mean images before and after; the particles one by one (the distribution of their
  brightness kept, where the dimmed ones are); the background left.
- `results.json`: every number, which `--compare` reads.

## Adding a case

1. Add a flow to `flows.py`: a `Flow` subclass with `velocity(X, Y, t)` (X right, Y up, px per pair, t in frames),
   and `solid(X, Y)` if it has a body. Set `steady = False` if it changes with time. Register it in `FLOWS`.
2. Add a `Case` to `cases.py`:
   - `common_checks(...)` (processing) with its tolerances;
   - `post_checks(...)` (post-processing);
   - its own checks.

   A new metric goes in `regression_level.specific_metrics` or `post_level.post_metrics`, with its label in
   `report.METRICS`.
3. Optionally, add a figure in `report.SPECIAL`, and a mask in `images.mask_polygons`.

## What the background removal shows

The first mode is about the mean image, which also holds the light of the particles averaged over the images (9.4
grey levels here, over a background of 16.3). A pixel without a particle at that instant ends about 9 grey levels
below 0. Before VelociHub 1.2 the result was clipped at 0 when cast back to 8 bits: 83 % of the pixels at 0, the tails
of the particles cut. The background removal now adds an offset (automatic: 22 grey levels here).

| 1 mode removed | Clipped at 0 (before 1.2) | With the automatic offset |
|---|---|---|
| Particles: brightness kept (median of each one's) | 89 % | 98 % |
| Pixels at 0 | 83 % | 0.9 % |
| Lines and circles left (largest) | 3.7 % | 0.2 % |
| Glare spot left | 6.8 % | 0.2 % |

The clipping also biased the removal of the reflections: around them, the values below 0 were cut. The Hough
transform finds none of the lines and circles in either case.

Each further mode dims the particles of a few images: with 41 images, the modes beyond the first carry the particles
of single images (their temporal coefficients peak there). With 2 modes removed, 19 % of the particles of the worst
image fall below half their brightness (56 % without the offset): that pair would suffer at the PIV/PTV. The check
`dimmed_worst_image` watches it with 1 mode.

## Next

- **The default number of POD modes:** 6 removes the particles of single images of short recordings. An automatic
  number (the threshold of the pre-processing page), or 1 by default, could be tested on the image level.
- **The settings of the engines:** the PIV windows and the TracTrac settings, compared on the hard images as the
  recipes are.
