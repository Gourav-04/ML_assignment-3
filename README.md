# CIFAR-10 Assignment 3

**Author:** Gourav  
**Roll number:** m26cse018  
**Split seed:** 26018

This project studies how spatial structure affects image classification, when an
MLP starts fitting incorrect labels, and how much class information an autoencoder
retains. Each question has its own Python script. The final report is in
[`report/main.pdf`](report/main.pdf).

## Results from my run

| Part | Experiment and result |
|---|---|
| Q1(a) | The dense MLP scored 41.57% on clean images. The patch MLP scored 52.10% with contiguous patches and 48.40% after pixel permutation. The 3.70 percentage-point gap measures the benefit of intact patches within the same architecture. |
| Q1(b) | Clean validation accuracy first exceeded 20% at epoch 1. Accuracy against the deliberately incorrect labels first exceeded 20% at epoch 33. Under this threshold, the model learned patterns that transferred to validation images before it substantially fitted the incorrect labels. |
| Q2(a) | On clean images, the CNN scored 52.66% and the MLP scored 41.57%. The CNN led by 11.09 percentage points under the project's comparison protocol. |
| Q2(b) | With a fixed pixel permutation, the CNN scored 38.82% and the MLP scored 43.21%. The CNN lost 13.84 points relative to its clean run. |
| Q2(c) | With shuffled tiles, the CNN scored 45.28% and the MLP scored 40.59%. The CNN recovered some accuracy relative to full pixel shuffling, consistent with useful local patterns surviving inside each tile. |
| Q3(a) | Among bottleneck widths 8, 32 and 128, width 128 gave the lowest validation reconstruction MSE, approximately 0.01132. It was selected using validation data. |
| Q3(b) | The same frozen CNN scored 26.78% on corrupted images, 26.28% after the clean autoencoder, and 27.05% after the denoising autoencoder. Denoising improved accuracy by 0.27 percentage points over the corrupted input in this run. |
| Q3(c) | Frozen-probe test accuracy was 29.22%, 34.53% and 39.78% for widths 8, 32 and 128 respectively. Across these three widths, reconstruction error fell and probe accuracy rose. |
| Q4(a) | Validation selected the RBF SVM with C = 1. It scored 42.79% on the test set. |
| Q4(b) | The report examines difficult class pairs and boundary examples. For cat and deer, 93 cat images were classified as deer and 60 deer images as cat. |
| Q4(c) | The selected clustering used k = 8. Test ARI was 0.0488 and NMI was 0.0817, indicating weak agreement between cluster assignments and CIFAR-10 classes. |

## Where to find each implementation

| File or folder | Purpose |
|---|---|
| `q1_mlp.py` | Patch structure and memorisation experiments |
| `q2_cnn_vs_mlp.py` | CNN and MLP comparisons on clean, pixel-permuted and tile-shuffled images |
| `q3_autoencoders.py` | Reconstruction, denoising and frozen-encoder probes |
| `q4_svm_clustering.py` | SVM selection, class-boundary analysis and clustering |
| `common/` | Shared code used across the four questions |
| `config.json` and `protocol.md` | Experiment settings, comparison rules and criteria used to interpret results |
| `run_all.py` | Runs the assignment in question order |
| `results/` | Saved measurements, training logs, split indices and model checkpoints |
| `figures.py` | Generates experiment plots |
| `build_report.py` and `report/make_visuals.py` | Prepare report measurements, tables and visuals from the saved run |
| `report/main.tex` and `report/main.pdf` | LaTeX source and compiled report |

## Run on Windows using VS Code and PowerShell

Open the extracted `CIFAR10_Assignment3` folder in VS Code. Open a PowerShell
terminal in the folder containing `run_all.py`.

### 1. Set up Python once

Run each line in order. The project was run with Python 3.12 and CPU PyTorch.

~~~powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r .\requirements.txt
~~~

When opening a new terminal later, repeat only the `$py = ...` line to select the
existing environment.

### 2. Run all four questions

~~~powershell
& $py .\run_all.py
~~~

Keep the downloaded archive in `data/cifar-10-python.tar.gz` so later runs can
reuse it. The project can read this archive directly. The results are written
under `results/`.

### 3. Inspect the results

Open `report/main.pdf` for the question-wise interpretation. To check the exact
classifier scores behind Q1(a) and Q2, display the saved CSV in PowerShell:

~~~powershell
Import-Csv .\results\classifiers_test.csv | Format-Table
~~~

For example, the clean CNN accuracy is stored as `0.5266`, which appears as
52.66% in the report. For Q1(b), inspect `results/memory.csv` to follow clean
validation accuracy and incorrect-label accuracy across epochs.

### 4. Rebuild the PDF after a completed run

Keep the saved results, checkpoints and CIFAR-10 archive available. Install a
LaTeX distribution such as MiKTeX and confirm that PowerShell can find
`pdflatex`. Then run:

~~~powershell
pdflatex --version
& $py .\build_report.py
~~~

After a successful build, open the generated report:

~~~powershell
Invoke-Item .\report\main.pdf
~~~

Rebuilding the report uses the saved experiment outputs. You can run this step
on its own when revising the report.