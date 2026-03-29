# Hafftka Latent Space Explorer
## Tech Spec & Build Instructions

---

## What You're Building

A four-stage pipeline:

1. **Train** — StyleGAN2-ADA learns the visual language of Hafftka's catalog, starting from a pretrained painting model rather than random noise
2. **Improve** — Apply research-backed techniques to get better outputs from a small dataset: transfer learning, dataset augmentation, and an upgraded discriminator
3. **Analyze** — GANSpace runs PCA on the trained model to find its meaningful axes (color, composition, figural density, etc.)
4. **Explore** — A browser lets you navigate a 2D grid where every point generates a unique Hafftka-like painting, and you can walk between projected real works

The output is a latent space browser you can screenshot, record, and post. The conceptual frame: the entire visual universe of a museum-collected painter, collapsed to a plane you can walk through.

---

## Prerequisites

### Hardware
- Windows 10/11, NVIDIA RTX 4080 Super (16GB VRAM)
- At least 50GB free disk space (dataset + training checkpoints)

### Software to install before starting

```
Miniconda3 (if not already installed)
  https://docs.conda.io/en/latest/miniconda.html

Git for Windows
  https://git-scm.com/download/win

Visual Studio Community 2022
  https://visualstudio.microsoft.com/vs/community/
  During install, select: "Desktop development with C++"
  This is required for CUDA kernel compilation on Windows.
```

---

## Stage 1: Environment Setup

### 1.1 Create the conda environment

Open **Anaconda Prompt** (not PowerShell, not cmd — Anaconda Prompt specifically).

```bash
conda create -n hafftka python=3.9 -y
conda activate hafftka
```

### 1.2 Install PyTorch pinned to the compatible version

StyleGAN2-ADA requires PyTorch ≤ 1.13. The 4080 Super supports CUDA 11.8, which is the sweet spot.

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Verify GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should print `True` and `NVIDIA GeForce RTX 4080 SUPER`.

### 1.3 Clone StyleGAN2-ADA PyTorch

```bash
cd C:\Users\<your-username>
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd stylegan2-ada-pytorch
```

### 1.4 Install Python dependencies

```bash
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 psutil scipy
```

### 1.5 Add Visual Studio to PATH

In Anaconda Prompt, run this (adjust version number if needed):

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

Do this every time you open a new Anaconda Prompt session before training.

---

## Stage 2: Dataset Preparation

### 2.1 Download the Hafftka dataset

```bash
pip install huggingface_hub
```

```python
# Run this as a script: download_hafftka.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Hafftka/michael-hafftka-catalog-raisonne",
    repo_type="dataset",
    local_dir="C:/hafftka-dataset"
)
```

```bash
python download_hafftka.py
```

The images will land in `C:\hafftka-dataset\`. Expect 3,000–4,000 images of varying sizes and aspect ratios.

### 2.2 Inspect and clean the dataset

Before converting, do a quick audit:

```python
# audit_dataset.py
import os
from PIL import Image

dataset_dir = "C:/hafftka-dataset/images"  # adjust if needed
sizes = []
broken = []

for fname in os.listdir(dataset_dir):
    fpath = os.path.join(dataset_dir, fname)
    try:
        img = Image.open(fpath)
        sizes.append(img.size)
    except Exception as e:
        broken.append(fname)
        print(f"Broken: {fname} — {e}")

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
print(f"Total images: {len(sizes)}")
print(f"Broken: {len(broken)}")
print(f"Width range: {min(widths)}–{max(widths)}")
print(f"Height range: {min(heights)}–{max(heights)}")
```

### 2.3 Choose your training resolution

This is the most consequential decision you make before training.

| Resolution | VRAM usage | Training time (4080S) | Recommended for |
|---|---|---|---|
| 256×256 | ~4GB | ~12–18 hrs | Fast iteration, good for testing |
| 512×512 | ~8GB | ~36–48 hrs | Good quality, practical |
| 1024×1024 | ~14GB | ~5–7 days | Maximum detail, long commitment |

**Recommendation: start at 512×512.** You can always train 1024 later using the 512 checkpoint as a starting point (transfer learning).

### 2.3.1 The Aspect Ratio Problem

`dataset_tool.py` center-crops by default. For a painter who works at wildly different scales — tall verticals, wide horizontals, sprawling compositions — a center-crop can reduce a full-figure painting to an abstract texture patch. The figure gets decapitated.

Before running the conversion, skim your images folder. Look for extreme aspect ratios (anything more than roughly 1:2 or 2:1). For those paintings, center-crop will lose the most important content.

**The fix: mirror-pad to square before converting.**

Do not use black or white padding bars. The model will learn to generate paintings with black borders. Mirror-padding extends the edges of the painting outward, which the model reads as more painting — not as an artifact.

```python
# pad_to_square.py
# Run this on your images folder before dataset_tool.py
# Replaces extreme aspect-ratio images with mirror-padded squares

import os
from PIL import Image, ImageOps

src_dir = "C:/hafftka-dataset/images"
out_dir = "C:/hafftka-dataset/images-padded"
os.makedirs(out_dir, exist_ok=True)

RATIO_THRESHOLD = 1.4  # pad anything more extreme than 1:1.4

for fname in os.listdir(src_dir):
    fpath = os.path.join(src_dir, fname)
    try:
        img = Image.open(fpath).convert("RGB")
        w, h = img.size
        ratio = max(w, h) / min(w, h)

        if ratio > RATIO_THRESHOLD:
            # Pad the short dimension to match the long dimension
            target = max(w, h)
            pad_w = (target - w) // 2
            pad_h = (target - h) // 2
            # Mirror-pad: reflect edges outward
            img = ImageOps.expand(img, border=(pad_w, pad_h, target - w - pad_w, target - h - pad_h))
            # ImageOps.expand uses fill color — use mirror via numpy instead
            import numpy as np
            arr = np.array(Image.open(fpath).convert("RGB"))
            arr = np.pad(arr, ((pad_h, target - h - pad_h), (pad_w, target - w - pad_w), (0, 0)), mode='reflect')
            img = Image.fromarray(arr)

        img.save(os.path.join(out_dir, fname))
    except Exception as e:
        print(f"Skipped {fname}: {e}")

print("Done.")
```

Use `images-padded` as input to the augmentation script in Stage 3.5, or directly to `dataset_tool.py` if skipping augmentation.

### 2.4 Convert dataset to StyleGAN2-ADA ZIP format

```bash
python dataset_tool.py ^
  --source=C:/hafftka-dataset/images ^
  --dest=C:/hafftka-dataset/hafftka-512.zip ^
  --resolution=512x512
```

The `^` is Windows line continuation in cmd. If using PowerShell use `` ` `` instead.

The tool will center-crop and resize all images to 512×512, output a ZIP with a `dataset.json` manifest.

After conversion, verify:

```bash
python -c "import zipfile; z = zipfile.ZipFile('C:/hafftka-dataset/hafftka-512.zip'); print(len(z.namelist()), 'files')"
```

Should be 3000+ files plus one `dataset.json`.

---

## Stage 3: Training

### 3.1 Understand the key flags

| Flag | What it does | Your value |
|---|---|---|
| `--cfg` | Architecture config | `stylegan2` |
| `--gpus` | Number of GPUs | `1` |
| `--batch` | Images per step | `32` (drop to 16 with capacity increase) |
| `--gamma` | R1 regularization weight | `10` (default for 512) |
| `--aug` | Augmentation mode | `ada` (adaptive — essential for small datasets) |
| `--augpipe` | Which augmentations to use | `bgcfnc` (all types — blit, geom, color, filter, noise, cutout) |
| `--target` | ADA augmentation target probability | `0.6` (raise to `0.7` if mode collapse) |
| `--mirror` | Horizontal flipping | `1` (free data augmentation) |
| `--snap` | Save checkpoint every N kimg | `10` |
| `--kimg` | Total training images (thousands) | `5000` minimum, `10000` ideal |
| `--resume` | Start from pretrained checkpoint | `metfaces-1024x1024.pkl` for first run |
| `--cbase` | Generator feature map base | `32768` default; `65536` for capacity increase |
| `--cmax` | Generator feature map max | `512` (keep this) |

### 3.2 Download the MetFaces transfer learning checkpoint

Don't start from random noise. MetFaces is a StyleGAN2-ADA model trained on portrait paintings from museum collections — it already knows oil paint, canvas texture, compositional structure. Starting from it means you're teaching the model what *Hafftka* is, not what paint is.

```bash
# Download the MetFaces pretrained checkpoint
curl -L -o C:/hafftka-dataset/metfaces-1024x1024.pkl ^
  "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2-ada-pytorch/versions/1/files/metfaces1024x1024.pkl"
```

Or download manually from: `https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan2_ada_pytorch`

MetFaces was trained at 1024×1024. You can use it to bootstrap 512×512 training — the model will adapt.

### 3.3 First training run (with transfer learning)

```bash
cd C:\Users\<your-username>\stylegan2-ada-pytorch

python train.py ^
  --outdir=C:/hafftka-runs ^
  --data=C:/hafftka-dataset/hafftka-512.zip ^
  --cfg=stylegan2 ^
  --gpus=1 ^
  --batch=32 ^
  --gamma=10 ^
  --aug=ada ^
  --augpipe=bgcfnc ^
  --target=0.6 ^
  --mirror=1 ^
  --snap=10 ^
  --kimg=5000 ^
  --resume=C:/hafftka-dataset/metfaces-1024x1024.pkl
```

The `--augpipe=bgcfnc` flag enables all augmentation types: blit, geometry, color, filter, noise, cutout. With a small dataset of varied paintings spanning 50 years of Hafftka's work, maxing out augmentation helps the discriminator generalize rather than memorize individual paintings. It is one flag and it is free.

Training will create a directory like `C:\hafftka-runs\00000-stylegan2-hafftka-512-gpus1-batch32-gamma10\`.

Inside it:
- `network-snapshot-<KIMG>.pkl` — model checkpoint every 10k images
- `fakes<KIMG>.png` — grid of generated images at each checkpoint
- `metric-fid50k_full.jsonl` — FID score log (lower = better)
- `training_stats.jsonl` — loss curves

### 3.4 Monitor training

Open the `fakes` PNGs as they appear. Because you're starting from MetFaces, the outputs should look recognizably painterly much earlier than training from scratch.

With transfer learning:
- At **200 kimg** you should see painterly forms with Hafftka-ish color
- At **1000–1500 kimg** the "latent shift" happens — Hafftka's gestural style takes over from the MetFaces prior. Before this point outputs may look like classical museum portraits with Hafftka's palette. This is normal. Do not adjust training, do not panic.
- At **3000+ kimg** you should see strong results

Without transfer learning (from scratch), those milestones typically happen 3–5x later.

**FID score guidance for art datasets** (not faces, so FID is less meaningful than visual inspection):
- < 50: Good. Visual quality is solid.
- 50–100: Acceptable for artistic work.
- > 100: The model may be underfitting or mode-collapsing.

### 3.5 Resuming training

If you need to stop and resume:

```bash
python train.py ^
  --outdir=C:/hafftka-runs ^
  --data=C:/hafftka-dataset/hafftka-512.zip ^
  --cfg=stylegan2 ^
  --gpus=1 ^
  --batch=32 ^
  --gamma=10 ^
  --aug=ada ^
  --mirror=1 ^
  --snap=10 ^
  --kimg=10000 ^
  --resume=C:/hafftka-runs/00000-stylegan2-hafftka-512-gpus1-batch32-gamma10/network-snapshot-005000.pkl
```

### 3.6 If training goes wrong

**Mode collapse** (all outputs look identical):
- Lower `--gamma` to `5` or raise `--target` to `0.7`

**Training unstable / losses diverging**:
- Reduce `--batch` to `16`, reduce learning rate isn't directly exposed but lower batch helps

**Out of VRAM**:
- Reduce `--batch` to `16` or drop to `256×256` resolution

---

## Stage 3.5: Model Improvements

These are research-backed techniques that apply directly to your setup: small dataset, single GPU, art domain. They are ordered by effort vs. payoff. Do them in sequence after your first working training run, not all at once.

---

### Improvement 1: Dataset Augmentation (free, do this before any training)

With 3–4k paintings, your effective dataset is small. Generating multi-scale crops from the same paintings creates new training samples without inventing data — you're teaching the model to understand Hafftka at different scales.

```python
# augment_dataset.py
# Run this BEFORE dataset_tool.py conversion
# Adds 2 random crops per image, roughly doubling dataset size

import os
from PIL import Image
import random

src_dir = "C:/hafftka-dataset/images"
out_dir = "C:/hafftka-dataset/images-augmented"
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    fpath = os.path.join(src_dir, fname)
    try:
        img = Image.open(fpath).convert("RGB")
        # Copy original
        img.save(os.path.join(out_dir, fname))
        w, h = img.size
        # Two random crops at 60–90% of original size
        for i in range(2):
            scale = random.uniform(0.6, 0.9)
            cw, ch = int(w * scale), int(h * scale)
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
            crop = img.crop((x, y, x + cw, y + ch))
            crop_name = f"{os.path.splitext(fname)[0]}_crop{i}.png"
            crop.save(os.path.join(out_dir, crop_name))
    except Exception as e:
        print(f"Skipped {fname}: {e}")

print("Done. Check C:/hafftka-dataset/images-augmented/")
```

Then convert the augmented directory instead:

```bash
python dataset_tool.py ^
  --source=C:/hafftka-dataset/images-augmented ^
  --dest=C:/hafftka-dataset/hafftka-512-augmented.zip ^
  --resolution=512x512
```

Use `hafftka-512-augmented.zip` as your `--data` flag in all training runs.

---

### Improvement 2: Transfer Learning from MetFaces (already in Stage 3)

Covered in Stage 3.2–3.3. The `--resume=metfaces.pkl` flag is the single highest-payoff change. A model that already knows paintings reaches good Hafftka quality 3–5x faster than training from random noise.

If you want to experiment with a different starting point, FFHQ (faces) is less ideal than MetFaces for painting data. The MetFaces model's prior is much closer to what Hafftka makes.

---

### Improvement 3: Projected GAN Discriminator (experimental, separate run)

This is the most significant architectural advance since ADA (NeurIPS 2021). Instead of the discriminator learning from raw pixels, it judges images using features from a pretrained EfficientNet — a network that already understands visual concepts from ImageNet. The result: faster convergence, better quality on small datasets.

**The tradeoff:** Projected GAN's latent space is more volatile during training, which makes GANSpace exploration slightly less stable. Run this as a separate experiment after you have a good baseline model from the standard ADA run. Compare the outputs visually.

```bash
# Clone Projected GAN
cd C:\Users\<your-username>
git clone https://github.com/autonomousvision/projected_gan.git
cd projected_gan
pip install -r requirements.txt
```

```bash
python train.py ^
  --outdir=C:/hafftka-runs-projected ^
  --data=C:/hafftka-dataset/hafftka-512-augmented.zip ^
  --cfg=fastgan ^
  --gpus=1 ^
  --batch=32 ^
  --mirror=1 ^
  --snap=10 ^
  --kimg=3000
```

Projected GAN uses FastGAN as the generator by default (lighter architecture, faster training). At 3000 kimg it will likely match or beat your ADA run at 5000 kimg. Generate a grid from both and compare.

**How to compare the two runs fairly:**

Train both from the same starting data. Compare `fakes` grids at equivalent wall-clock time — not kimg, since Projected GAN trains faster. The question is which model's latent space is more useful for GANSpace exploration. Visual quality is secondary.

---

### Improvement 4: Increase Model Capacity (later, not first)

If you have a working model and quality plateaus, doubling the generator's feature map count adds detail. This is a last resort — it makes training 3× slower.

Add these flags to your training command:

```bash
  --cbase=65536 ^
  --cmax=512
```

Default values are `--cbase=32768 --cmax=512`. Doubling `cbase` roughly doubles the feature maps at higher resolutions. Your 4080 Super with 16GB VRAM can handle it at 512×512 with batch 16. Drop `--batch` to 16 if you hit memory errors.

Only do this after you've confirmed the baseline model is working and you want to push quality further.

---

### The Recommended Sequence

```
Run 1 (baseline):
  augmented dataset + MetFaces resume + bgcfnc augpipe
  → train to 3000–5000 kimg
  → evaluate visually

Run 2 (if quality is good, push further):
  same as Run 1 but resume from Run 1 checkpoint
  → increase cbase to 65536, drop batch to 16
  → train another 3000 kimg

Run 3 (experimental, separate):
  Projected GAN on augmented dataset
  → train to 2000–3000 kimg
  → compare to Run 1 at same wall-clock time
  → use whichever has better visual quality for GANSpace
```

Do not run all three simultaneously. Disk space and checkpoints add up fast.

---

## Stage 4: GANSpace Analysis

GANSpace finds the meaningful directions in W-space automatically via PCA. This is what gives you the semantic sliders — the model's own learned axes.

### 4.1 Clone GANSpace

In a new terminal (still in the `hafftka` conda env):

```bash
cd C:\Users\<your-username>
git clone https://github.com/harskish/ganspace.git
cd ganspace
pip install -r requirements.txt
```

### 4.2 Run PCA on your trained model

```bash
python visualize.py ^
  --model=StyleGAN2 ^
  --model_path=C:/hafftka-runs/00000-stylegan2-hafftka-512-gpus1-batch32-gamma10/network-snapshot-010000.pkl ^
  --out_dir=C:/hafftka-ganspace ^
  --n_samples=50000 ^
  --n_components=80
```

This samples 50,000 W vectors from your model, runs PCA, and saves 80 principal components. Each component is a direction in W-space that corresponds to a distinct visual axis the model learned.

### 4.3 Explore components interactively

GANSpace has a built-in interactive viewer:

```bash
python interactive.py ^
  --model=StyleGAN2 ^
  --model_path=C:/hafftka-runs/00000-stylegan2-hafftka-512-gpus1-batch32-gamma10/network-snapshot-010000.pkl ^
  --directions_path=C:/hafftka-ganspace/directions.pt
```

This opens a GUI with sliders. Each slider is one PCA component. Move slider 0, see what changes. Move slider 1, see something else. The first 10–15 components will be the most visually dramatic.

**What to look for in Hafftka's space:**
- Component that controls warm/cool color temperature
- Component that controls figure density vs. negative space
- Component that controls gestural mark-making intensity
- Component that controls compositional verticality vs. horizontality

Name them as you find them. These become your navigation axes.

---

## Stage 5: Project Real Paintings

This is the Portrait of a Process move: take actual Hafftka works and find their coordinates in the learned space.

### 5.1 Prepare target images

Pick 10–20 real Hafftka paintings from the dataset. Crop them to 512×512 (same as training resolution). Put them in `C:\hafftka-targets\`.

### 5.2 Project each image into W+ space

```bash
python projector.py ^
  --outdir=C:/hafftka-projections ^
  --target=C:/hafftka-targets/painting_001.png ^
  --network=C:/hafftka-runs/00000-stylegan2-hafftka-512-gpus1-batch32-gamma10/network-snapshot-010000.pkl ^
  --num-steps=1000
```

Outputs:
- `proj.png` — the model's reconstruction of the painting
- `projected_w.npz` — the W+ latent vector (the painting's coordinates)
- `proj.mp4` — video of the projection converging

Run this for each target painting. You now have W+ coordinates for real works.

### 5.3 Interpolate between two real paintings

```python
# interpolate.py
import numpy as np
import torch
import pickle
from PIL import Image

# Load model
with open('C:/hafftka-runs/.../network-snapshot-010000.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

# Load two projected paintings
w1 = np.load('C:/hafftka-projections/painting_001/projected_w.npz')['w']
w2 = np.load('C:/hafftka-projections/painting_002/projected_w.npz')['w']

# Generate 12 frames between them
frames = []
for i, t in enumerate(np.linspace(0, 1, 12)):
    w_interp = (1 - t) * w1 + t * w2
    w_tensor = torch.tensor(w_interp).cuda()
    img = G.synthesis(w_tensor, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    frames.append(Image.fromarray(img[0].cpu().numpy()))

# Save as grid
widths = [f.width for f in frames]
grid = Image.new('RGB', (sum(widths), frames[0].height))
x = 0
for f in frames:
    grid.paste(f, (x, 0))
    x += f.width
grid.save('C:/hafftka-interpolation.png')
```

This is the content of Portrait of a Process — the journey between two real paintings through the model's learned understanding of Hafftka.

---

## Stage 6: The Browser

Build a 2D navigation grid using two GANSpace components as axes. Every cell in the grid is a generated image. Navigate by choosing which two components to use as X and Y.

### 6.1 Generate a component grid

```python
# grid_browser.py
import numpy as np
import torch
import pickle
from PIL import Image

with open('C:/hafftka-runs/.../network-snapshot-010000.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

directions = torch.load('C:/hafftka-ganspace/directions.pt')

# Pick your two axes (experiment with these numbers)
component_x = 0   # e.g. "color temperature"
component_y = 2   # e.g. "compositional density"
strength = 8.0    # how far to push each direction
grid_size = 7     # 7x7 grid = 49 images

# Base latent vector (the center of the space)
z = torch.randn([1, G.z_dim]).cuda()
w_base = G.mapping(z, None, truncation_psi=0.7)

images = []
x_vals = np.linspace(-strength, strength, grid_size)
y_vals = np.linspace(-strength, strength, grid_size)

for y in y_vals:
    row = []
    for x in x_vals:
        w = w_base.clone()
        w[:, :, :] += x * directions[component_x]
        w[:, :, :] += y * directions[component_y]
        img = G.synthesis(w, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        row.append(Image.fromarray(img[0].cpu().numpy()))
    images.append(row)

# Assemble grid
cell_w, cell_h = images[0][0].size
grid = Image.new('RGB', (cell_w * grid_size, cell_h * grid_size))
for row_idx, row in enumerate(images):
    for col_idx, img in enumerate(row):
        grid.paste(img, (col_idx * cell_w, row_idx * cell_h))

grid.save(f'C:/hafftka-grid-c{component_x}-c{component_y}.png')
print("Saved.")
```

Run this for different component pairs. Each output is a 7×7 map of the learned Hafftka space along two axes.

### 6.2 The interactive version (optional but worth it)

If you want a live clickable browser rather than static grids, use the built-in StyleGAN2-ADA visualizer:

```bash
python visualizer.py
```

Then load your trained `.pkl`. This is the merzazine-style browser — you click and drag through the latent space in real time. With your 4080 Super you'll get near-instant generation.

---

## Stage 7: What to Post

### The grid
The 7×7 component grid is the screenshot. Top-left to bottom-right is a journey across two dimensions of Hafftka's visual vocabulary. Caption it with what the axes mean once you've identified them.

### The interpolation video
Two real Hafftka paintings, 120 frames between them, exported as MP4. This is the work that earns the credit line: "trained on the Hafftka catalog raisonné, CC-BY-NC-4.0."

### The thread structure
1. What Hafftka did — why his catalog matters (museum collections, the open-source decision)
2. What you did — trained a GAN on it, found its axes
3. The grid — the learned space visualized
4. The interpolation — two real works, the journey between
5. Credit: Hafftka, CC-BY-NC-4.0

---

## File Structure Reference

```
C:\
├── hafftka-dataset\
│   ├── images\                    ← raw downloaded images
│   ├── images-augmented\          ← originals + multi-scale crops
│   ├── hafftka-512.zip            ← converted from raw (backup)
│   ├── hafftka-512-augmented.zip  ← converted from augmented (use this)
│   └── metfaces-1024x1024.pkl     ← transfer learning starting point
├── hafftka-runs\
│   └── 00000-...\                 ← ADA + MetFaces baseline run
│       ├── network-snapshot-005000.pkl
│       ├── network-snapshot-010000.pkl
│       ├── fakes005000.png
│       └── metric-fid50k_full.jsonl
├── hafftka-runs-projected\        ← Projected GAN experimental run
│   └── 00000-...\
├── hafftka-ganspace\
│   └── directions.pt              ← PCA components from GANSpace
├── hafftka-targets\               ← real paintings to project
├── hafftka-projections\           ← W+ coordinates of real paintings
├── stylegan2-ada-pytorch\
└── projected_gan\                 ← Projected GAN repo (optional)
```

---

## Estimated Timeline

| Stage | Time |
|---|---|
| Environment setup | 1–2 hours |
| Dataset download | 30–60 minutes |
| Dataset augmentation (crop script) | 15–30 minutes |
| Dataset conversion | 15–30 minutes |
| MetFaces checkpoint download | 10 minutes |
| Training Run 1 to 5000 kimg (512px, MetFaces start) | 24–36 hours |
| Training Run 2 (capacity increase, optional) | +18–24 hours |
| Projected GAN Run 3 (optional) | 12–18 hours |
| GANSpace PCA | 20–30 minutes |
| Projecting 10 real paintings | 2–3 hours |
| Generating grids and interpolations | 30 minutes |

Transfer learning from MetFaces cuts the baseline training time roughly in half vs. training from scratch. Run 1 is the one to get right. Runs 2 and 3 are experiments once you have a working model.

---

## Troubleshooting Quick Reference

| Error | Fix |
|---|---|
| `CUDA kernels failed to build` | Run `vcvars64.bat` first; confirm VS2022 C++ tools installed |
| `conv2d_gradfix` error | You're on PyTorch > 1.13 — re-create env with pinned 1.13.1 |
| `out of memory` | Reduce `--batch` to 16; or drop to 256×256 for testing |
| `out of memory` with capacity increase | Drop `--batch` to 16 when using `--cbase=65536` |
| `train.py` hangs on `Setting up PyTorch plugin "bias_act_plugin"` for 10+ minutes | `vcvars64.bat` hasn't hit the C++ compiler correctly. Close prompt, re-run `vcvars64.bat`, then `train.py` again. If it completes and prints `Done.`, ninja compiled successfully. |
| MetFaces checkpoint size mismatch warning | Normal — it adapts from 1024 to 512. Continue. |
| FID plateaus early | Increase `--target` to 0.7; confirm `--augpipe=bgcfnc` is set |
| Outputs look like MetFaces, not Hafftka | Normal before 1000–1500 kimg. The latent shift takes time. Do not stop training. |
| Projected GAN outputs look blurry | Expected early in training; Projected GAN converges differently — check at 1500+ kimg |
| Projected GAN install errors | Requires separate pip deps; don't mix with ADA conda env |
| Paintings center-cropped badly | Use `pad_to_square.py` (Stage 2.3.1) on extreme aspect ratios before conversion |
| Projection looks nothing like target | Increase `--num-steps` to 2000; crop target to match training res |
| GANSpace `directions.pt` not found | Check `--out_dir` path; run PCA step before interactive |

---

## Credit Line for All Outputs

> Trained on the Michael Hafftka Catalog Raisonné (Hugging Face: `Hafftka/michael-hafftka-catalog-raisonne`), CC-BY-NC-4.0. StyleGAN2-ADA, NVlabs. GANSpace, Härkönen et al.