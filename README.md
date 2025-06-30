# VisionFashion: Multi‑Modal Style Embedding

**Multi‑Modal Learning with Vision Transformers and BERT for Fashion Image Analysis and Recommendation**

**Huggingface 🤗: https://huggingface.co/tugcantopaloglu/visionfashion**

---

## Repository Contents

| Path                  | Description                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `VisionFashion.ipynb` | Clean Jupyter notebook that trains the ViT + BERT model in two phases (contrastive pre‑training and classification fine‑tuning). |
| `VisionFashion.pdf`   | Full research paper describing the methodology, experiments and results.                                                         |
| `data/`               | (Not included) Place the **DeepFashion‑MultiModal** dataset splits here.                                                         |
| `checkpoints/`        | (Optional) Put or save trained model weights here.                                                                               |
| `requirements.txt`    | Minimal Python package list to reproduce the study.                                                                              |

> **Tip:** If you cloned this repo with `git lfs`, the PDF will download automatically.  
> Otherwise, grab it from the link below.

---

## Quick Start

```bash
# 1. Clone the repo and enter it
git clone https://github.com/<your‑user>/VisionFashion.git
cd VisionFashion

# 2. Create environment (Python 3.10 recommended)
python -m venv venv
source venv/bin/activate         # on Windows use venv\Scripts\activate
pip install -r requirements.txt

# 3. Download & unzip the DeepFashion‑MultiModal dataset
#    (about 11 GB) into ./data
#    Expected structure:
#    data/
#      train/
#        000123.jpg
#        ...
#      valid/
#      test/
#      captions.csv

# 4. (Optional) Add a pretrained ViT + BERT checkpoint to ./checkpoints

# 5. Launch the notebook
jupyter notebook VisionFashion.ipynb
```

The notebook is fully annotated with **cell‑by‑cell instructions** for:

1. Loading and caching the dataset.
2. Phase 1 – CLIP‑style contrastive learning of image/text embeddings.
3. Phase 2 – Fine‑tuning classification heads for _Category_ and _Attribute_ prediction.
4. Evaluation and visualisation of retrieval metrics and learning curves.

---

## Key Results (Reproduced)

| Task                       | Metric     | Score           |
| -------------------------- | ---------- | --------------- |
| **Image ↔ Text Retrieval** | R@10 (I→T) | **0.549 ± .01** |
|                            | R@10 (T→I) | **0.554 ± .01** |
| **Category Prediction**    | Top‑1 Acc. | **0.947**       |
| **Attribute Prediction**   | Avg. R@5   | **0.729**       |

These figures match those reported in the accompanying paper. fileciteturn0file0

---

## Paper

The complete methodology, ablation studies and references are provided in **VisionFashion.pdf** (see `VisionFashion.pdf` in this repo).

If you use this codebase, please cite:

```bibtex
@unpublished{topaloglu2025visionfashion,
  author  = {Tuğcan Topaloğlu},
  title   = {{VisionFashion}: Multi-Modal Style Embedding Learning with Vision Transformers and BERT for Fashion Image Analysis and Recommendation},
  year    = {2025},
  note    = {Work in progress},
  url     = {https://github.com/tugcantopaloglu/vision-fashion-paper-deeplearning}
}
```

---

## Requirements

```
torch>=2.3
torchvision>=0.18
transformers>=4.41
timm>=0.9
pandas
scikit-learn
matplotlib
```

All dependencies are listed in `requirements.txt`.

---

## Re‑training Tips

- Use a GPU with **at least 16 GB** of VRAM. The full experiment was run on an **A100** (80 GB) but can be reproduced on a **T4/RTX 4000** with batch 32.
- Lower `batch_size` in the notebook if you encounter OOM.
- Monitor **validation contrastive loss**; best checkpoints typically occur between epochs 15‑20.
- The notebook automatically saves the **best contrastive** and **best classification** checkpoints.

---

## Licence

The code is released under the MIT licence.  
The DeepFashion dataset is subject to its own licence terms—please make sure you comply with them before redistribution.

Happy experimenting! ✨
