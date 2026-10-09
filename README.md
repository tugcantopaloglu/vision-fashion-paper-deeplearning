# VisionFashion

Research notebook for fashion image and text embeddings using a Vision Transformer and BERT. The notebook trains a contrastive model and evaluates image-to-text and text-to-image retrieval.

## Repository contents

| File | Purpose |
| --- | --- |
| `VisionFashion.ipynb` | Google Colab notebook for data loading, contrastive training, retrieval evaluation and embedding examples. |
| [VisionFashion_TugcanTopaloglu.pdf](VisionFashion_TugcanTopaloglu.pdf) | Author's research paper, including experiments beyond the implemented notebook. |
| `tests/test_notebook.py` | Small CPU regression fixtures that do not download models or data. |
| `LICENSE` | MIT license for the repository code. |

The linked [Hugging Face model repository](https://huggingface.co/tugcantopaloglu/visionfashion) is external to this checkout. Dataset files, trained checkpoints and a complete experiment environment lock are not included.

## Running the research notebook

```sh
git clone https://github.com/tugcantopaloglu/vision-fashion-paper-deeplearning.git
cd vision-fashion-paper-deeplearning
```

Open `VisionFashion.ipynb` in Google Colab with a GPU runtime. Its setup cell installs Python packages and mounts Google Drive. The existing Colab and Drive paths require adjustment for another environment; the notebook is not a standalone local application.

1. Obtain the DeepFashion-MultiModal dataset under its own access and license terms.
2. Set `DRIVE_BASE_PATH`, `IMAGES_ZIP_PATH`, `CAPTIONS_JSON_PATH` and `COLAB_WORKING_DIR` to your data locations. The notebook reads an image archive and `captions.json`, not a CSV of captions or prearranged train/valid/test folders.
3. Captions can be a filename-to-caption JSON object or a list of objects with `image` and `caption` keys. Image filenames must match files in the extracted directory. Both a flat image archive and an archive with an `images/` subdirectory are supported.
4. Run cells in order. The notebook loads `google/vit-base-patch16-224-in21k` and `bert-base-uncased` from Hugging Face, requiring network access or a populated local cache.
5. Review `BATCH_SIZE`, `LEARNING_RATE` and `EPOCHS` before starting training. The checked-in defaults are 16, `1e-5` and 10.

At least three valid image-caption pairs are required for nonempty train, validation and test partitions. Splits use seed 42; keep the input file order, data and package versions fixed when comparing runs. GPU kernels are not configured for strict deterministic execution. Encoder and processor downloads use pinned Hugging Face revisions in the configuration cell; these are maintenance pins, not a record of the original experiment revisions.

The notebook saves `best_multimodal_fashion_model.pth` based on validation loss in the runtime working directory. State-dict loads use `weights_only=True`. Treat checkpoints as trusted external inputs and verify that they match the model architecture.

The category/attribute evaluation cell is a scaffold. `MultiModalFashionModel` does not implement `predict_category` or `predict_attributes`, and the dataset does not return category or attribute labels. Reproducing that phase requires the original annotated data, classification heads and training implementation.

## Paper-reported results

These are values from the accompanying paper's Tables I-III, not results reproduced by the checked-in notebook or the maintenance tests. The PDF and historical notebook outputs remain unchanged.

| Task | Metric | Paper value |
| --- | --- | --- |
| Image-to-text retrieval | R@10 | 0.5492 |
| Text-to-image retrieval | R@10 | 0.5539 |
| Category prediction | Top-1 accuracy | 0.9470 |
| Attribute prediction | Average recall@5 | 0.7291 |

The paper describes an A100 experiment and separate contrastive and classification phases. No smaller-GPU capacity or reproduction of these metrics has been verified here. For retrieval sets smaller than a requested K, evaluation searches all available candidates while keeping the R@K label.

## Local regression checks

Maintenance fixtures passed on Python 3.13.1 with PyTorch 2.13.0. To run them with a suitable CPU PyTorch installation:

Create and activate a virtual environment for your shell, then install the fixture dependencies.

```sh
python -m venv .venv
python -m pip install torch numpy Pillow nbformat ipython
python -m unittest discover -s tests -v
```

The fixtures execute selected notebook definitions with tiny images, identity embeddings and processor stubs. They check notebook syntax and evaluation ordering, JSON caption loading, directory selection, nonempty seeded splits, small-set retrieval, skipped-batch loss accounting and embedding examples. They do not load pretrained ViT/BERT weights, train the research model or establish dataset quality.

## Reproduction requirements

- Data: original images, captions, annotation definitions and experiment split provenance are external. A fixed seed alone does not recover the author's earlier split.
- Models: pretrained encoders and matching learned state dictionaries are external. Linked model weights have not been loaded in the fixture tests.
- Hardware: a working GPU runtime and sufficient RAM/VRAM for the chosen dataset and batch size are needed for full training. Capacity and full inference performance require separate measurement.
- Experiments: record Python/package versions, dataset version, split indices, seeds, hardware, checkpoints and evaluation settings. Changes to split handling and invalid-batch accounting may change newly generated metrics.

## Citation

```bibtex
@unpublished{topaloglu2025visionfashion,
  author = {Tuğcan Topaloğlu},
  title = {{VisionFashion}: Multi-Modal Style Embedding Learning with Vision Transformers and BERT for Fashion Image Analysis and Recommendation},
  year = {2025},
  note = {Work in progress},
  url = {https://github.com/tugcantopaloglu/vision-fashion-paper-deeplearning}
}
```

## License

Code is MIT licensed. The DeepFashion dataset and external pretrained models have their own access and license terms.
