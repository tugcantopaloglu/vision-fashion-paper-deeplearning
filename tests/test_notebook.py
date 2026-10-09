import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
from IPython.core.inputtransformer2 import TransformerManager
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


NOTEBOOK = Path(__file__).resolve().parents[1] / "VisionFashion.ipynb"


class Progress:
    def __init__(self, items, **kwargs):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def set_postfix(self, *args, **kwargs):
        pass


class ImageProcessor:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def __call__(self, images, return_tensors):
        values = torch.from_numpy(np.array(images).copy()).permute(2, 0, 1).float() / 255
        return {"pixel_values": values.unsqueeze(0)}


class Tokenizer:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def encode_image(self, values):
        means = values.flatten(1).mean(1) * self.weight
        return torch.stack((means, torch.ones_like(means)), dim=1)

    def encode_text(self, input_ids, attention_mask):
        return torch.ones(input_ids.shape[0], 2) * self.weight

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.encode_image(pixel_values), self.encode_text(input_ids, attention_mask)

    def contrastive_loss(self, image_embeddings, text_embeddings):
        return ((image_embeddings - text_embeddings) ** 2).mean()


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = nbformat.read(NOTEBOOK, as_version=4)
        self.namespace = {"os": os, "json": json, "Image": Image, "torch": torch,
                          "np": np, "Dataset": Dataset, "DataLoader": DataLoader,
                          "tqdm": Progress, "ViTImageProcessor": ImageProcessor,
                          "BertTokenizer": Tokenizer, "AdamW": torch.optim.AdamW}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def load_functions(self):
        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(TransformerManager().transform_cell(cell.source))
            nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            nodes = [n for n in nodes if not isinstance(n, ast.ClassDef) or n.name == "FashionMultiModalDataset"]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), self.namespace)

    def prepare_data(self, count):
        images = self.root / "images"
        images.mkdir(exist_ok=True)
        captions = {}
        for i in range(count):
            name = f"image-{i}.png"
            Image.new("RGB", (8, 8), (40 + i, 60, 80)).save(images / name)
            captions[name] = "blue shirt"
        caption_path = self.root / "captions.json"
        caption_path.write_text(json.dumps(captions), encoding="utf-8")
        self.namespace.update(IMAGE_DIR_UNZIPPED=str(images), CAPTIONS_JSON_PATH=str(caption_path),
                              IMG_MODEL_NAME="offline-image", TEXT_MODEL_NAME="offline-text",
                              BATCH_SIZE=2, SEED=42, IMG_MODEL_REVISION="offline-image-revision", TEXT_MODEL_REVISION="offline-text-revision")
        return images, caption_path

    def run_data_cell(self):
        cell = next(c for c in self.notebook.cells if "class FashionMultiModalDataset" in c.source)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell.source, str(NOTEBOOK), "exec"), self.namespace)

    def test_notebook_code_is_syntactically_valid(self):
        nbformat.validate(self.notebook)
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                compile(TransformerManager().transform_cell(cell.source), str(NOTEBOOK), "exec")

    def test_caption_formats_and_rgb_tensor_loading(self):
        self.load_functions()
        images, caption_path = self.prepare_data(3)
        load = self.namespace["load_text_descriptions"]
        expected = {"image-0.png": "café shirt"}
        caption_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(load(str(caption_path)), expected)
        caption_path.write_text(json.dumps([{"image": "image-0.png", "caption": "café shirt"}]), encoding="utf-8")
        self.assertEqual(load(str(caption_path)), expected)
        dataset = self.namespace["FashionMultiModalDataset"](str(images), expected, ImageProcessor(), Tokenizer())
        item = dataset[0]
        self.assertEqual(tuple(item["pixel_values"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["input_ids"].shape), (4,))

    def test_splits_are_nonempty_disjoint_and_repeatable(self):
        for count in (3, 6, 20):
            self.prepare_data(count)
            self.run_data_cell()
            partitions = [self.namespace[k].indices for k in ("train_dataset", "val_dataset", "test_dataset")]
            self.assertTrue(all(partitions))
            self.assertEqual(sorted(sum(partitions, [])), list(range(count)))
            self.run_data_cell()
            repeated = [self.namespace[k].indices for k in ("train_dataset", "val_dataset", "test_dataset")]
            self.assertEqual(partitions, repeated)

    def test_too_few_pairs_fail_before_creating_loaders(self):
        self.prepare_data(2)
        with self.assertRaisesRegex(ValueError, "At least three"):
            self.run_data_cell()

    def test_image_directory_accepts_flat_and_nested_archives(self):
        cell = next(c for c in self.notebook.cells if c.source.startswith("nested_image_dir ="))
        self.namespace["IMAGE_DIR_UNZIPPED"] = str(self.root)
        exec(cell.source, self.namespace)
        self.assertEqual(self.namespace["IMAGE_DIR_UNZIPPED"], str(self.root))
        (self.root / "images").mkdir()
        exec(cell.source, self.namespace)
        self.assertEqual(Path(self.namespace["IMAGE_DIR_UNZIPPED"]), self.root / "images")

    def test_retrieval_with_fewer_than_ten_candidates(self):
        self.load_functions()
        metric = self.namespace["calculate_retrieval_metrics"]
        embeddings = torch.eye(3)
        image, text = metric(embeddings, embeddings)
        self.assertTrue(all(v == 1.0 for v in list(image.values()) + list(text.values())))
        image, text = metric(embeddings, embeddings[[1, 0, 2]])
        self.assertAlmostEqual(image["R@1 (I2T)"], 1 / 3)
        self.assertEqual(image["R@10 (I2T)"], 1.0)
        self.assertEqual(text["R@5 (T2I)"], 1.0)

    def test_skipped_batches_do_not_reduce_reported_loss(self):
        self.load_functions()
        self.prepare_data(3)
        self.run_data_cell()
        batch = next(iter(self.namespace["train_dataloader"]))
        model = TinyModel()
        validate = self.namespace["validate_epoch"]
        expected = validate(model, [batch], torch.device("cpu"))
        self.assertEqual(validate(model, [None, batch, None], torch.device("cpu")), expected)
        with self.assertRaisesRegex(ValueError, "No valid batches"):
            validate(model, [None], torch.device("cpu"))

    def test_training_evaluation_and_examples_work_in_cell_order(self):
        self.prepare_data(3)
        self.run_data_cell()
        self.namespace.update(model=TinyModel(), device=torch.device("cpu"), LEARNING_RATE=1e-3, EPOCHS=1)
        old_cwd = Path.cwd()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                for cell in self.notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    if cell.source.startswith("@torch.no_grad()") and "def evaluate_model" in cell.source:
                        exec(cell.source, self.namespace)
                    elif cell.source.startswith("optimizer = AdamW"):
                        exec(cell.source, self.namespace)
                    elif "def get_image_embedding" in cell.source:
                        exec(cell.source, self.namespace)
            self.assertIn("final_scores", self.namespace)
            self.assertEqual(self.namespace["img_emb"].shape, (1, 2))
            self.assertNotIn("hata:", output.getvalue().lower())
            self.assertTrue((self.root / "best_multimodal_fashion_model.pth").is_file())
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
