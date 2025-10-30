from dataclasses import dataclass
from pathlib import Path
import os

try:
    # Load environment variables from .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; skip if not installed
    pass


@dataclass
class DatasetConfig:
    root: Path = Path(
        os.getenv(
            "DATASET_ROOT",
            str(Path(__file__).resolve().parents[2] / "IS 2025 OrganAMNIST"),
        )
    )
    train_images: Path = Path(os.getenv("TRAIN_IMAGES_DIR", str(root / "train" / "images_train")))
    train_labels: Path = Path(os.getenv("TRAIN_LABELS_CSV", str(root / "train" / "labels_train.csv")))
    val_images: Path = Path(os.getenv("VAL_IMAGES_DIR", str(root / "val" / "images_val")))
    val_labels: Path = Path(os.getenv("VAL_LABELS_CSV", str(root / "val" / "labels_val.csv")))
    test_images: Path = Path(os.getenv("TEST_IMAGES_DIR", str(root / "test" / "images")))
    test_manifest: Path = Path(os.getenv("TEST_MANIFEST_CSV", str(root / "test" / "manifest_public.csv")))


@dataclass
class OutputConfig:
    root: Path = Path(
        os.getenv(
            "OUTPUT_ROOT",
            str(Path(__file__).resolve().parents[2] / "analysis_outputs"),
        )
    )
    analysis_root: Path = Path(os.getenv("ANALYSIS_ROOT", str(root / "analysis")))
    models_root: Path = Path(os.getenv("MODELS_ROOT", str(root / "models")))
    figures: Path = Path(os.getenv("FIGURES_DIR", str(analysis_root / "figures")))
    tables: Path = Path(os.getenv("TABLES_DIR", str(analysis_root / "tables")))
    reports: Path = Path(os.getenv("REPORTS_DIR", str(analysis_root / "reports")))


DATASET_CONFIG = DatasetConfig()
OUTPUT_CONFIG = OutputConfig()
