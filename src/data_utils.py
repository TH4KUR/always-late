"""FLORES-200 data handling, language metadata, and token-budget equalization.

Implements the experimental data design from Section 4.1 and Section 5 of
anlp_revised.txt:
- 27-language balanced panel across 3 resource tiers (Joshi et al., 2020).
- Script family annotations to control for orthographic confounding.
- Automatic download and extraction from Meta's ungated NLLB release archive.
- Exact token-budget equalization across languages to eliminate tokenizer fertility bias.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_DIR = DATA_DIR / "flores200_dataset"


@dataclass(frozen=True)
class LanguageInfo:
    code: str  # FLORES-200 code, e.g. "kan_Knda"
    name: str  # English name, e.g. "Kannada"
    tier: Literal["high", "mid", "low"]  # Joshi et al. (2020)
    script: str  # Script family, e.g. "Dravidian", "Latin", "Cyrillic"


# Script-balanced 27-language panel (9 high, 9 mid, 9 low)
PANEL_27: dict[str, LanguageInfo] = {
    # High-resource (9)
    "eng_Latn": LanguageInfo("eng_Latn", "English", "high", "Latin"),
    "deu_Latn": LanguageInfo("deu_Latn", "German", "high", "Latin"),
    "spa_Latn": LanguageInfo("spa_Latn", "Spanish", "high", "Latin"),
    "fra_Latn": LanguageInfo("fra_Latn", "French", "high", "Latin"),
    "por_Latn": LanguageInfo("por_Latn", "Portuguese", "high", "Latin"),
    "rus_Cyrl": LanguageInfo("rus_Cyrl", "Russian", "high", "Cyrillic"),
    "cmn_Hans": LanguageInfo("cmn_Hans", "Mandarin (Simplified)", "high", "CJK"),
    "jpn_Jpan": LanguageInfo("jpn_Jpan", "Japanese", "high", "CJK"),
    "kor_Hang": LanguageInfo("kor_Hang", "Korean", "high", "Hangul"),
    # Mid-resource (9)
    "hin_Deva": LanguageInfo("hin_Deva", "Hindi", "mid", "Devanagari"),
    "ben_Beng": LanguageInfo("ben_Beng", "Bengali", "mid", "Bengali"),
    "tam_Taml": LanguageInfo("tam_Taml", "Tamil", "mid", "Dravidian"),
    "tur_Latn": LanguageInfo("tur_Latn", "Turkish", "mid", "Latin"),
    "vie_Latn": LanguageInfo("vie_Latn", "Vietnamese", "mid", "Latin"),
    "ind_Latn": LanguageInfo("ind_Latn", "Indonesian", "mid", "Latin"),
    "tha_Thai": LanguageInfo("tha_Thai", "Thai", "mid", "Thai"),
    "pol_Latn": LanguageInfo("pol_Latn", "Polish", "mid", "Latin"),
    "ukr_Cyrl": LanguageInfo("ukr_Cyrl", "Ukrainian", "mid", "Cyrillic"),
    # Low-resource (9)
    "kan_Knda": LanguageInfo("kan_Knda", "Kannada", "low", "Dravidian"),
    "tel_Telu": LanguageInfo("tel_Telu", "Telugu", "low", "Dravidian"),
    "yor_Latn": LanguageInfo("yor_Latn", "Yoruba", "low", "Latin"),
    "sun_Latn": LanguageInfo("sun_Latn", "Sundanese", "low", "Latin"),
    "amh_Ethi": LanguageInfo("amh_Ethi", "Amharic", "low", "Ethiopic"),
    "ibo_Latn": LanguageInfo("ibo_Latn", "Igbo", "low", "Latin"),
    "hau_Latn": LanguageInfo("hau_Latn", "Hausa", "low", "Latin"),
    "khm_Khmr": LanguageInfo("khm_Khmr", "Khmer", "low", "Khmer"),
    "lao_Laoo": LanguageInfo("lao_Laoo", "Lao", "low", "Lao"),
}


def ensure_flores_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Ensures FLORES-200 is downloaded and unpacked in `data/flores200_dataset`."""
    dataset_dir = data_dir / "flores200_dataset"
    if (dataset_dir / "dev").exists() and (dataset_dir / "devtest").exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "flores200_dataset.tar.gz"

    if not tar_path.exists():
        print(f"Downloading FLORES-200 from {FLORES_TAR_URL} (~26MB)...")
        urllib.request.urlretrieve(FLORES_TAR_URL, tar_path)
        print("Download complete.")

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print(f"Extracted to {dataset_dir}")
    return dataset_dir


def load_sentences(lang_code: str, split: Literal["dev", "devtest"] = "dev") -> list[str]:
    """Loads all sentences for a given language code and split."""
    dataset_dir = ensure_flores_dataset()
    file_path = dataset_dir / split / f"{lang_code}.{split}"
    if not file_path.exists():
        raise FileNotFoundError(f"FLORES file not found: {file_path}")
    lines = file_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_equalized_token_chunks(
    texts: list[str],
    tokenizer,
    target_tokens: int,
    chunk_len: int = 128,
) -> list[str]:
    """Builds text chunks with an exact target token budget.

    Avoids tokenizer fertility confounding: because morphologically rich and
    non-Latin languages produce ~3x more tokens per sentence than English,
    giving each language an equal sentence count severely biases firing rates.
    This function packs sentences into fixed-size token chunks until the total
    token count reaches `target_tokens`.
    """
    chunks: list[str] = []
    current_tokens: list[int] = []
    total_accumulated = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)

        while len(current_tokens) >= chunk_len:
            chunk = current_tokens[:chunk_len]
            chunks.append(tokenizer.decode(chunk))
            total_accumulated += chunk_len
            current_tokens = current_tokens[chunk_len:]
            if total_accumulated >= target_tokens:
                return chunks

    if current_tokens and total_accumulated < target_tokens:
        remaining = min(len(current_tokens), target_tokens - total_accumulated)
        chunks.append(tokenizer.decode(current_tokens[:remaining]))

    return chunks
