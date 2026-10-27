"""File contracts shared by completed transfer follow-up analyses.

No scientific filtering, imputation or estimator belongs in this module.
"""
import csv
import hashlib
from pathlib import Path

_INTS = ("feature", "sentence", "cluster", "language_index")
_TEXT = ("kind", "split")


def rows_from_csv(path):
    """The per-item table of a report run as the row dictionaries the design reads: {(feature, sentence, kind): row}. Integers for the ids, a boolean for `eligible`, text for `kind` and `split`, floats for everything
    else, a blank cell is None (an item with a missing value is dropped by `design`, never filled)."""
    rows = {}
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            r = {}
            for k, v in rec.items():
                if k in _TEXT:
                    r[k] = v
                elif v == "":
                    r[k] = None
                elif k in _INTS:
                    r[k] = int(v)
                elif k == "eligible":
                    r[k] = v == "True"
                else:
                    r[k] = float(v)
            key = (r["feature"], r["sentence"], r["kind"])
            if key in rows:
                raise ValueError(f"item {key} occurs twice in {path}")
            rows[key] = r
    return rows


def sha256_file(path):
    """SHA256 of the exact stored bytes; filesystem errors propagate unchanged."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, columns=None):
    """Write rows with original ordering/newlines and optional fixed columns.

    Without explicit columns, use first appearance across all rows. With explicit
    columns, retain DictWriter's rejection of unexpected keys. Do not sort rows.
    """
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
