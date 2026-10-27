"""Compatibility entry point for :mod:`transfer_followup.cross_corpus`.

The implementation moved; existing commands and imports continue to work.
Historical source is retained at commit e17f926.
"""
import sys
from transfer_followup import cross_corpus as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
