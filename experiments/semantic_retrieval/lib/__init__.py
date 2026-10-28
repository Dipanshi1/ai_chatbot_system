"""
B1 experiment library.

Submodules are imported explicitly by callers. This package must not import torch or
sentence-transformers at import time, so that resource probes can take the RSS0
checkpoint before those libraries are loaded (spec §11.1).
"""
