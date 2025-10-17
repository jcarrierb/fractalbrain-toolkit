# fractalbrain/backends.py
import os, importlib

def _current_device():
    return os.environ.get("FRACTALBRAIN_DEVICE", "cpu").lower()

def xp():
    dev = _current_device()
    if dev in ("cuda", "gpu"):
        try:
            return importlib.import_module("cupy")
        except Exception:
            # fallback CPU si CuPy absent
            return importlib.import_module("numpy")
    return importlib.import_module("numpy")

def asnumpy(a):
    mod = xp()
    if mod.__name__ == "cupy":
        return importlib.import_module("cupy").asnumpy(a)
    return a
