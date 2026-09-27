import sys
import numpy as np
import torch

print(f"Python {sys.version}")

print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Vector: {a}, dot product with itself: {np.dot(a, a)}")

print(f"CUDA available: {torch.cuda.is_available()}")           # False on macOS — expected
print(f"MPS available:  {torch.backends.mps.is_available()}")   # True on Apple Silicon
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot product of {a} and {b}: {np.dot(a, b)}")

