# mnist-kernel

A small MNIST multilayer perceptron implemented from scratch in C++20. The
default implementation uses OpenMP and trains with a Muon + AdamW optimizer.
A legacy C implementation is also included.

## CPU build

Requirements:

- CMake 3.20 or newer
- A C++20 compiler
- OpenMP

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

To also build the legacy C implementation:

```bash
cmake -S . -B build -DMNIST_BUILD_LEGACY_C=ON
cmake --build build --parallel
```

## CUDA build

Requirements:

- NVIDIA CUDA Toolkit with C++20 support
- cuBLAS
- A compatible NVIDIA driver and GPU

Compile the CUDA implementation directly with `nvcc`:

```bash
nvcc -std=c++20 -O3 --use_fast_math -lineinfo \
    mnist.cu -lcublas -o mnist-cuda
```

Run it with the default 20 epochs and `./data` directory:

```bash
./mnist-cuda
```

The only optional argument is the number of epochs. The CUDA program always
loads the MNIST files from `./data`:

```bash
./mnist-cuda 20
```

## Data

Place the four raw binary files below in `data/`:

- `X_train.bin`: 60,000 × 784 little-endian `float32` values
- `y_train.bin`: 60,000 little-endian `int32` labels
- `X_test.bin`: 10,000 × 784 little-endian `float32` values
- `y_test.bin`: 10,000 little-endian `int32` labels

The dataset binaries are intentionally not tracked because `X_train.bin`
exceeds GitHub's regular 100 MB per-file limit.
