# cudaJourney

`cudaJourney` is a personal CUDA learning repository focused on building GPU intuition through small, self-contained programs.

The codebase currently covers:

- vector operations such as elementwise add, reductions, prefix sums, histograms, and merge-based routines
- matrix operations such as GEMM, 3D stencil kernels, sparse matrix-vector multiplication, and graph traversal experiments
- image processing kernels such as grayscale conversion, blur, and convolution-based edge detection

Most of the work is based on the Oak Ridge National Laboratory CUDA training material, with additional experiments and profiling notes added along the way.

## Repository Layout

| Path | Focus |
| --- | --- |
| `vector_operations/` | foundational CUDA kernels for scan, reduction, histogram, and related patterns. Results: `vector_operations/RESULTS_SUMMARY.md` |
| `matrix_operations/` | matrix multiplication, stencil, sparse operations, and graph-oriented kernels. Results: `matrix_operations/RESULTS_SUMMARY.md` |
| `ImageProcessing/` | image transforms using CUDA and OpenCV-backed host I/O. Results: `ImageProcessing/RESULTS_SUMMARY.md` |

## Requirements

- NVIDIA GPU with a working CUDA runtime
- CUDA Toolkit with `nvcc`
- OpenCV for the image-processing programs

## Build and Run

There is no single build system yet. Each experiment is compiled directly with `nvcc`.

Example:

```bash
nvcc -O2 -o matrix_operations/1_matrix_multiplications matrix_operations/1_matrix_multiplications.cu
./matrix_operations/1_matrix_multiplications 3 1 555 777 999 0
```

For image-processing programs, link OpenCV in the compile step. One common pattern is:

```bash
nvcc -O2 ImageProcessing/1_grayscale_conversion.cu -o ImageProcessing/grayscale_conversion $(pkg-config --cflags --libs opencv4)
./ImageProcessing/grayscale_conversion DEVICE BASIC
```

The source files print their expected CLI arguments when invoked incorrectly, so the fastest way to explore a kernel is to compile it and run it once with no parameters.

## Notes

- Some folders include screenshots or generated images that document kernel results and Nsight observations.
- Performance numbers in the checked-in result markdown files are machine-specific and were collected on the author’s local setup.

## Why This Repo Exists

This repository is intentionally incremental. The goal is not a framework or library; it is a documented CUDA learning path with runnable experiments, saved result snapshots, and a clear record of optimization attempts.
