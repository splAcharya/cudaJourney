# Vector Operations Results Summary

This document summarizes the vector-oriented experiments in this folder using the checked-in timings, console captures, and profiler output now grouped under `results/` and `profiles/`.

## 1. Vector Addition

Source file: `0_add.cu`

The source contains multiple device execution paths, but there is no checked-in timing log for this kernel yet. It should be added later if you want this folder to show a full benchmark progression.

## 2. 1D Reductions

Source file: `1_reductions_sum_1D.cu`

Available kernels in source:

- `devicek_basic`
- `devicek_tiled`
- `devicek_warp_shuffle`

Raw timing file: `results/timings/1_reductions_sm_et_float.txt`

Saved profile: `profiles/1_reductions_sum_1D.ncu-rep`

Recorded run:

| N | Block Size | Blocks | Device Mode | Elapsed Time (ms) |
| --- | ---: | ---: | ---: | ---: |
| `5000` | `256` | `20` | `0` | `4.063072` |

Summary:

- The reduction source already includes multiple kernel variants, but the committed raw data currently captures only one measured device run.
- This is enough to show the experiment exists, but not enough yet to compare the reduction strategies against each other.

## 3. Prefix Sums, Float

Source file: `2_prefix_sums.cu`

Raw timing file: `results/timings/2_prefix_sums_et_float.txt`

Logged modes:

- Mode 0: basic scan
- Mode 1: two-step scan (`devicek_step1` + `devicek_step2`)
- Mode 2: work-efficient two-step scan (`devicek_wef_step1` + `devicek_wef_step2`)

Representative runs:

| N | Host (ms) | Mode 0 (ms) | Mode 1 (ms) | Mode 2 (ms) |
| --- | ---: | ---: | ---: | ---: |
| `9999` | `0.007456` | `0.582848` | `0.246144` | `0.214112` |
| `22222` | `0.031424` | `2.368896` | `0.488928` | `0.487136` |
| `77777` | `0.165568` | `26.224960` | `7.171712` | `7.252832` |
| `99999` | `0.153696` | `43.040897` | `12.921248` | `11.573984` |

Supplemental console capture: `results/logs/prefix_sums_test_output.txt`

Summary:

- All logged runs verify correctly through the final-element checks in the raw data.
- Mode 1 and Mode 2 are clearly better than the basic scan once the input grows.
- In the committed float runs, the current CPU baseline is still faster than the GPU implementations, which suggests the scan kernels are still paying substantial launch and memory overhead relative to the problem sizes tested here.

## 4. Prefix Sums, Double

Source file: `2_prefix_sums.cu`

Raw timing file: `results/timings/2_prefix_sums_et_double.txt`

Representative runs:

| N | Host (ms) | Mode 0 (ms) | Mode 1 (ms) |
| --- | ---: | ---: | ---: |
| `9999` | `0.028384` | `4.447200` | `0.767168` |
| `22222` | `0.051808` | `21.393663` | `3.266880` |
| `99999` | `0.243904` | `311.328003` | `39.799072` |
| `999999` | `4.420032` | skipped | `42.449471` |
| `9999999` | `48.285023` | skipped | `3028.995361` |

Summary:

- The two-step scan remains much stronger than the basic scan in the logged double-precision runs.
- For the largest inputs, the raw data explicitly skips Mode 0 and keeps only the more practical multi-step path.
- As with the float runs, the checked-in results still favor the CPU baseline, so this folder reads as an optimization journey rather than a finished performance claim.

## 5. Work Present In Source But Missing Checked-In Result Logs

These kernels exist in the folder but do not yet have a committed markdown or raw-text benchmark summary:

- `3_histograms_2D_image.cu`
- `4_merge_sorted.cu`

Those are good next candidates if you want the folder to read as a complete sequence instead of a partial notebook.
