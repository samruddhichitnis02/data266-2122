
### Seeds and Hyperparameter Configuration

```text
SID4: 2122
SEED: 2122
Training seeds: 2122, 2123, 2124
HP_ID: 4
```

The baseline and modified configurations were:

| Model | Hidden layers | Learning rate | Epochs |
|---|---|---:|---:|
| Baseline | 64, 32 | 0.001 | 30 |
| Modified, HP_ID 4 | 64, 32 | 0.001 | 15 |

HP_ID 4 modifies the baseline by reducing the training duration from 30 epochs to 15 epochs. The hidden-layer architecture and learning rate remain unchanged.

### Test Accuracy Across Three Training Seeds

| Framework | Model | Seed 2122 | Seed 2123 | Seed 2124 | Mean test accuracy | Standard deviation |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | Baseline | 78.95% | 78.07% | 78.07% | 78.36% | 0.51 percentage points |
| PyTorch | Modified, HP_ID 4 | 76.32% | 77.19% | 77.19% | 76.90% | 0.51 percentage points |
| TensorFlow | Baseline | 78.07% | 77.19% | 77.19% | 77.49% | 0.51 percentage points |
| TensorFlow | Modified, HP_ID 4 | 74.56% | 73.68% | 76.32% | 74.85% | 1.34 percentage points |

The mean and standard deviation are calculated from the three test accuracy values for each model and framework.

### Part 2 Accuracy Analysis

The PyTorch baseline achieved the highest mean test accuracy at 78.36%. The PyTorch modified model achieved 76.90%, which is 1.46 percentage points lower than the PyTorch baseline.

The TensorFlow baseline achieved a mean test accuracy of 77.49%. The TensorFlow modified model achieved 74.85%, which is 2.64 percentage points lower than the TensorFlow baseline.

Therefore, the modified HP_ID 4 configuration performed worse than the baseline in both frameworks. The reduction from 30 to 15 epochs appears to have limited the amount of optimization performed before evaluation.

### Training and Validation Loss: PyTorch

The PyTorch baseline was trained for 30 epochs. Its training loss decreased from 0.6425 at epoch 1 to 0.4020 at epoch 30. Its validation loss decreased from 0.6034 at epoch 1 to a minimum of 0.4370 at epoch 14, then increased to 0.4491 by epoch 30.

The PyTorch modified model was trained for 15 epochs. Its training loss decreased from 0.6912 at epoch 1 to 0.4396 at epoch 15. Its validation loss decreased from 0.6479 at epoch 1 to a minimum of 0.4351 at epoch 12, then increased slightly to 0.4395 at epoch 15.

| PyTorch model | Initial training loss | Final training loss | Best validation loss | Best validation-loss epoch | Final validation loss |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6425 | 0.4020 | 0.4370 | 14 | 0.4491 |
| Modified, HP_ID 4 | 0.6912 | 0.4396 | 0.4351 | 12 | 0.4395 |

The PyTorch baseline shows mild overfitting after approximately epoch 14 because training loss continues to decrease while validation loss generally increases. The modified model reaches its lowest validation loss at epoch 12 and shows less late-training overfitting because training stops earlier.

#### PyTorch Loss Values by Epoch

| Epoch | Baseline training loss | Baseline validation loss | Modified training loss | Modified validation loss |
|---:|---:|---:|---:|---:|
| 1 | 0.6425 | 0.6034 | 0.6912 | 0.6479 |
| 2 | 0.5845 | 0.5469 | 0.6215 | 0.5828 |
| 3 | 0.5389 | 0.5053 | 0.5636 | 0.5337 |
| 4 | 0.5065 | 0.4782 | 0.5236 | 0.4915 |
| 5 | 0.4887 | 0.4587 | 0.4943 | 0.4646 |
| 6 | 0.4774 | 0.4454 | 0.4812 | 0.4497 |
| 7 | 0.4693 | 0.4440 | 0.4711 | 0.4442 |
| 8 | 0.4644 | 0.4401 | 0.4663 | 0.4417 |
| 9 | 0.4597 | 0.4407 | 0.4620 | 0.4352 |
| 10 | 0.4562 | 0.4372 | 0.4576 | 0.4401 |
| 11 | 0.4520 | 0.4402 | 0.4536 | 0.4362 |
| 12 | 0.4483 | 0.4420 | 0.4513 | 0.4351 |
| 13 | 0.4456 | 0.4416 | 0.4461 | 0.4395 |
| 14 | 0.4441 | 0.4370 | 0.4438 | 0.4390 |
| 15 | 0.4392 | 0.4417 | 0.4396 | 0.4395 |


### Training and Validation Loss: TensorFlow

The TensorFlow baseline was trained for 30 epochs. Its training loss decreased from 0.6459 at epoch 1 to 0.3834 at epoch 30. Its validation loss decreased from 0.5841 at epoch 1 to a minimum of 0.4410 at epoch 8, then increased to 0.4706 by epoch 30.

The TensorFlow modified model was trained for 15 epochs. Its training loss decreased from 0.6206 at epoch 1 to 0.4287 at epoch 15. Its validation loss decreased from 0.5759 at epoch 1 to a minimum of 0.4422 at epoch 11, then increased to 0.4456 by epoch 15.

| TensorFlow model | Initial training loss | Final training loss | Best validation loss | Best validation-loss epoch | Final validation loss |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6459 | 0.3834 | 0.4410 | 8 | 0.4706 |
| Modified, HP_ID 4 | 0.6206 | 0.4287 | 0.4422 | 11 | 0.4456 |

The TensorFlow baseline shows clearer overfitting after approximately epoch 8 because training loss continues to fall while validation loss rises. The modified model also shows mild overfitting after epoch 11, but stopping at 15 epochs limits the increase in validation loss.

#### TensorFlow Loss Values by Epoch

| Epoch | Baseline training loss | Baseline validation loss | Modified training loss | Modified validation loss |
|---:|---:|---:|---:|---:|
| 1 | 0.6459 | 0.5841 | 0.6206 | 0.5759 |
| 2 | 0.5719 | 0.5206 | 0.5619 | 0.5294 |
| 3 | 0.5265 | 0.4818 | 0.5258 | 0.4989 |
| 4 | 0.4971 | 0.4591 | 0.5010 | 0.4788 |
| 5 | 0.4782 | 0.4475 | 0.4845 | 0.4641 |
| 6 | 0.4656 | 0.4426 | 0.4730 | 0.4539 |
| 7 | 0.4568 | 0.4413 | 0.4649 | 0.4481 |
| 8 | 0.4501 | 0.4410 | 0.4585 | 0.4450 |
| 9 | 0.4448 | 0.4413 | 0.4530 | 0.4433 |
| 10 | 0.4402 | 0.4423 | 0.4481 | 0.4424 |
| 11 | 0.4361 | 0.4434 | 0.4436 | 0.4422 |
| 12 | 0.4323 | 0.4446 | 0.4397 | 0.4429 |
| 13 | 0.4288 | 0.4459 | 0.4359 | 0.4439 |
| 14 | 0.4255 | 0.4475 | 0.4322 | 0.4445 |
| 15 | 0.4223 | 0.4487 | 0.4287 | 0.4456 |

### Test Accuracy for the SEED Run

For the required `SEED = 2122` run, the notebook reports the following three-seed table values:

| Framework | Model | Test accuracy |
|---|---|---:|
| PyTorch | Baseline | 78.95% |
| PyTorch | Modified, HP_ID 4 | 76.32% |
| TensorFlow | Baseline | 78.07% |
| TensorFlow | Modified, HP_ID 4 | 74.56% |

### Loss Curve Figures

The notebook contains the combined loss plots for the baseline and modified models. The numerical evidence used for the analysis above is taken from the recorded epoch outputs and the minimum validation-loss summary.

## Part 3: CUDA Performance Measurements

### CUDA Environment

| Metric | Value | Notes |
|---|---|---|
| GPU | NVIDIA Tesla T4 | Reported by `nvidia-smi` in the notebook |
| Driver version | 580.82.07 | Reported by `nvidia-smi` |
| CUDA version | 13.0 | Reported by `nvidia-smi` |
| Compute capability | 7.5 | Reported by the profiler |
| Kernel | `matrixMultiply` | Matrix multiplication kernel |
| Matrix operation | `C = A x B` | Single-precision floating-point matrices |

### CUDA Run Timing

The uploaded CUDA notebook does not contain the separate `Run started`, `Run finished`, or total notebook duration output. Those values should not be invented. The benchmark measurements below are the actual recorded CUDA results.

### CUDA Benchmark Results

| Matrix size | CPU time (ms) | GPU kernel time (ms) | Transfer time, H2D plus D2H (ms) | GPU end-to-end time (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 256 x 256 | 20.107 | 21.066 | 0.293 | 21.360 | 0.941x |
| 1024 x 1024 | 3274.540 | 9.168 | 3.095 | 12.263 | 267.034x |
| 4096 x 4096 | 817454.761 | 418.377 | 47.463 | 465.840 | 1754.797x |

Speedup is calculated as:

```text
Speedup = CPU time / GPU end-to-end time
```

The transfer column is the combined host-to-device and device-to-host transfer time reported by the benchmark program.

### CUDA Performance Analysis

For the 256 x 256 matrix, the GPU end-to-end time was 21.360 ms compared with a CPU time of 20.107 ms. The GPU was slightly slower, with a speedup of 0.941x. For this small workload, kernel launch and transfer overhead offset the benefit of parallel execution.

For the 1024 x 1024 matrix, the GPU end-to-end time was 12.263 ms compared with a CPU time of 3274.540 ms. The GPU achieved a speedup of 267.034x.

For the 4096 x 4096 matrix, the GPU end-to-end time was 465.840 ms compared with a CPU time of 817454.761 ms. The GPU achieved the largest speedup of 1754.797x.

The results show that CUDA becomes substantially more effective as the matrix size increases. The computation grows rapidly with matrix size, while the transfer overhead remains comparatively small.

### Profiler Evidence

The notebook includes an Nsight Compute profiler run for the `matrixMultiply` kernel using a small 2 x 2 grid and 2 x 2 x 1 blocks. The profiler reported:

| Profiler metric | Value |
|---|---:|
| Kernel duration | 3.74 microseconds |
| Memory throughput | 0.80% |
| Compute throughput | 0.10% |
| SM busy | 0.93% |
| L1/TEX hit rate | 55.56% |
| L2 hit rate | 88.35% |
| Average active threads per warp | 4 |

The profiler identified that the small kernel grid underutilizes the GPU. It also reported nonoptimal global load and store memory access patterns and low warp utilization. These observations apply to the small profiler demonstration run. The benchmark timing table above is the authoritative result for matrix sizes 256, 1024, and 4096.

## Overall Conclusion

In Part 2, the baseline model performed better than the HP_ID 4 modified model in both PyTorch and TensorFlow. The loss curves show that the longer baseline training runs introduce overfitting, but the modified 15-epoch configuration reduced test accuracy more than it improved generalization.

In Part 3, CUDA was slightly slower for the 256 x 256 matrix but substantially faster for the 1024 x 1024 and 4096 x 4096 matrices. The largest measured speedup was 1754.797x for the 4096 x 4096 matrix.
