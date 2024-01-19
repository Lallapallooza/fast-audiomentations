# fast-audiomentations
`fast-audiomentations` is a Python library that leverages GPU acceleration for efficient audio augmentation, 
suitable for high-throughput audio analysis and machine learning applications.

## Key Features

- Performance Showcase: Highlights the advantages of using GPU for audio processing tasks.
- Diverse Audio Augmentations: Offers a variety of transformations including noise addition, filtering, and gain adjustments, optimized for GPU performance.
- User-Friendly API: Designed for ease of use, enabling quick integration into audio processing pipelines.
- Triton-based GPU code: Triton enables maximized GPU utilization. Although real-world examples of Triton for straightforward kernels are limited, this library serves as an invaluable practical study resource.

## Installation
Currently, there is no ready-to-use package on PyPI, so you need to:
```bash
git clone https://github.com/Lallapallooza/fast-audiomentations
cd fast-audiomentations
python -m pip install -r requirements.txt
python -m examples.test
```
Assuming "SUCCESS" is printed, the library is now ready to use. 
Examples can be found in the `examples/` and `benchmark/` directories.

## Usage 
Here's an example of how to apply background noise addition to an audio sample 
using `fast-audiomentations`:

```python
from fast_audiomentations import AddBackgroundNoise
import torch

batch_size = 128

dataloader = AddBackgroundNoise.get_dali_dataloader(
    'some/path/to/noises/in/dali/format',
    buffer_size=batch_size,
    n_workers=4
)
add_background_noise = AddBackgroundNoise(
    noises_dataloader=dataloader,
    min_snr=-10,
    max_snr=10,
    buffer_size=batch_size,
    p=1.0
)

sample_rate = 48000
audio = load_my_audio_batch() # load as torch.Tensor 
audio_lens = torch.tensor([audio.shape[0] for i in range(batch_size)], device='cuda')

# torch.Tensor
audios_with_noise = add_background_noise(
    samples=audio, 
    samples_lens=audio_lens, 
    sample_rate=sample_rate
)
```

## Benchmarking
To validate the hypothesis that GPU can accelerate audio augmentation processing, 
benchmarks were conducted in comparison with two libraries: 
[audiomentations](https://github.com/iver56/audiomentations)
and 
[torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations).

Details about the specific batch sizes, augmentation types, and parameters can be found in the code. 
See the `benchmark/` directory.

For my configuration:
- NVIDIA GeForce RTX 3090 Ti
- 64GB RAM
- 12th Gen Intel(R) Core(TM) i9-12900KF
- Samsung 980 PRO

The results are available in the `benchmark_results_local/` directory.
To run benchmarks, you need to:
1. Change the parameters in  `benchmark/benchmark_data.py`.
2. Run the command: ```python -m benchmark.run_all```.

Observations show that for stateless operations (no IO required), 
the performance increase is significant (dozens of times). 
However, for augmentations like adding background noise, 
the performance benefit is less significant due to IO/PCIe overhead. 
Even utilizing DALI does not change the situation much, as the GPU is heavily underutilized. 
Nonetheless, it's observed that the NVMe device is not fully utilized, 
suggesting there might be some code inefficiency or a limitation in Python's IO performance.

## Future work

- Add more augmentations (RIR convolve, SpecAug, etc.). We welcome requests for additional augmentations and will endeavor to implement them promptly.
- Rewrite the IO process for stateful cases (RIR, add_background_noise) to fully utilize IO capabilities. 
- Conduct benchmarks with network devices for stateful scenarios to demonstrate that asynchronous data reading may reduce latency.
- Support XLA backend
- Add wrappers like `Compose`, `OneOf`, etc...

