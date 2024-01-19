import torch
from benchmark.benchmark_tool import (
    Benchmark,
    BenchmarkSuite,
    SingleAudioProvider,
    SingleAudioProviderClasses,
    benchmark
)


@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=1, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=16, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=32, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=64, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=128, dtype='float32'),
    n_iters=1000
)
class AudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Audiomentations Gain (dtype={data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size
        )

        self.data_provider = data_provider

    def on_start(self):
        from audiomentations import Gain
        self.gain = Gain(min_gain_in_db=-10, max_gain_in_db=10, p=1.0)
        self.audio = self.data_provider.get()[0]

    def run_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.gain(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.gain(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float16),
    n_iters=1000
)
class FastAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Fast Audiomentations Gain ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )

        self.data_provider = data_provider

    def on_start(self):
        from fast_audiomentations.transforms.gain import Gain
        self.gain = Gain(min_gain_in_db=-10, max_gain_in_db=10, p=1.0)
        self.audio = self.data_provider.get()

    def run_iteration(self):
        self.gain(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.gain(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    'gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    'gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    'gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    'gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    'gain',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    n_iters=1000
)
class TorchAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Torch Audiomentations Gain ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )
        self.data_provider = data_provider

    def on_start(self):
        from torch_audiomentations import Gain
        self.gain = Gain(min_gain_in_db=-10, max_gain_in_db=10, p=1.0)
        self.audio = self.data_provider.get().unsqueeze(1)

    def run_iteration(self):
        self.gain(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.gain(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


if __name__ == '__main__':
    benchmark = Benchmark(name="Gain Benchmark", bench_type='gain')
    benchmark.run()
    benchmark.print_results()
