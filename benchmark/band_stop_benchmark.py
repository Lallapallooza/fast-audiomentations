import torch
from benchmark.benchmark_tool import (
    Benchmark,
    BenchmarkSuite,
    SingleAudioProvider,
    SingleAudioProviderClasses,
    benchmark
)

@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=1, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=16, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=32, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=64, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=128, dtype='float32'),
    n_iters=1000
)
class AudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Audiomentations Band Stop Filter (dtype={data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size
        )

        self.data_provider = data_provider

    def on_start(self):
        from audiomentations import BandStopFilter
        self.band_stop_filter = BandStopFilter(min_center_freq=100, max_center_freq=10000, p=1.0)
        self.audio = self.data_provider.get()[0]

    def run_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float16),
    n_iters=1000
)
class FastAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Fast Audiomentations Band Stop Filter ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )

        self.data_provider = data_provider

    def on_start(self):
        from fast_audiomentations import BandStopFilter
        self.band_stop_filter = BandStopFilter(min_center_freq=100, max_center_freq=10000, p=1.0)
        self.audio = self.data_provider.get()

    def run_iteration(self):
        self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='band_stop_filter',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    n_iters=1000
)
class TorchAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Torch Audiomentations Band Stop Filter ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )
        self.data_provider = data_provider

    def on_start(self):
        from torch_audiomentations import BandStopFilter
        self.band_stop_filter = BandStopFilter(min_center_frequency=100, max_center_frequency=10000, p=1.0)
        self.audio = self.data_provider.get().unsqueeze(1)

    def run_iteration(self):
        self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.band_stop_filter(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass

if __name__ == '__main__':
    benchmark = Benchmark(name="Band Stop Filter Benchmark", bench_type='band_stop_filter')
    benchmark.run()
    benchmark.print_results()
