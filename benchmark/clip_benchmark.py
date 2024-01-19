import torch
from benchmark.benchmark_tool import (
    Benchmark,
    BenchmarkSuite,
    SingleAudioProvider,
    SingleAudioProviderClasses,
    benchmark
)


@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=1, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=16, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=32, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=64, dtype='float32'),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=128, dtype='float32'),
    n_iters=1000
)
class AudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Audiomentations Clip (dtype={data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size
        )

        self.data_provider = data_provider

    def on_start(self):
        from audiomentations import Clip
        self.clip = Clip(a_min=-0.6, a_max=0.6, p=1.0)
        self.audio = self.data_provider.get()[0]

    def run_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.clip(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.clip(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float16),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    n_iters=1000
)
@benchmark(
    bench_type='clip',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float16),
    n_iters=1000
)
class FastAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, n_iters):
        super().__init__(
            name=f"Fast Audiomentations Clip ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )

        self.data_provider = data_provider

    def on_start(self):
        from fast_audiomentations import Clip
        self.clip = Clip(min=-0.6, max=0.6, p=1.0)
        self.audio = self.data_provider.get()

    def run_iteration(self):
        self.clip(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.clip(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


if __name__ == '__main__':
    benchmark = Benchmark(name="Clip Benchmark", bench_type="clip")
    benchmark.run()
    benchmark.print_results()
