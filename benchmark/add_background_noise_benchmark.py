import torch
from benchmark.benchmark_tool import (
    Benchmark,
    BenchmarkSuite,
    SingleAudioProvider,
    SingleAudioProviderClasses,
    benchmark
)
from benchmark.benchmark_data import PATH_TO_NOISES

@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=1, dtype='float32'),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=16, dtype='float32'),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=32, dtype='float32'),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=64, dtype='float32'),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.NUMPY, batch_size=128, dtype='float32'),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
class AudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, path_to_sounds: str, n_iters):
        super().__init__(
            name=f"Audiomentations Add Background Noise (dtype={data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size
        )

        self.data_provider = data_provider
        self.path_to_sounds = path_to_sounds

    def on_start(self):
        from audiomentations import AddBackgroundNoise
        self.add_background_noise = AddBackgroundNoise(
            sounds_path=self.path_to_sounds,
            min_snr_db=-10,
            max_snr_in_db=10,
            p=1.0
        )
        self.audio = self.data_provider.get()[0]

    def run_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.add_background_noise(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        for i in range(self.data_provider.batch_size):
            self.add_background_noise(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float16),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float16),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float16),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float16),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float16),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
class FastAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, path_to_sounds: str, n_iters):
        super().__init__(
            name=f"Fast Audiomentations Add Background Noise ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )

        self.data_provider = data_provider
        self.path_to_sounds = path_to_sounds

    def on_start(self):
        from fast_audiomentations import AddBackgroundNoise

        dataloader = AddBackgroundNoise.get_dali_dataloader(
            self.path_to_sounds,
            buffer_size=self.data_provider.batch_size,
            n_workers=4
        )
        self.add_background_noise = AddBackgroundNoise(
            noises_dataloader=dataloader,
            min_snr=-10,
            max_snr=10,
            buffer_size=self.data_provider.batch_size,
            p=1.0
        )
        self.audio = self.data_provider.get()
        self.audio_lens = torch.tensor([self.audio.shape[0] for i in range(self.data_provider.batch_size)], device='cuda')


    def run_iteration(self):
        self.add_background_noise(samples=self.audio, samples_lens=self.audio_lens, sample_rate=44100)

    def run_warmup_iteration(self):
        self.add_background_noise(samples=self.audio, samples_lens=self.audio_lens, sample_rate=44100)

    def on_end(self):
        pass


@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=1, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=16, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=32, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=64, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
@benchmark(
    bench_type='add_background_noise',
    data_provider=SingleAudioProvider(SingleAudioProviderClasses.TORCH_GPU, batch_size=128, dtype=torch.float32),
    path_to_sounds=PATH_TO_NOISES,
    n_iters=300
)
class TorchAudiomentationsBenchmarkSuite(BenchmarkSuite):
    def __init__(self, data_provider: SingleAudioProvider, path_to_sounds: str, n_iters):
        super().__init__(
            name=f"Torch Audiomentations Add Background Noise ({data_provider.dtype}) (batch_size={data_provider.batch_size})",
            warmup_iterations=10,
            iterations=n_iters,
            samples_per_iter=data_provider.batch_size,
            is_gpu_timer_required=True
        )
        self.data_provider = data_provider
        self.path_to_sounds = path_to_sounds

    def on_start(self):
        from torch_audiomentations import AddBackgroundNoise
        self.add_background_noise = AddBackgroundNoise(
            background_paths=self.path_to_sounds,
            min_snr_in_db=-10,
            max_snr_in_db=10,
            p=1.0
        )
        self.audio = self.data_provider.get().unsqueeze(1)

    def run_iteration(self):
        self.add_background_noise(samples=self.audio, sample_rate=44100)

    def run_warmup_iteration(self):
        self.add_background_noise(samples=self.audio, sample_rate=44100)

    def on_end(self):
        pass

if __name__ == '__main__':
    benchmark = Benchmark(name="Add Background Noise Benchmark", bench_type='add_background_noise')
    benchmark.run()
    benchmark.print_results()
