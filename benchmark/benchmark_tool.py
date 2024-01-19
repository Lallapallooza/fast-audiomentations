from time import time, sleep
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from collections import defaultdict
import torch
from benchmark.benchmark_data import PATH_TO_SINGLE_AUDIO
import soundfile as sf
import enum
import numpy as np

class BenchmarkRegistry:
    def __init__(self):
        self.suites = defaultdict(list)

    def register(self, suite_class, bench_type, *args, **kwargs):
        for (this_suite_class, this_args, this_kwargs) in self.suites[bench_type]:
            if (this_suite_class.__name__ == suite_class.__name__
                    and set(this_args) == set(args)
                    and this_kwargs == kwargs):
                return
        self.suites[bench_type].append((suite_class, args, kwargs))

    def get_suites(self, bench_type):
        return [suite_class(*args, **kwargs) for suite_class, args, kwargs in self.suites[bench_type]]

_benchmark_registry = BenchmarkRegistry()

def benchmark(bench_type, *args, **kwargs):
    def decorator(suite_class):
        _benchmark_registry.register(suite_class, bench_type, *args, **kwargs)
        return suite_class
    return decorator

class SingleAudioProviderClasses(enum.Enum):
    TORCH_GPU = 1
    TORCH_CPU = 2
    NUMPY = 3

class SingleAudioProvider:
    def __init__(self, cls: SingleAudioProviderClasses, dtype: str | torch.dtype='float32', batch_size=1):
        samples, sr = sf.read(PATH_TO_SINGLE_AUDIO, dtype='float32')
        self.samples = samples
        self.sr = sr
        self.cls = cls
        self.dtype = dtype
        self.batch_size = batch_size

    def get(self):
        if self.cls == SingleAudioProviderClasses.TORCH_GPU:
            return torch.tensor(self.samples, device='cuda', dtype=self.dtype).repeat(self.batch_size, 1)
        elif self.cls == SingleAudioProviderClasses.TORCH_CPU:
            return torch.tensor(self.samples, device='cpu', dtype=self.dtype).repeat(self.batch_size, 1)
        elif self.cls == SingleAudioProviderClasses.NUMPY:
            return np.tile(self.samples, self.batch_size).reshape(self.batch_size, -1)
        else:
            raise ValueError(f"Unknown class {self.cls}")

class BenchmarkSuite:
    def __init__(self,
                 name: str,
                 warmup_iterations: int = 10,
                 iterations: int = 1000,
                 samples_per_iter: int = 1,
                 is_gpu_timer_required: bool = False):
        self.__name = name
        self.__warmup_iterations = warmup_iterations
        self.__iterations = iterations
        self.__samples_per_iter = samples_per_iter
        self.__is_gpu_timer_required = is_gpu_timer_required

    def iterations(self):
        return self.__iterations

    def warmup_iterations(self):
        return self.__warmup_iterations

    def samples_per_iter(self):
        return self.__samples_per_iter

    def name(self):
        return self.__name

    def run(self, progress, task):
        elapsed = []
        for i in range(self.__iterations):

            torch.cuda.synchronize()
            if self.__is_gpu_timer_required:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start = time()

            self.run_iteration()
            if self.__is_gpu_timer_required:
                end_event.record()
                end_event.synchronize()
                elapsed.append(start_event.elapsed_time(end_event) / 1000)
            else:
                end = time()
                elapsed.append(end - start)

            progress.update(task, advance=100. / self.__iterations)
        return elapsed

    def run_iteration(self):
        raise NotImplementedError

    def on_start(self):
        raise NotImplementedError

    def on_end(self):
        raise NotImplementedError

    def warmup(self, progress, task):
        for i in range(self.__warmup_iterations):
            self.run_warmup_iteration()
            progress.update(task, advance=100. / self.__warmup_iterations)

    def run_warmup_iteration(self):
        raise NotImplementedError



class Benchmark:
    def __init__(self, name: str, bench_type: str, auto_collect_suites=True):
        self.suites = []
        self.name = name
        self.bench_type = bench_type
        self.timings_on_start = {}
        self.timings_on_end = {}
        self.timings_on_warmup = {}
        self.timings_on_run = {}

        if auto_collect_suites:
            self.collect_suites()

    def collect_suites(self):
        global _benchmark_registry
        for suite in _benchmark_registry.get_suites(self.bench_type):
            self.add_suite(suite)

    def add_suite(self, suite: BenchmarkSuite):
        self.suites.append(suite)

    def run(self):
        with Progress() as progress:
            for suite in self.suites:
                # Create a task for each suite
                task = progress.add_task(f"[green]Running {suite.name()}...", total=100)

                # Run the suite with progress updates
                self.run_suite_with_progress(suite, progress, task)

    @staticmethod
    def seconds_to_microseconds(seconds):
        return seconds * 1000000

    def run_suite_with_progress(self, suite, progress, task):
        # Start the suite
        start = time()
        suite.on_start()
        end = time()
        self.timings_on_start[suite.name()] = end - start
        progress.update(task, advance=25)

        # Warmup
        warmup_task = progress.add_task(f"Warmup...", total=100)
        suite.warmup(progress, warmup_task)
        end = time()
        self.timings_on_warmup[suite.name()] = end - start
        progress.update(task, advance=25)

        # Run
        run_task = progress.add_task(f"Running...", total=100)
        elapsed = suite.run(progress, run_task)
        self.timings_on_run[suite.name()] = sum(elapsed)
        progress.update(task, advance=25)

        # print(suite.name())
        # for i in range(len(elapsed)):
        #     print(f"{i}: {self.seconds_to_microseconds(elapsed[i])}")

        # End
        start = time()
        suite.on_end()
        end = time()
        self.timings_on_end[suite.name()] = end - start
        progress.update(task, advance=25)

    def print_results(self):
        console = Console()

        # Calculate warmup and run times
        warmup_times = {suite.name(): self.timings_on_warmup[suite.name()] for suite in self.suites}
        warmup_times_average = {suite.name(): self.timings_on_warmup[suite.name()] /
                                              suite.warmup_iterations() /
                                              suite.samples_per_iter() for suite in self.suites}

        run_times = {suite.name(): self.timings_on_run[suite.name()] for suite in self.suites}
        run_times_average = {suite.name(): self.timings_on_run[suite.name()] /
                                           suite.iterations() /
                                           suite.samples_per_iter() for suite in self.suites}

        # Calculate relative slowdown based on run times
        fastest_run_time = min(run_times_average.values())
        relative_slowdown = {suite: time / fastest_run_time for suite, time in run_times_average.items()}

        # Gradient colors and emojis for performance
        def get_color_and_emoji(slowdown):
            if slowdown <= 1.1:  # Close to fastest
                return "green", "🚀"
            elif slowdown <= 2:  # Moderately slower
                return "yellow", "⚡"
            elif slowdown <= 5:  # Slower
                return "orange", "🐢"
            else:  # Much slower
                return "red", "🐌"

        # Create a table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Suite", style="dim", width=75)
        table.add_column("Warmup Time Total (mcs)", justify="right")
        table.add_column("Warmup Time Per Sample (mcs)", justify="right")
        table.add_column("Run Time Total (mcs)", justify="right")
        table.add_column("Run Time Per Sample (mcs)", justify="right")
        table.add_column("Relative Slowdown", justify="right")
        table.add_column("Percentage Slower (%)", justify="right")

        # Sort by run times
        sorted_suites = sorted(run_times_average.items(), key=lambda x: x[1])

        # Add rows to the table
        for suite, _ in sorted_suites:
            run_time = self.seconds_to_microseconds(run_times[suite])
            warmup_time = self.seconds_to_microseconds(warmup_times[suite])
            warmup_time_per_sample = self.seconds_to_microseconds(warmup_times_average[suite])
            run_time_per_sample = self.seconds_to_microseconds(run_times_average[suite])
            slowdown = relative_slowdown[suite]
            percent_slower = (slowdown - 1) * 100
            color, emoji = get_color_and_emoji(slowdown)

            table.add_row(
                f"[{color}]{emoji} {suite}[/]",
                      f"{warmup_time:.4f}mcs", f"{warmup_time_per_sample:.4f}mcs",
                      f"{run_time:.4f}mcs", f"{run_time_per_sample:.4f}mcs",
                      f"{slowdown:.4f}x", f"{percent_slower:.4f}%"
            )

        console.print(table)




