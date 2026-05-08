import enum
from collections import defaultdict
from collections.abc import Callable
from time import time
from typing import Any

import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from benchmark.benchmark_data import PATH_TO_SINGLE_AUDIO


class BenchmarkRegistry:
    def __init__(self) -> None:
        self.suites: dict[
            str, list[tuple[type, tuple[Any, ...], dict[str, Any]]]
        ] = defaultdict(list)

    def register(
        self,
        suite_class: type,
        bench_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for this_suite_class, this_args, this_kwargs in self.suites[
            bench_type
        ]:
            if (
                this_suite_class.__name__ == suite_class.__name__
                and set(this_args) == set(args)
                and this_kwargs == kwargs
            ):
                return
        self.suites[bench_type].append((suite_class, args, kwargs))

    def get_suites(self, bench_type: str) -> list["BenchmarkSuite"]:
        return [
            suite_class(*args, **kwargs)
            for suite_class, args, kwargs in self.suites[bench_type]
        ]


_benchmark_registry = BenchmarkRegistry()


def benchmark(
    bench_type: str, *args: Any, **kwargs: Any
) -> Callable[[type], type]:
    def decorator(suite_class: type) -> type:
        _benchmark_registry.register(suite_class, bench_type, *args, **kwargs)
        return suite_class

    return decorator


class SingleAudioProviderClasses(enum.Enum):
    TORCH_GPU = 1
    TORCH_CPU = 2
    NUMPY = 3


class SingleAudioProvider:
    def __init__(
        self,
        cls: SingleAudioProviderClasses,
        dtype: str | torch.dtype = "float32",
        batch_size: int = 1,
    ) -> None:
        samples, sr = sf.read(PATH_TO_SINGLE_AUDIO, dtype="float32")
        self.samples = samples
        self.sr = sr
        self.cls = cls
        self.dtype = dtype
        self.batch_size = batch_size

    def get(self) -> Any:
        torch_dtype = (
            self.dtype if isinstance(self.dtype, torch.dtype) else None
        )
        if self.cls == SingleAudioProviderClasses.TORCH_GPU:
            return torch.tensor(
                self.samples, device="cuda", dtype=torch_dtype
            ).repeat(self.batch_size, 1)
        if self.cls == SingleAudioProviderClasses.TORCH_CPU:
            return torch.tensor(
                self.samples, device="cpu", dtype=torch_dtype
            ).repeat(self.batch_size, 1)
        if self.cls == SingleAudioProviderClasses.NUMPY:
            return np.tile(self.samples, self.batch_size).reshape(
                self.batch_size, -1
            )
        raise ValueError(f"Unknown class {self.cls}")


class BenchmarkSuite:
    def __init__(
        self,
        name: str,
        warmup_iterations: int = 10,
        iterations: int = 1000,
        samples_per_iter: int = 1,
        is_gpu_timer_required: bool = False,
    ) -> None:
        self.__name = name
        self.__warmup_iterations = warmup_iterations
        self.__iterations = iterations
        self.__samples_per_iter = samples_per_iter
        self.__is_gpu_timer_required = is_gpu_timer_required

    def iterations(self) -> int:
        return self.__iterations

    def warmup_iterations(self) -> int:
        return self.__warmup_iterations

    def samples_per_iter(self) -> int:
        return self.__samples_per_iter

    def name(self) -> str:
        return self.__name

    def run(self, progress: Progress, task: TaskID) -> list[float]:
        elapsed: list[float] = []
        for _ in range(self.__iterations):
            torch.cuda.synchronize()
            if self.__is_gpu_timer_required:
                start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
                end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
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

            progress.update(task, advance=100.0 / self.__iterations)
        return elapsed

    def run_iteration(self) -> None:
        raise NotImplementedError

    def on_start(self) -> None:
        raise NotImplementedError

    def on_end(self) -> None:
        raise NotImplementedError

    def warmup(self, progress: Progress, task: TaskID) -> None:
        for _ in range(self.__warmup_iterations):
            self.run_warmup_iteration()
            progress.update(task, advance=100.0 / self.__warmup_iterations)

    def run_warmup_iteration(self) -> None:
        raise NotImplementedError


class Benchmark:
    def __init__(
        self,
        name: str,
        bench_type: str,
        auto_collect_suites: bool = True,
    ) -> None:
        self.suites: list[BenchmarkSuite] = []
        self.name = name
        self.bench_type = bench_type
        self.timings_on_start: dict[str, float] = {}
        self.timings_on_end: dict[str, float] = {}
        self.timings_on_warmup: dict[str, float] = {}
        self.timings_on_run: dict[str, float] = {}

        if auto_collect_suites:
            self.collect_suites()

    def collect_suites(self) -> None:
        for suite in _benchmark_registry.get_suites(self.bench_type):
            self.add_suite(suite)

    def add_suite(self, suite: BenchmarkSuite) -> None:
        self.suites.append(suite)

    def run(self) -> None:
        with Progress() as progress:
            for suite in self.suites:
                task = progress.add_task(
                    f"[green]Running {suite.name()}...", total=100
                )
                self.run_suite_with_progress(suite, progress, task)

    @staticmethod
    def seconds_to_microseconds(seconds: float) -> float:
        return seconds * 1000000

    def run_suite_with_progress(
        self,
        suite: BenchmarkSuite,
        progress: Progress,
        task: TaskID,
    ) -> None:
        start = time()
        suite.on_start()
        end = time()
        self.timings_on_start[suite.name()] = end - start
        progress.update(task, advance=25)

        warmup_task = progress.add_task("Warmup...", total=100)
        suite.warmup(progress, warmup_task)
        end = time()
        self.timings_on_warmup[suite.name()] = end - start
        progress.update(task, advance=25)

        run_task = progress.add_task("Running...", total=100)
        elapsed = suite.run(progress, run_task)
        self.timings_on_run[suite.name()] = sum(elapsed)
        progress.update(task, advance=25)

        start = time()
        suite.on_end()
        end = time()
        self.timings_on_end[suite.name()] = end - start
        progress.update(task, advance=25)

    def print_results(self) -> None:
        console = Console()

        warmup_times = {
            suite.name(): self.timings_on_warmup[suite.name()]
            for suite in self.suites
        }
        warmup_times_average = {
            suite.name(): self.timings_on_warmup[suite.name()]
            / suite.warmup_iterations()
            / suite.samples_per_iter()
            for suite in self.suites
        }

        run_times = {
            suite.name(): self.timings_on_run[suite.name()]
            for suite in self.suites
        }
        run_times_average = {
            suite.name(): self.timings_on_run[suite.name()]
            / suite.iterations()
            / suite.samples_per_iter()
            for suite in self.suites
        }

        fastest_run_time = min(run_times_average.values())
        relative_slowdown = {
            name: t / fastest_run_time for name, t in run_times_average.items()
        }

        def get_color_and_emoji(slowdown: float) -> tuple[str, str]:
            if slowdown <= 1.1:
                return "green", "rocket"
            if slowdown <= 2:
                return "yellow", "lightning"
            if slowdown <= 5:
                return "orange", "turtle"
            return "red", "snail"

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Suite", style="dim", width=75)
        table.add_column("Warmup Time Total (mcs)", justify="right")
        table.add_column("Warmup Time Per Sample (mcs)", justify="right")
        table.add_column("Run Time Total (mcs)", justify="right")
        table.add_column("Run Time Per Sample (mcs)", justify="right")
        table.add_column("Relative Slowdown", justify="right")
        table.add_column("Percentage Slower (%)", justify="right")

        sorted_suites = sorted(run_times_average.items(), key=lambda x: x[1])

        for suite_name, _ in sorted_suites:
            run_time = self.seconds_to_microseconds(run_times[suite_name])
            warmup_time = self.seconds_to_microseconds(
                warmup_times[suite_name]
            )
            warmup_time_per_sample = self.seconds_to_microseconds(
                warmup_times_average[suite_name]
            )
            run_time_per_sample = self.seconds_to_microseconds(
                run_times_average[suite_name]
            )
            slowdown = relative_slowdown[suite_name]
            percent_slower = (slowdown - 1) * 100
            color, label = get_color_and_emoji(slowdown)

            table.add_row(
                f"[{color}]{label} {suite_name}[/]",
                f"{warmup_time:.4f}mcs",
                f"{warmup_time_per_sample:.4f}mcs",
                f"{run_time:.4f}mcs",
                f"{run_time_per_sample:.4f}mcs",
                f"{slowdown:.4f}x",
                f"{percent_slower:.4f}%",
            )

        console.print(table)
