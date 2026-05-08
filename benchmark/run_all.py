from benchmark.benchmark_tool import Benchmark, _benchmark_registry


def do_init() -> None:
    import glob
    import importlib.util
    import os

    directory_path = os.path.dirname(__file__)

    for file_path in glob.glob(os.path.join(directory_path, "*.py")):
        if os.path.basename(file_path).startswith("__"):
            continue

        module_name = os.path.splitext(os.path.basename(file_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        globals()[module_name] = module


if __name__ == "__main__":
    do_init()
    print("Start benchmarking")
    for bench_type in _benchmark_registry.suites:
        bench = Benchmark(
            name=f"Benchmark of type '{bench_type}'", bench_type=bench_type
        )
        bench.run()
        bench.print_results()
