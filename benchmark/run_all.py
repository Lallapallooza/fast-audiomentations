from benchmark.benchmark_tool import Benchmark, _benchmark_registry
import time

def do_init():
    import os
    import glob
    import importlib.util

    # List of module names to be imported when 'from benchmark import *' is used
    __all__ = []

    # Path to the directory of this file
    directory_path = os.path.dirname(__file__)

    # Import all .py files in this directory
    for file_path in glob.glob(os.path.join(directory_path, '*.py')):
        # Skip __init__.py
        if os.path.basename(file_path).startswith('__'):
            continue

        module_name = os.path.splitext(os.path.basename(file_path))[0]
        __all__.append(module_name)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Add the module to globals so it's accessible as if 'from <module> import *'
        globals()[module_name] = module

if __name__ == '__main__':
    do_init()
    print('Start benchmarking')
    for bench_type in _benchmark_registry.suites.keys():
        benchmark = Benchmark(name=f"Benchmark of type \'{bench_type}\'", bench_type=bench_type)
        benchmark.run()
        benchmark.print_results()
