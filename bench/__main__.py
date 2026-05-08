"""``python -m bench`` entry point. Forwards to ``bench.run.main``."""

from bench.run import main

if __name__ == "__main__":
    raise SystemExit(main())
