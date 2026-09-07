"""Compatibility entrypoint; general implementation is factor_sdk_baseline.py."""

import argparse
from factor_sdk_baseline import main

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    main(p.parse_args().bundle)
