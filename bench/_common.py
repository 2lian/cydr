import importlib.metadata
import platform
import statistics
import timeit
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    count: int
    payload_size: int
    xcdrjit_fn: Callable[[], object]
    cyclonedds_fn: Callable[[], object]


@dataclass(frozen=True)
class Measurement:
    loops: int
    best_seconds: float
    median_seconds: float


def measure_runtime(fn: Callable[[], object], *, repeat: int, min_time: float) -> Measurement:
    timer = timeit.Timer(fn)
    loops = 1

    while True:
        elapsed = timer.timeit(number=loops)
        if elapsed >= min_time:
            break
        loops *= 2

    samples = timer.repeat(repeat=repeat, number=loops)
    per_call = [sample / loops for sample in samples]
    return Measurement(
        loops=loops,
        best_seconds=min(per_call),
        median_seconds=statistics.median(per_call),
    )


def benchmark_case(case: BenchmarkCase, *, repeat: int, min_time: float) -> dict[str, Measurement]:
    case.xcdrjit_fn()
    case.cyclonedds_fn()

    return {
        "xcdrjit_cached": measure_runtime(case.xcdrjit_fn, repeat=repeat, min_time=min_time),
        "cyclonedds_idl": measure_runtime(case.cyclonedds_fn, repeat=repeat, min_time=min_time),
    }


def format_time_microseconds(seconds: float) -> str:
    return f"{seconds * 1_000_000:.2f}"


def format_throughput_mib_per_second(payload_size: int, seconds: float) -> str:
    mib_per_second = payload_size / seconds / (1024 * 1024)
    return f"{mib_per_second:.2f}"


def print_environment(title: str, description_lines: list[str]) -> None:
    cyclonedds_version = importlib.metadata.version("cyclonedds-idl")
    xcdrjit_version = importlib.metadata.version("xcdrjit")

    print(title)
    for line in description_lines:
        print(line)
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"cyclonedds-idl: {cyclonedds_version}")
    print(f"xcdrjit: {xcdrjit_version}")
    print()


def print_results(
    cases: list[BenchmarkCase],
    measurements: dict[str, dict[str, Measurement]],
) -> None:
    print(
        f"{'Case':<12} {'Count':>8} {'Bytes':>10} {'Serializer':<20} "
        f"{'Loops':>10} {'Best us':>12} {'Median us':>12} {'MiB/s':>12} {'Speedup':>10}"
    )
    print("-" * 114)

    for case in cases:
        case_measurements = measurements[case.label]
        cyclonedds_best = case_measurements["cyclonedds_idl"].best_seconds

        for serializer_name in ("xcdrjit_cached", "cyclonedds_idl"):
            measurement = case_measurements[serializer_name]
            speedup = cyclonedds_best / measurement.best_seconds
            print(
                f"{case.label:<12} "
                f"{case.count:>8} "
                f"{case.payload_size:>10} "
                f"{serializer_name:<20} "
                f"{measurement.loops:>10} "
                f"{format_time_microseconds(measurement.best_seconds):>12} "
                f"{format_time_microseconds(measurement.median_seconds):>12} "
                f"{format_throughput_mib_per_second(case.payload_size, measurement.best_seconds):>12} "
                f"{speedup:>9.2f}x"
            )
