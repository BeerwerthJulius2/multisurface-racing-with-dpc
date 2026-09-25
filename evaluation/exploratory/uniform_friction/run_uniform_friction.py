"""Run the pooled-data DPC controller on a track with uniform friction.

Control experiment for the implied-friction claim: the track geometry and the
reference speed profile are identical across runs, only the (constant) friction
differs. Any structure left in the implied friction therefore cannot come from
the spatial arrangement of the friction zones.

Usage:
    python3 evaluation/exploratory/uniform_friction/run_uniform_friction.py
    python3 evaluation/exploratory/uniform_friction/run_uniform_friction.py --frictions 0.5 0.8 1.1 --laps 5

Writes one log per friction value: log_dpc_eval_<map>_uniform_<mu>
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_one(map_name, friction, laps, hydra="", tag="", timeout=300, velocity=None):
    log_suffix = f"_uniform_{friction}{tag}"
    env = dict(
        os.environ,
        DPC_MAP_NAME=map_name,
        DPC_USE_DYN_FRICTION="0",
        DPC_CONSTANT_FRICTION=str(friction),
        DPC_NUMBER_OF_LAPS=str(laps),
        DPC_LOG_SUFFIX=log_suffix,
        DPC_HYDRA_OVERRIDES=hydra,
        DPC_HEADLESS="1",
        **({"DPC_VELOCITY_MULTIPLIER": str(velocity)} if velocity else {}),
    )
    print(f"=== mu = {friction}, {laps} laps on {map_name} ===", flush=True)
    try:
        subprocess.run([sys.executable, "DPC.py"], cwd=REPO_ROOT, env=env,
                       check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A controller that spins out never reaches the lap count; no log is written.
        print(f"!! timed out after {timeout}s, no log for mu={friction}", flush=True)
        return None
    return f"log_dpc_eval_{map_name}{log_suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="l_shape")
    parser.add_argument("--frictions", nargs="+", type=float, default=[0.5, 1.1],
                        help="constant friction values, one run each")
    parser.add_argument("--laps", type=int, default=11)
    parser.add_argument("--hydra", default="", help="extra hydra overrides, e.g. controller.lambda_g=10.0")
    parser.add_argument("--tag", default="", help="log suffix tag, to keep sweeps apart")
    parser.add_argument("--timeout", type=int, default=300, help="wall-clock cap per run [s]")
    parser.add_argument("--velocity", type=float, default=None, help="velocity profile multiplier")
    args = parser.parse_args()

    logs = [run_one(args.map, mu, args.laps, args.hydra, args.tag, args.timeout, args.velocity)
            for mu in args.frictions]
    logs = [log for log in logs if log]

    print("\nWrote:")
    for log in logs:
        print(f"  {log}")
    print("\nNext: python3 evaluation/exploratory/uniform_friction/analyze_uniform_friction.py "
          f"--map {args.map} --frictions {' '.join(str(mu) for mu in args.frictions)}")


if __name__ == "__main__":
    main()
