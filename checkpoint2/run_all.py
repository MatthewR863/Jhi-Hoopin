
import argparse
import os
import platform
import subprocess
import sys
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script)] + list(args)
    print("\n$ python3 " + " ".join([script] + list(args)), flush=True)
    code = subprocess.run(cmd, env=os.environ).returncode
    if code != 0:
        sys.exit("Step failed: %s (exit code %d)" % (script, code))


def write_environment(path):
    lines = ["python " + platform.python_version(), "platform " + platform.platform()]
    for pkg in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib", "nba_api"]:
        try:
            lines.append("%s %s" % (pkg, metadata.version(pkg)))
        except metadata.PackageNotFoundError:
            lines.append(pkg + " (not installed)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use synthetic data in demo_run/")
    parser.add_argument("--fetch", action="store_true", help="download real data with nba_api first")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    if args.demo:
        os.environ["JHI_WORKDIR"] = os.path.join(HERE, "demo_run")
    import paths  # imported after JHI_WORKDIR is set

    if args.demo:
        print("DEMO MODE: synthetic data only. These numbers are not real NBA results.")
        run("make_synthetic_data.py", "--force")
    elif args.fetch:
        run("fetch_data.py", "--seasons", *args.seasons)

    if not os.path.exists(paths.RAW_PATH):
        sys.exit("No raw data at %s\nRun with --fetch (needs internet), or copy your "
                 "team_game_logs.csv there." % paths.RAW_PATH)

    run("features.py")
    run("verify_features.py")
    run("train.py", "--model", "all")
    run("evaluate.py")
    run("inspect_results.py")
    write_environment(os.path.join(paths.RESULTS_DIR, "environment.txt"))
    print("\nDone. Outputs are in: " + paths.RESULTS_DIR)


if __name__ == "__main__":
    main()
