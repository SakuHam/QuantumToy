from __future__ import annotations

import argparse
import builtins
import contextlib
import curses
import fcntl
import json
import math
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AppConfig
from main import QuantumSimulationApp


_PRINT_LOCAL = threading.local()
_ORIGINAL_PRINT = builtins.print


def _thread_aware_print(*args, **kwargs):
    if getattr(_PRINT_LOCAL, "silent", False):
        return
    _ORIGINAL_PRINT(*args, **kwargs)


@contextlib.contextmanager
def silence_worker_prints():
    old = getattr(_PRINT_LOCAL, "silent", False)
    _PRINT_LOCAL.silent = True
    try:
        yield
    finally:
        _PRINT_LOCAL.silent = old


@dataclass
class WorkerStatus:
    worker_id: int
    state: str = "waiting"
    seed: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    message: str = ""
    completed: int = 0
    errors: int = 0


class JsonSeedStore:
    def __init__(self, path: Path, start_seed: int):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.thread_lock = threading.Lock()
        self.start_seed = int(start_seed)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def prepare_for_restart(self):
        """
        Normalize an existing results file before starting workers.

        Completed seeds are kept and therefore skipped by future reservations.
        In-progress reservations are process-local, so after a restart they are
        stale and should be released for recomputation.
        """
        with self.locked_data() as data:
            if data.get("running"):
                data["running"] = {}
            data["max"] = self._max_result_from_computed(data)
            data["next_seed"] = self._first_uncomputed_seed(
                data,
                start=int(data.get("next_seed", self.start_seed)),
            )
            data["updated_at"] = time.time()

    @contextlib.contextmanager
    def locked_data(self):
        with self.thread_lock:
            with self.lock_path.open("w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    data = self._read_unlocked()
                    changed = False
                    if not data:
                        data = self._fresh_data()
                        changed = True
                    if "computed" not in data:
                        data["computed"] = {}
                        changed = True
                    if "running" not in data:
                        data["running"] = {}
                        changed = True
                    if "next_seed" not in data:
                        data["next_seed"] = self._infer_next_seed(data)
                        changed = True
                    if "max" not in data:
                        data["max"] = self._max_result_from_computed(data)
                        changed = True
                    if data.get("max") is None and data.get("computed"):
                        data["max"] = self._max_result_from_computed(data)
                        changed = True
                    if changed:
                        self._write_unlocked(data)
                    yield data
                    self._write_unlocked(data)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def reserve_seed(self, worker_id: int) -> int:
        with self.locked_data() as data:
            seed = int(data.get("next_seed", self.start_seed))
            computed = data.setdefault("computed", {})
            running = data.setdefault("running", {})

            while str(seed) in computed or str(seed) in running:
                seed += 1

            data["next_seed"] = seed + 1
            running[str(seed)] = {
                "worker_id": int(worker_id),
                "reserved_at": time.time(),
            }
            return seed

    def record_result(self, seed: int, result: dict):
        with self.locked_data() as data:
            running = data.setdefault("running", {})
            running.pop(str(seed), None)

            computed = data.setdefault("computed", {})
            computed[str(seed)] = result

            current_max = data.get("max")
            if (
                current_max is None
                or float(result.get("separation", -math.inf))
                > float(current_max.get("separation", -math.inf))
            ):
                data["max"] = result

            data["updated_at"] = time.time()

    def record_error(self, seed: int, error: str):
        with self.locked_data() as data:
            running = data.setdefault("running", {})
            running.pop(str(seed), None)

            errors = data.setdefault("errors", {})
            errors[str(seed)] = {
                "seed": int(seed),
                "error": error,
                "time": time.time(),
            }
            data["updated_at"] = time.time()

    def snapshot(self) -> dict:
        with self.locked_data() as data:
            return json.loads(json.dumps(data))

    def _fresh_data(self) -> dict:
        now = time.time()
        return {
            "schema": 1,
            "created_at": now,
            "updated_at": now,
            "next_seed": self.start_seed,
            "computed": {},
            "running": {},
            "errors": {},
            "max": None,
        }

    def _read_unlocked(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            backup = self.path.with_suffix(self.path.suffix + f".bad-{int(time.time())}")
            self.path.replace(backup)
            return {}

    def _write_unlocked(self, data: dict):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(self.path)

    def _infer_next_seed(self, data: dict) -> int:
        seeds = [self.start_seed - 1]
        for section in ("computed", "running", "errors"):
            for seed_text in data.get(section, {}).keys():
                try:
                    seeds.append(int(seed_text))
                except Exception:
                    pass
        return max(seeds) + 1

    def _first_uncomputed_seed(self, data: dict, start: int) -> int:
        seed = max(int(start), self.start_seed)
        computed = data.get("computed", {})
        running = data.get("running", {})
        while str(seed) in computed or str(seed) in running:
            seed += 1
        return seed

    def _max_result_from_computed(self, data: dict) -> dict | None:
        best = None
        for result in data.get("computed", {}).values():
            if not isinstance(result, dict):
                continue
            if best is None:
                best = result
                continue
            if float(result.get("separation", -math.inf)) > float(best.get("separation", -math.inf)):
                best = result
        return best


def apply_search_config(cfg: AppConfig, seed: int, args: argparse.Namespace):
    cfg.CLICK_RNG_SEED = int(seed)
    cfg.SAVE_COMPLEX_STATE_FRAMES = True
    cfg.ENABLE_FLUX_BATCH_SAMPLER = False
    cfg.BREAK_ON_DETECTOR_CLICK = bool(args.break_on_detector_click)
    cfg.PRINT_ALIGNMENT_STATS = False
    cfg.PRINT_DIVERGENCE_STATS = False
    cfg.PRINT_BOHMIAN_STATS = False

    if hasattr(cfg, "POSTHOC_SAVE_GAMMA_LIKE"):
        cfg.POSTHOC_SAVE_GAMMA_LIKE = False

    if args.n_steps is not None:
        cfg.n_steps = int(args.n_steps)
    if args.dt is not None:
        cfg.dt = float(args.dt)
    if args.save_every is not None:
        cfg.save_every = int(args.save_every)
    if args.theory is not None:
        cfg.THEORY_NAME = str(args.theory)
    if args.detector is not None:
        cfg.DETECTOR_NAME = str(args.detector)


def run_seed_to_click(seed: int, args: argparse.Namespace) -> dict:
    with silence_worker_prints():
        cfg = AppConfig()
        apply_search_config(cfg, seed, args)

        app = QuantumSimulationApp(cfg)
        setup = app.build_setup()
        forward = app.run_forward(setup)
        click = app.resolve_click(setup, forward)

    x_a = click.x_click_a if click.x_click_a is not None else click.x_click
    y_a = click.y_click_a if click.y_click_a is not None else click.y_click
    x_b = click.x_click_b
    y_b = click.y_click_b

    if x_b is None or y_b is None:
        separation = 0.0
    else:
        separation = float(math.hypot(float(x_a) - float(x_b), float(y_a) - float(y_b)))

    return {
        "seed": int(seed),
        "separation": float(separation),
        "t_det": float(click.t_det),
        "idx_det": int(click.idx_det),
        "x_click": float(click.x_click),
        "y_click": float(click.y_click),
        "x_click_a": None if click.x_click_a is None else float(click.x_click_a),
        "y_click_a": None if click.y_click_a is None else float(click.y_click_a),
        "x_click_b": None if click.x_click_b is None else float(click.x_click_b),
        "y_click_b": None if click.y_click_b is None else float(click.y_click_b),
        "coincidence_channel": click.coincidence_channel,
        "coincidence_channel_probs": click.coincidence_channel_probs,
        "used_detector_click": bool(click.used_detector_click),
        "actual_last_step": int(forward.actual_last_step),
        "frames": int(len(forward.times)),
        "computed_at": time.time(),
    }


def worker_loop(
    worker_id: int,
    store: JsonSeedStore,
    args: argparse.Namespace,
    stop_event: threading.Event,
    statuses: list[WorkerStatus],
):
    status = statuses[worker_id]
    stagger_s = float(args.stagger_seconds) * float(worker_id)
    deadline = time.time() + stagger_s

    while not stop_event.is_set() and time.time() < deadline:
        status.state = "stagger"
        status.message = f"starts in {max(0.0, deadline - time.time()):.1f}s"
        time.sleep(min(0.25, max(0.0, deadline - time.time())))

    while not stop_event.is_set():
        seed = store.reserve_seed(worker_id)
        status.state = "running"
        status.seed = seed
        status.started_at = time.time()
        status.finished_at = None
        status.elapsed_s = None
        status.message = "forward -> click"

        try:
            result = run_seed_to_click(seed, args)
            elapsed = time.time() - float(status.started_at)
            result["elapsed_s"] = elapsed
            store.record_result(seed, result)
            status.completed += 1
            status.state = "done"
            status.finished_at = time.time()
            status.elapsed_s = elapsed
            status.message = f"sep={result['separation']:.6f}"
        except Exception:
            status.errors += 1
            err = traceback.format_exc(limit=8)
            store.record_error(seed, err)
            status.state = "error"
            status.finished_at = time.time()
            status.elapsed_s = (
                None if status.started_at is None else time.time() - float(status.started_at)
            )
            status.message = err.strip().splitlines()[-1][:120]

        time.sleep(0.05)

    status.state = "stopping"
    status.message = "clean stop after current seed"


def draw_dashboard(stdscr, store: JsonSeedStore, statuses: list[WorkerStatus], stop_event: threading.Event):
    curses.curs_set(0)
    stdscr.nodelay(True)

    while not stop_event.is_set():
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q"), ord("e"), ord("E"), 27):
            stop_event.set()
            break

        snapshot = store.snapshot()
        computed = snapshot.get("computed", {})
        running = snapshot.get("running", {})
        max_result = snapshot.get("max")

        stdscr.erase()
        stdscr.addstr(0, 0, "CLICK_RNG_SEED separation search  |  press q/e/Esc to stop now")
        stdscr.addstr(1, 0, f"json: {store.path}")
        stdscr.addstr(
            2,
            0,
            f"computed={len(computed)} running={len(running)} next_seed={snapshot.get('next_seed')}",
        )

        if max_result:
            stdscr.addstr(
                4,
                0,
                "max: "
                f"seed={max_result.get('seed')} "
                f"sep={float(max_result.get('separation', 0.0)):.6f} "
                f"A=({max_result.get('x_click_a')}, {max_result.get('y_click_a')}) "
                f"B=({max_result.get('x_click_b')}, {max_result.get('y_click_b')}) "
                f"channel={max_result.get('coincidence_channel')}",
            )
        else:
            stdscr.addstr(4, 0, "max: none yet")

        row = 6
        stdscr.addstr(row, 0, "workers:")
        row += 1
        for status in statuses:
            elapsed = ""
            if status.started_at is not None and status.state == "running":
                elapsed = f" elapsed={time.time() - status.started_at:.1f}s"
            elif status.elapsed_s is not None:
                elapsed = f" elapsed={status.elapsed_s:.1f}s"

            stdscr.addstr(
                row,
                0,
                (
                    f"[{status.worker_id}] {status.state:9s} "
                    f"seed={status.seed} done={status.completed} errors={status.errors}"
                    f"{elapsed}  {status.message}"
                )[: max(0, curses.COLS - 1)],
            )
            row += 1

        stdscr.refresh()
        time.sleep(0.2)


def print_dashboard(store: JsonSeedStore, statuses: list[WorkerStatus], stop_event: threading.Event):
    while not stop_event.is_set():
        snapshot = store.snapshot()
        max_result = snapshot.get("max")
        worker_bits = ", ".join(
            f"{s.worker_id}:{s.state}:seed={s.seed}:done={s.completed}" for s in statuses
        )
        if max_result:
            max_text = (
                f"max seed={max_result.get('seed')} "
                f"sep={float(max_result.get('separation', 0.0)):.6f}"
            )
        else:
            max_text = "max none"
        print(
            f"[SEARCH] computed={len(snapshot.get('computed', {}))} "
            f"running={len(snapshot.get('running', {}))} {max_text} | {worker_bits}",
            flush=True,
        )
        stop_event.wait(5.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search CLICK_RNG_SEED values for maximum A/B coincidence click separation."
    )
    parser.add_argument("--json", default="click_seed_separation_search.json", help="state/results JSON path")
    parser.add_argument("--workers", type=int, default=4, help="number of worker threads")
    parser.add_argument("--start-seed", type=int, default=0, help="first seed to reserve for a fresh JSON file")
    parser.add_argument("--stagger-seconds", type=float, default=60.0, help="delay between worker starts")
    parser.add_argument("--no-curses", action="store_true", help="use line logging instead of curses UI")
    parser.add_argument(
        "--graceful-stop",
        action="store_true",
        help="wait for active seed computations to finish before exiting",
    )
    parser.add_argument("--break-on-detector-click", action="store_true", help="stop forward run once detector clicks")
    parser.add_argument("--n-steps", type=int, default=None, help="override cfg.n_steps")
    parser.add_argument("--dt", type=float, default=None, help="override cfg.dt")
    parser.add_argument("--save-every", type=int, default=None, help="override cfg.save_every")
    parser.add_argument("--theory", default=None, help="override cfg.THEORY_NAME")
    parser.add_argument("--detector", default=None, help="override cfg.DETECTOR_NAME")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    builtins.print = _thread_aware_print

    workers = max(1, int(args.workers))
    store = JsonSeedStore(Path(args.json).resolve(), start_seed=int(args.start_seed))
    store.prepare_for_restart()
    stop_event = threading.Event()
    statuses = [WorkerStatus(worker_id=i) for i in range(workers)]

    def handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(i, store, args, stop_event, statuses),
            daemon=not bool(args.graceful_stop),
            name=f"click-seed-worker-{i}",
        )
        for i in range(workers)
    ]

    for t in threads:
        t.start()

    try:
        if args.no_curses or not sys.stdout.isatty():
            print_dashboard(store, statuses, stop_event)
        else:
            curses.wrapper(draw_dashboard, store, statuses, stop_event)
    finally:
        stop_event.set()
        if args.graceful_stop:
            for t in threads:
                t.join()

    snapshot = store.snapshot()
    max_result = snapshot.get("max")
    print("\nSearch stopped cleanly.")
    print(f"Results JSON: {store.path}")
    print(f"Computed seeds: {len(snapshot.get('computed', {}))}")
    if max_result:
        print(
            "Current maximum: "
            f"seed={max_result.get('seed')} "
            f"separation={float(max_result.get('separation', 0.0)):.6f} "
            f"A=({max_result.get('x_click_a')}, {max_result.get('y_click_a')}) "
            f"B=({max_result.get('x_click_b')}, {max_result.get('y_click_b')}) "
            f"channel={max_result.get('coincidence_channel')}"
        )
    else:
        print("Current maximum: none")

    if not args.graceful_stop:
        print(
            "Forced stop: active seed computations were abandoned. "
            "Any stale running entries will be cleared on restart."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
