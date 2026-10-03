"""Run the history Theory in a locked double-slit preset and export a visual lab.

Example: python src/quantumtoy/analysis/debug/run_history_instrument.py --shots 512
The HTML is offline. Its scenario menu selects computed runs; changing physics
parameters requires rerunning this command. No server or GUI is needed.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.spatial_effect_measurement import double_slit_effect_experiment, initial_spatial_state, spatial_effect_potential
from analysis.temporal_history_profile import TemporalHistoryEnvelope
from analysis.memory_banks import MemoryBankConfig
from analysis.spatial_detector_inference import DetectorResponse
from core.grid import build_grid
from core.simulation_types import PotentialSpec
from core.history_runner import simulate_history, export_history_runs
from theories.history_instrument import HistoryInstrumentTheory


def double_slit_setup(size=48):
    exp = double_slit_effect_experiment(nx=size, ny=size)
    grid = build_grid(exp.lx, exp.ly, exp.nx, exp.ny, 1)
    potential = spatial_effect_potential(exp)
    mask = np.zeros_like(potential, bool)
    screen = np.abs(grid.X-exp.detector_x) < exp.detector_width
    spec = PotentialSpec(potential, np.zeros_like(potential), screen, screen,
                         potential > exp.barrier_height/2, mask, mask)
    return exp, grid, spec, initial_spatial_state(exp).reshape(size, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo/history_instrument_theory.html'))
    parser.add_argument('--grid', type=int, default=48)
    parser.add_argument('--shots', type=int, default=512)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--strengths', type=float, nargs='+', default=[0., 1., 4.])
    parser.add_argument('--g', type=float, default=1., help='Reference sigma=0.2/g; not an independent measured calibration')
    parser.add_argument('--sigma', type=float, help='Direct width, replacing the reference g calibration')
    parser.add_argument('--alpha', type=float, default=.2/2.76)
    parser.add_argument('--clock-start', type=float, default=-2.)
    parser.add_argument('--front-time', type=float, help='Direct center, replacing clock_start+sigma/alpha')
    parser.add_argument('--absorption', type=float, default=4.)
    parser.add_argument('--detector-x', type=float, default=1.5)
    parser.add_argument('--detector-width', type=float, default=.3)
    parser.add_argument('--pointer-y', type=float, default=0.)
    parser.add_argument('--keep', type=float, default=.3)
    parser.add_argument('--fade', type=float, default=1.)
    parser.add_argument('--fade-power', type=float, default=1.5)
    parser.add_argument('--reference-copies', type=int, default=1)
    parser.add_argument('--delayed-copies', type=int, default=4)
    parser.add_argument('--reference-survival', type=float, default=.995)
    parser.add_argument('--reads', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=.2)
    parser.add_argument('--read-wait', type=float, default=0.)
    parser.add_argument('--read-efficiency', type=float, default=.9)
    parser.add_argument('--loss-mode', choices=['independent', 'shared'], default='independent')
    parser.add_argument('--efficiency', type=float, default=.9)
    parser.add_argument('--dark', type=float, default=.01)
    parser.add_argument('--blur', type=float, default=.5)
    parser.add_argument('--jitter', type=float, default=0.)
    args = parser.parse_args()
    if not np.isfinite(args.g) or args.g <= 0 or not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('g and alpha must be finite and positive')
    if args.sigma is not None and args.g != 1:
        parser.error('Use either --sigma or --g to set the clock width')
    if args.steps < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('steps and duration must be positive')
    sigma = .2/args.g if args.sigma is None else args.sigma
    center = args.clock_start+sigma/args.alpha if args.front_time is None else args.front_time
    exp, grid, potential, packet = double_slit_setup(args.grid)
    envelope = TemporalHistoryEnvelope(sigma, args.keep, args.fade, args.fade_power)
    memory = MemoryBankConfig(args.reference_copies, args.delayed_copies, args.reference_survival,
        args.loss_mode, args.reads, args.spacing, args.read_efficiency)
    response = DetectorResponse(efficiency=args.efficiency, dark_probability=args.dark,
        blur_sigma_bins=args.blur, timing_jitter=args.jitter)
    runs = []
    # A computed null is always present for comparison; do not change λ silently.
    for strength in dict.fromkeys([0., *args.strengths]):
        print(f'Computing λ={strength:g}, σ={sigma:g}, center={center:g}, shots={args.shots}', flush=True)
        theory = HistoryInstrumentTheory(grid, potential, sigma_t=sigma, front_time=center,
            selection_strength=strength, pointer_y=args.pointer_y, absorption_rate=args.absorption,
            detector_x=args.detector_x, detector_width=args.detector_width, y_bins=exp.y_bins,
            propagation_step=exp.propagation_step, rng_seed=args.seed)
        run = simulate_history(theory, packet, dt=args.duration/args.steps, steps=args.steps,
            trajectories=args.shots, save_every=args.save_every, envelope=envelope,
            memory=memory, response=response, read_wait=args.read_wait)
        run.parameters['clock_calibration'] = dict(mode='reference_0.2/g' if args.sigma is None else 'direct_sigma',
            g=args.g if args.sigma is None else None, alpha=args.alpha, clock_start=args.clock_start,
            center_mode='direct' if args.front_time is not None else 'start+sigma/alpha')
        run.parameters['geometry'] = 'locked_double_slit_periodic'
        runs.append(run)
        print(f'  Physical detections: {int(theory.detected.sum())}/{args.shots}; records: {len(run.records)}; '
              f'last recovered: {run.reads[-1]["recovered"]}', flush=True)
    print(f'Wrote {export_history_runs(runs, potential, args.output)}', flush=True)


if __name__ == '__main__':
    main()
