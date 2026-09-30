"""Validate and export the dynamical history/detector model for the demo."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.history_dynamics import (
    HistoryDynamicsConfig, build_history_dynamics_basis,
    evaluate_history_dynamics, read_history_detector,
)
from analysis.temporal_history_profile import TemporalHistoryEnvelope
from analysis.spatial_effect_measurement import double_slit_effect_experiment
from analysis.spatial_detector_inference import DetectorResponse, detector_blur_matrix


def summary(basis, envelope, *, center=.76, strength=1):
    run = evaluate_history_dynamics(basis, envelope, front_time=center, strength=strength)
    law = read_history_detector(basis, run, envelope)
    return dict(
        physical_click=run.detected_probability,
        surviving_particle=run.surviving_probability,
        saved_click=float(law.joint.sum()), erased_click=law.erased_record,
        no_record=law.no_record,
        selected=run.selected_probability, selection_memory=run.accessible_selection,
        normalization_error=float(abs(law.probabilities.sum()-1)),
    ), run, law


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('history_dynamics_study.json'))
    args = parser.parse_args()
    start = time.perf_counter()
    experiment = double_slit_effect_experiment(nx=48, ny=48)
    config = HistoryDynamicsConfig()
    envelope = TemporalHistoryEnvelope(sigma_t=.2, retention_time=.3, fade_time=1, fade_power=1.5)
    basis = build_history_dynamics_basis(experiment, config)
    cases = {}
    results = {}
    for name, env, strength in [
        ('quantum_null', envelope, 0), ('history', envelope, 1),
        ('strong_selection', envelope, 4),
        ('fast_fading', replace(envelope, fade_time=.1), 1),
        ('slow_fading', replace(envelope, fade_time=100), 1),
    ]:
        cases[name], results[name], _ = summary(basis, env, strength=strength)
    reference = results['history']
    null = results['quantum_null']
    cases['history']['complete_click_tv_vs_null'] = float(.5*(
        np.abs(reference.ideal_joint-null.ideal_joint).sum()
        + abs(reference.surviving_probability-null.surviving_probability)))
    cases['history']['surviving_density_l1_vs_null'] = float(np.abs(reference.density-null.density).sum())
    convergence = {}
    for name, grid, steps in [('dt_half', 48, 200), ('dt_quarter', 48, 400),
                              ('grid_64', 64, 100), ('grid_80', 80, 100)]:
        refined_basis = build_history_dynamics_basis(
            double_slit_effect_experiment(nx=grid, ny=grid), replace(config, steps=steps))
        refined = evaluate_history_dynamics(refined_basis, envelope, front_time=.76)
        joint = refined.ideal_joint.reshape(config.steps, steps//config.steps, experiment.y_bins).sum(axis=1)
        convergence[name] = float(.5*(np.abs(joint-reference.ideal_joint).sum()
            + abs(refined.surviving_probability-reference.surviving_probability)))
        print(name, convergence[name], flush=True)
        del refined_basis

    # The small display raster is a sum of cell probabilities, never a resized
    # amplitude. Clicking probabilities remain on the original y-bin grid.
    densities = np.sum(np.abs(basis.final_kets)**2, axis=1)
    display = densities.reshape(config.steps+1, 24, 2, 24, 2).sum(axis=(2, 4))
    response = DetectorResponse(timing_jitter=0)
    export = {
        'status': 'conditional_open_system_candidate',
        'assumptions': [
            'Single irreversible upper/lower half-plane projective measurement.',
            'Gaussian selection time conditioned on no pre-preparation selection.',
            'Independent clock, q=1-exp(-lambda); no record-driven Hamiltonian feedback.',
            'Absorbing detector with rate Gamma and per-record classical erasure.',
            'Periodic box, finite lattice, split-step evolution and midpoint event quadrature.',
        ],
        'experiment': asdict(experiment), 'config': asdict(config),
        'envelope': asdict(envelope), 'front_time': .76,
        'cases': cases, 'convergence_complete_tv': convergence,
        'elapsed_s': time.perf_counter()-start,
        'demo': {
            'edges': basis.edges.tolist(), 'times': basis.click_times.tolist(),
            'clicks': np.round(basis.clicks, 12).tolist(),
            'densities': np.round(display.reshape(config.steps+1, -1), 12).tolist(),
            'densityShape': [24, 24], 'yBins': experiment.y_bins,
            'duration': config.duration, 'absorptionRate': config.absorption_rate,
            'efficiency': response.efficiency, 'darkProbability': response.dark_probability,
            'blur': detector_blur_matrix(experiment.y_bins, response.blur_sigma_bins).tolist(),
            'timingJitter': 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=None, allow_nan=False)+'\n')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for name in ['quantum_null', 'history', 'strong_selection']:
        run = results[name]
        ax[0].plot(basis.click_times, run.ideal_joint.sum(axis=1)/(.02), label=name)
        ax[1].plot(run.ideal_joint.sum(axis=0), label=name)
    ax[0].set(xlabel='Detection time / T0', ylabel='First-click density per preparation')
    ax[1].set(xlabel='Detector y bin', ylabel='Unconditional click probability')
    names = ['fast_fading', 'history', 'slow_fading']
    bottom = np.zeros(3)
    for key in ['saved_click', 'erased_click', 'no_record']:
        values = [cases[n][key] for n in names]
        ax[2].bar(names, values, bottom=bottom, label=key)
        bottom += values
    ax[2].set(ylabel='Complete record probability', ylim=(0, 1))
    for axis in ax:
        axis.legend(fontsize=8)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps({'cases': cases, 'convergence': convergence, 'seconds': export['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
