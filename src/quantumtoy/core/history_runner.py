"""Forward-only history simulations, event archives, and delayed memory reads."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import numpy as np

from analysis.memory_banks import MemoryBankConfig
from analysis.spatial_detector_inference import DetectorResponse, detector_blur_matrix
from analysis.temporal_history_profile import TemporalHistoryEnvelope


@dataclass
class HistoryRun:
    times: np.ndarray
    densities: np.ndarray
    example_densities: np.ndarray
    diagnostics: list
    events: list
    records: list
    reads: list
    initial_state: np.ndarray
    final_state: np.ndarray
    parameters: dict


def digitize_history_events(events, *, shots, edges, y_bins, response, rng):
    """Window-level electronics matching read_history_detector.

    At most one accepted record per preparation. Dark records are assigned
    after acquisition only where there is no accepted genuine record. This
    is the existing finite-window model, not a causal Poisson dark-count process.
    Both physical birth time and jittered timestamp are retained.
    """
    blur = detector_blur_matrix(y_bins, response.blur_sigma_bins)
    records = {}
    for event in events:
        if event['type'] != 'detection' or rng.random() >= response.efficiency:
            continue
        timestamp = event['time'] + rng.normal(0, response.timing_jitter)
        if not edges[0] <= timestamp < edges[-1]:
            continue
        records[event['trajectory']] = dict(trajectory=event['trajectory'],
            created=event['time'], timestamp=float(timestamp), dark=False,
            y_bin=int(rng.choice(y_bins, p=blur[:, event['y_bin']])) )
    for shot in range(shots):
        if shot not in records and rng.random() < response.dark_probability:
            k = int(rng.integers(len(edges)-1))
            born = float((edges[k]+edges[k+1])/2)
            records[shot] = dict(trajectory=shot, created=born, timestamp=born,
                                 dark=True, y_bin=int(rng.integers(y_bins)))
    return sorted(records.values(), key=lambda r: (r['created'], r['trajectory']))


def sample_memory_reads(records, envelope, memory, *, acquisition_end, read_wait, rng):
    """One lifetime per copy; successful reads go to a persistent classical log."""
    if not np.isfinite(read_wait) or read_wait < 0:
        raise ValueError('read_wait must be nonnegative')
    count = len(records)
    born = np.array([r['created'] for r in records])
    if np.any(born > acquisition_end):
        raise ValueError('Records cannot be created after acquisition')
    reference = rng.random((count, memory.reference_copies)) < memory.reference_survival
    ref_any = reference.any(axis=1)
    cols = 1 if memory.loss_mode == 'shared' and memory.delayed_copies else memory.delayed_copies
    lifetimes = envelope.retention_time + envelope.fade_time*rng.weibull(envelope.fade_power, (count, cols))
    if memory.loss_mode == 'shared':
        lifetimes = np.repeat(lifetimes, memory.delayed_copies, axis=1)
    logged = np.zeros(count, bool)
    reads = []
    for k in range(memory.read_count):
        time = acquisition_end+read_wait+k*memory.read_spacing
        alive = lifetimes >= (time-born)[:, None]
        success = alive & (rng.random(alive.shape) < memory.read_efficiency)
        latest = success.any(axis=1)
        logged |= latest
        reads.append(dict(time=float(time), last=int(latest.sum()), logged=int(logged.sum()),
            reference=int(ref_any.sum()), recovered=int((ref_any | logged).sum()),
            copies=int(reference.sum()+success.sum()),
            last_ids=[records[i]['trajectory'] for i in np.flatnonzero(latest)],
            logged_ids=[records[i]['trajectory'] for i in np.flatnonzero(logged)],
            reference_ids=[records[i]['trajectory'] for i in np.flatnonzero(ref_any)]))
    return reads


def replay_examples(theory, initial_state, indices, *, dt, steps, save_every):
    """Reconstruct selected actual trajectories from their recorded jumps.

    No random draws, no replacement histories. Choosing illustrative examples
    after acquisition affects only this panel, never the ensemble statistics.
    """
    state = np.broadcast_to(initial_state, (len(indices),)+initial_state.shape).copy()
    lookup = {shot: j for j, shot in enumerate(indices)}
    schedule = {}
    for event in theory.events:
        if event['trajectory'] in lookup:
            step = int(round(event['time']/dt-.5))
            schedule.setdefault(step, []).append(event)
    alive = np.ones(len(indices), bool)
    area = theory.grid.dx*theory.grid.dy
    no_click = np.exp(-(theory._detector_rate+theory._escape_rate)*dt/2)
    frames = []
    for k in range(steps+1):
        if k % save_every == 0 or k == steps:
            frames.append((np.abs(state)**2).astype(np.float32))
        if k == steps:
            break
        state = theory._unitary(state, dt/2)
        for event in schedule.get(k, []):
            if event['type'] == 'selection':
                j = lookup[event['trajectory']]
                state[j] *= theory._upper if event['half'] == 'upper' else ~theory._upper
                state[j] /= np.sqrt(np.sum(np.abs(state[j])**2)*area)
        state = theory._unitary(state, dt/2)
        state *= no_click
        for event in schedule.get(k, []):
            if event['type'] in ('detection', 'escape'):
                j = lookup[event['trajectory']]
                state[j] = 0
                alive[j] = False
        if np.any(alive):
            state[alive] /= np.sqrt(np.sum(np.abs(state[alive])**2, axis=(1, 2))*area)[:, None, None]
    return np.array(frames)


def simulate_history(theory, initial_state, *, dt, steps, trajectories=128, save_every=1,
                     envelope=TemporalHistoryEnvelope(sigma_t=.2, retention_time=.3, fade_time=1),
                     memory=MemoryBankConfig(), response=DetectorResponse(timing_jitter=0), read_wait=0):
    for value, name in [(steps, 'steps'), (trajectories, 'trajectories'), (save_every, 'save_every')]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if trajectories*theory.grid.Nx*theory.grid.Ny*16 > 512*1024**2:
        raise ValueError('Trajectory state exceeds 512 MiB; reduce HISTORY_TRAJECTORIES or the grid')
    state = theory.initialize_state(np.broadcast_to(initial_state, (trajectories,)+initial_state.shape))
    normalized_initial = state[0].copy()
    times, densities, diagnostics = [], [], []
    aux = dict(time=0., events=[], surviving=1., detected=0., escaped=0., selected=0.)
    for k in range(steps+1):
        if k % save_every == 0 or k == steps:
            times.append(k*dt)
            densities.append(theory.density(state).astype(np.float32))
            diagnostics.append({name: value for name, value in aux.items() if name != 'events'})
        if k < steps:
            result = theory.step_forward(state, dt)
            state, aux = result.state, result.aux
    end = steps*dt
    electronic_seed, memory_seed = np.random.SeedSequence([theory.rng_seed, 9841]).spawn(2)
    records = digitize_history_events(theory.events, shots=trajectories,
        edges=np.arange(steps+1)*dt, y_bins=theory.y_bins, response=response,
        rng=np.random.default_rng(electronic_seed))
    reads = sample_memory_reads(records, envelope, memory, acquisition_end=end,
        read_wait=read_wait, rng=np.random.default_rng(memory_seed))
    candidates = []
    for mask in [theory.detected, theory.alive & (theory.selected_half == 1),
                 theory.alive & (theory.selected_half == 0), theory.alive & (theory.selected_half == -1)]:
        matches = np.flatnonzero(mask)
        if matches.size:
            candidates.append(int(matches[0]))
    indices = list(dict.fromkeys([*candidates, *range(min(4, trajectories))]))[:min(4, trajectories)]
    examples = replay_examples(theory, normalized_initial, indices, dt=dt, steps=steps, save_every=save_every)
    parameters = dict(theory='history_instrument', dt=dt, steps=steps, trajectories=trajectories,
        save_every=save_every, sigma_t=theory.sigma_t, front_time=theory.front_time,
        selection_strength=theory.selection_strength, pointer_y=theory.pointer_y,
        absorption_rate=theory.absorption_rate, detector_x=theory.detector_x,
        detector_width=theory.detector_width, y_bins=theory.y_bins,
        propagation_step=theory.propagation_step, seed=theory.rng_seed,
        mass=theory.m_mass, hbar=theory.hbar, grid=[theory.grid.Ny, theory.grid.Nx],
        extent=[float(theory.grid.x[0]),float(theory.grid.x[-1]+theory.grid.dx),
                float(theory.grid.y[0]),float(theory.grid.y[-1]+theory.grid.dy)],
        envelope=asdict(envelope), memory=asdict(memory), response=asdict(response), read_wait=read_wait,
        example_indices=indices, example_selection='illustrative_detected_and_surviving_histories')
    return HistoryRun(np.array(times), np.array(densities), examples, diagnostics,
                      list(theory.events), records, reads, normalized_initial, state, parameters)


def export_history_runs(runs, potential, output):
    """Float32 frame archive plus an offline, interactive HTML viewer."""
    from viz.history_instrument_view import build_history_html
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = {'potential': potential.V_real, 'absorber': potential.W}
    report = []
    for i, run in enumerate(runs):
        archive.update({f'initial_state_{i}': run.initial_state, f'times_{i}': run.times, f'density_{i}': run.densities,
                        f'examples_{i}': run.example_densities})
        report.append(dict(parameters=run.parameters, diagnostics=run.diagnostics,
                           events=run.events, records=run.records, reads=run.reads))
    np.savez_compressed(output.with_suffix('.npz'), **archive)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    output.write_text(build_history_html(runs, potential), encoding='utf-8')
    return output


def run_history_setup(setup):
    """App entry point: use the configured grid/packet, bypass the old click pipeline."""
    from core.packets import PacketFactory
    cfg, theory = setup.cfg, setup.theory
    initial = PacketFactory.build_initial_packet(cfg, setup.grid).psi0
    envelope = TemporalHistoryEnvelope(sigma_t=cfg.HISTORY_SIGMA_T,
        retention_time=cfg.HISTORY_KEEP_TIME, fade_time=cfg.HISTORY_FADE_TIME, fade_power=cfg.HISTORY_FADE_POWER)
    memory = MemoryBankConfig(cfg.HISTORY_REFERENCE_COPIES, cfg.HISTORY_DELAYED_COPIES,
        cfg.HISTORY_REFERENCE_SURVIVAL, cfg.HISTORY_MEMORY_LOSS_MODE,
        cfg.HISTORY_READ_COUNT, cfg.HISTORY_READ_SPACING, cfg.HISTORY_READ_EFFICIENCY)
    response = DetectorResponse(efficiency=cfg.HISTORY_DETECTION_EFFICIENCY,
        dark_probability=cfg.HISTORY_DARK_PROBABILITY, blur_sigma_bins=cfg.HISTORY_BLUR_SIGMA,
        timing_jitter=cfg.HISTORY_TIMING_JITTER)
    runs = [simulate_history(replace(theory, selection_strength=strength), initial,
        dt=cfg.dt, steps=cfg.n_steps, trajectories=cfg.HISTORY_TRAJECTORIES,
        save_every=cfg.save_every, envelope=envelope, memory=memory, response=response,
        read_wait=cfg.HISTORY_READ_WAIT) for strength in dict.fromkeys([0., theory.selection_strength])]
    path = export_history_runs(runs, setup.potential, cfg.HISTORY_OUTPUT)
    print(f'History instrument: {path} (plus JSON events and NPZ frames)')
    return runs
