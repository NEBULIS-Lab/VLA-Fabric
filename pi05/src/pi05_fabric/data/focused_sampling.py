"""Deterministic paired-window sampling; command anchors are not contact labels."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ('grasp_lift', 'handover', 'placement_release')


def command_events(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 20 or not np.isfinite(actions).all():
        raise ValueError('expected finite paired EEF actions [T,20]')
    events = {}
    for col, role in ((9, 'donor'), (19, 'receiver')):
        delta = np.diff((actions[:, col] > .5).astype(int))
        close, opened = np.flatnonzero(delta == -1)+1, np.flatnonzero(delta == 1)+1
        if len(close) != 1 or len(opened) != 1:
            raise ValueError(f'ambiguous {role} gripper transitions: {close}, {opened}')
        events[f'{role}_close'], events[f'{role}_open'] = int(close[0]), int(opened[0])
    if not (events['donor_close'] < events['receiver_close'] < events['donor_open'] < events['receiver_open']):
        raise ValueError('expert donor/receiver event order not satisfied')
    return events


def event_windows(events, length, execution_horizon=25):
    anchors = {
        'grasp_lift': [events['donor_close'], min(events['donor_close']+12, events['receiver_close']-1)],
        'handover': [events['receiver_close'], events['donor_open']],
        'placement_release': [events['receiver_open'], min(events['receiver_open']+6, length-1)],
    }
    # E25 must contain only recorded commands. H50 may retain the original
    # terminal-repeat convention, but we do not oversample synthetic execution.
    return {name: sorted({start for a in group
                          for start in range(max(0, a-execution_horizon+1), min(a, length-execution_horizon)+1)})
            for name, group in anchors.items()}


@dataclass(frozen=True)
class Draw:
    episode_id: int
    frame: int
    group: str
    event_class: str | None


class FocusedSampler:
    def __init__(self, index_path, dataset_root, *, microbatch, accumulation):
        if microbatch != 2 or accumulation != 2:
            raise ValueError('focused continuation requires microbatch2 and accumulation2')
        raw = Path(index_path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.index = json.loads(raw)
        manifest_raw = (Path(dataset_root)/'manifest.json').read_bytes()
        if hashlib.sha256(manifest_raw).hexdigest() != self.index['dataset_manifest_sha256']:
            raise ValueError('focus index dataset manifest hash mismatch')
        if (self.index['schema_version'], self.index['action_horizon'], self.index['execution_horizon']) != (1,50,25):
            raise ValueError('focus index protocol mismatch')
        manifest = json.loads(manifest_raw)
        lengths = {int(e['id']): int(e['length']) for e in manifest['episodes']}
        self.episodes = self.index['episodes']
        if {e['id']:e['length'] for e in self.episodes} != lengths:
            raise ValueError('focus index episode coverage mismatch')
        for e in self.episodes:
            for name in CLASSES:
                frames=e['windows'][name]
                if not frames or frames != sorted(set(frames)) or not all(0<=t<e['length'] for t in frames):
                    raise ValueError('invalid focus windows')
        self.uniform = tuple((int(e['id']), t) for e in manifest['episodes'] for t in range(e['length']))
        self.microbatch = microbatch

    def draw(self, *, seed, step, sample_index):
        if min(seed, step, sample_index) < 0:
            raise ValueError('sampling counters must be nonnegative')
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, sample_index]))
        if (sample_index // self.microbatch) % 2 == 0:
            ep, frame = self.uniform[int(rng.integers(len(self.uniform)))]
            return Draw(ep, frame, 'uniform', None)
        name = CLASSES[int(rng.integers(len(CLASSES)))]
        ep = self.episodes[int(rng.integers(len(self.episodes)))]
        frames = ep['windows'][name]
        return Draw(ep['id'], frames[int(rng.integers(len(frames)))], 'focus', name)
