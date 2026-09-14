"""Upload completed-run weights and matching normalization/config snapshots to TOS."""
import argparse
import os
from pathlib import Path
import shutil


def stage_payload(run, weights):
    run, weights = Path(run).resolve(), Path(weights).resolve()
    if not weights.is_relative_to(run) or not weights.is_file():
        raise ValueError('Checkpoint must be a regular file inside this training run')
    step = int(weights.stem.removeprefix('step_'))
    payload = run / 'tos_payload' / weights.stem
    payload.mkdir(parents=True, exist_ok=True)
    target = payload / 'weights.pt'
    if target.exists():
        if not os.path.samefile(weights, target):
            raise ValueError('Existing payload refers to a different checkpoint')
    else:
        os.link(weights, target)
    stats = run / 'dataset_stats.json'
    if not stats.is_file():
        raise FileNotFoundError(f'Missing training normalization snapshot: {stats}')
    for source in [stats, *run.glob('*.yaml')]:
        target = payload / source.name
        if not target.exists():
            shutil.copy2(source, target)
    return payload, step


def make_store():
    from starwam.tools.checkpoint_tos.backend import TosCheckpointStore, TosUploadConfig
    return TosCheckpointStore(TosUploadConfig(
        endpoint=os.environ.get('TOS_ENDPOINT', 'https://tos-cn-beijing.ivolces.com'),
        region=os.environ.get('TOS_REGION', 'cn-beijing'),
        bucket=os.environ['TOS_BUCKET'], prefix=os.environ['TOS_PREFIX']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--doctor', action='store_true')
    args = parser.parse_args()
    store = make_store()
    if args.doctor:
        print(store.check_access())
        return
    if args.run is None:
        parser.error('--run is required')
    run = args.run.resolve()
    weights = sorted((run / 'checkpoints' / 'weights').glob('step_*.pt'))
    if not weights:
        raise FileNotFoundError(f'No completed checkpoints under {run}')
    for path in weights:
        payload, step = stage_payload(run, path)
        result = store.upload_checkpoint(payload, run_name=run.name, checkpoint_id=step,
            source_world_size=int(os.environ.get('NUM_GPUS', '8')),
            state_root=run / '.tos-upload-state' / str(step))
        print(result['destination'], flush=True)


if __name__ == '__main__':
    main()
