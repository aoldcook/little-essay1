from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


TRAINER_MODULES = {
    'plain': 'context_aware_encoder_model.train_context_aware_encoder',
    'mntp': 'context_aware_encoder_model.train_context_aware_encoder_with_mntp',
}


def resolve_optional_path(raw_value: str) -> Optional[Path]:
    value = raw_value.strip()
    return Path(value) if value else None


def resolve_input_file(explicit_path: Optional[Path], split_dir: Optional[Path], default_name: str, label: str) -> Path:
    if explicit_path is not None:
        candidate = explicit_path
    elif split_dir is not None:
        candidate = split_dir / default_name
    else:
        raise ValueError(f'{label} is required when split_dir is not provided')

    if not candidate.exists():
        raise FileNotFoundError(f'{label} not found: {candidate}')
    return candidate


def load_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_stage_command(
    trainer_type: str,
    python_executable: str,
    train_file: Path,
    dev_file: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    temperature: float,
    seed: int,
    mlm_probability: float,
    lambda_mntp: float,
) -> List[str]:
    module_name = TRAINER_MODULES[trainer_type]
    command = [
        python_executable,
        '-X',
        'utf8',
        '-m',
        module_name,
        '--train_file',
        str(train_file),
        '--dev_file',
        str(dev_file),
        '--output_dir',
        str(output_dir),
        '--model_name',
        model_name,
        '--epochs',
        str(epochs),
        '--batch_size',
        str(batch_size),
        '--lr',
        str(lr),
        '--max_length',
        str(max_length),
        '--temperature',
        str(temperature),
        '--seed',
        str(seed),
    ]
    if trainer_type == 'mntp':
        command.extend(
            [
                '--mlm_probability',
                str(mlm_probability),
                '--lambda_mntp',
                str(lambda_mntp),
            ]
        )
    return command


def run_stage(
    stage_name: str,
    command: List[str],
    output_dir: Path,
    project_root: Path,
) -> Dict:
    print(f'[{stage_name}] command: {subprocess.list2cmdline(command)}')
    started_at = time.time()
    subprocess.run(command, cwd=project_root, check=True)
    elapsed = time.time() - started_at

    record = {
        'stage': stage_name,
        'output_dir': str(output_dir),
        'elapsed_seconds': elapsed,
        'best_metrics': load_json_if_exists(output_dir / 'best_metrics.json'),
        'training_history_file': str(output_dir / 'training_history.json'),
        'last_checkpoint_dir': str(output_dir / 'last'),
        'best_checkpoint_dir': str(output_dir),
    }
    print(f'[{stage_name}] finished in {elapsed:.1f}s')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run a two-stage curriculum for the context-aware sentence encoder: train on train_gold first, then continue on full train.'
    )
    parser.add_argument('--trainer_type', choices=sorted(TRAINER_MODULES.keys()), default='plain')
    parser.add_argument('--split_dir', type=str, default='')
    parser.add_argument('--train_gold_file', type=str, default='')
    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen3-Embedding-8B')
    parser.add_argument('--python_executable', type=str, default=sys.executable)
    parser.add_argument('--stage1_epochs', type=int, default=3)
    parser.add_argument('--stage2_epochs', type=int, default=2)
    parser.add_argument('--stage1_batch_size', type=int, default=4)
    parser.add_argument('--stage2_batch_size', type=int, default=4)
    parser.add_argument('--stage1_lr', type=float, default=2e-5)
    parser.add_argument('--stage2_lr', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--lambda_mntp', type=float, default=0.3)
    parser.add_argument('--stage1_dir_name', type=str, default='stage1_gold')
    parser.add_argument('--stage2_dir_name', type=str, default='stage2_full')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    split_dir = resolve_optional_path(args.split_dir)
    train_gold_file = resolve_input_file(
        resolve_optional_path(args.train_gold_file), split_dir, 'train_gold.jsonl', 'train_gold_file'
    )
    train_file = resolve_input_file(resolve_optional_path(args.train_file), split_dir, 'train.jsonl', 'train_file')
    dev_file = resolve_input_file(resolve_optional_path(args.dev_file), split_dir, 'dev.jsonl', 'dev_file')

    test_file_optional = resolve_optional_path(args.test_file)
    if test_file_optional is None and split_dir is not None:
        default_test = split_dir / 'test.jsonl'
        test_file_optional = default_test if default_test.exists() else None

    output_dir = Path(args.output_dir)
    stage1_dir = output_dir / args.stage1_dir_name
    stage2_dir = output_dir / args.stage2_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_command = build_stage_command(
        trainer_type=args.trainer_type,
        python_executable=args.python_executable,
        train_file=train_gold_file,
        dev_file=dev_file,
        output_dir=stage1_dir,
        model_name=args.base_model_name,
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        lr=args.stage1_lr,
        max_length=args.max_length,
        temperature=args.temperature,
        seed=args.seed,
        mlm_probability=args.mlm_probability,
        lambda_mntp=args.lambda_mntp,
    )
    stage2_command = build_stage_command(
        trainer_type=args.trainer_type,
        python_executable=args.python_executable,
        train_file=train_file,
        dev_file=dev_file,
        output_dir=stage2_dir,
        model_name=str(stage1_dir),
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        lr=args.stage2_lr,
        max_length=args.max_length,
        temperature=args.temperature,
        seed=args.seed + 1,
        mlm_probability=args.mlm_probability,
        lambda_mntp=args.lambda_mntp,
    )

    plan = {
        'trainer_type': args.trainer_type,
        'base_model_name': args.base_model_name,
        'split_dir': str(split_dir) if split_dir is not None else '',
        'files': {
            'train_gold': str(train_gold_file),
            'train': str(train_file),
            'dev': str(dev_file),
            'test': str(test_file_optional) if test_file_optional is not None else '',
        },
        'stages': {
            'stage1': {
                'name': 'gold_warmup',
                'output_dir': str(stage1_dir),
                'epochs': args.stage1_epochs,
                'batch_size': args.stage1_batch_size,
                'lr': args.stage1_lr,
                'command': stage1_command,
            },
            'stage2': {
                'name': 'full_finetune',
                'output_dir': str(stage2_dir),
                'epochs': args.stage2_epochs,
                'batch_size': args.stage2_batch_size,
                'lr': args.stage2_lr,
                'init_from': str(stage1_dir),
                'command': stage2_command,
            },
        },
        'shared': {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'seed': args.seed,
            'python_executable': args.python_executable,
        },
    }
    with (output_dir / 'curriculum_plan.json').open('w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    stage_records = []
    stage_records.append(run_stage('stage1_gold', stage1_command, stage1_dir, project_root))
    stage_records.append(run_stage('stage2_full', stage2_command, stage2_dir, project_root))

    summary = {
        'trainer_type': args.trainer_type,
        'base_model_name': args.base_model_name,
        'files': plan['files'],
        'output_dir': str(output_dir),
        'stage1_best_checkpoint': str(stage1_dir),
        'stage2_best_checkpoint': str(stage2_dir),
        'stage2_last_checkpoint': str(stage2_dir / 'last'),
        'stage_records': stage_records,
    }
    with (output_dir / 'curriculum_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

