"""Pytest-based smoke tests for sampling and evaluation utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from ..sampling_and_evaluation import AIGEvaluator, AIGSampler, load_model_and_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / 'results' / 'ckpt.pt'
TOKENIZER_PATH = PROJECT_ROOT / 'aig2pt' / 'dataset' / 'tokenizer'
TRAIN_DATA_DIR = PROJECT_ROOT / 'aig2pt' / 'dataset' / 'aig_prepared'


@pytest.fixture(scope='module')
def model_and_tokenizer():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found at {CHECKPOINT_PATH}")
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH}")

    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=str(CHECKPOINT_PATH),
        tokenizer_path=str(TOKENIZER_PATH),
        device='cpu',
    )
    return model, tokenizer


@pytest.fixture()
def sampler(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    return AIGSampler(model, tokenizer, device='cpu')


@pytest.fixture(scope='module')
def training_sequences(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer

    if not TRAIN_DATA_DIR.exists():
        pytest.skip(f"training data directory not found at {TRAIN_DATA_DIR}")

    meta_path = TRAIN_DATA_DIR / 'data_meta.json'
    train_dir = TRAIN_DATA_DIR / 'train'
    if not meta_path.exists() or not train_dir.exists():
        pytest.skip("training metadata or split directory missing")

    with meta_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    shape = metadata.get('train', {}).get('token_ids_shape')
    num_graphs = metadata.get('train', {}).get('num_graphs', 0)
    if not shape or num_graphs == 0:
        pytest.skip('no training graphs available')

    count = min(5, num_graphs)
    token_ids = np.memmap(train_dir / 'token_ids.bin', dtype=np.int16, mode='r', shape=tuple(shape))

    sequences = []
    for idx in range(count):
        tokens = token_ids[idx]
        tokens = tokens[tokens != -100]
        sequences.append(tokenizer.decode(tokens.tolist(), skip_special_tokens=False))

    if not sequences:
        pytest.skip('failed to load any training sequences')

    return sequences


def test_multinomial_sampling_and_evaluation(sampler, model_and_tokenizer, training_sequences):
    model, tokenizer = model_and_tokenizer
    sequences = sampler.multinomial_sample(
        num_samples=2,
        temperature=1.0,
        max_new_tokens=16,
        batch_size=2,
    )

    assert len(sequences) == 2
    assert all(isinstance(seq, list) and seq for seq in sequences)

    decoded = sampler.decode_sequences(sequences)
    evaluator = AIGEvaluator(tokenizer)
    eval_results = evaluator.comprehensive_evaluation(decoded, training_sequences)
    print(f"Multinomial sampling evaluation: {eval_results}")

    assert eval_results['summary']['total_generated'] == 2
    assert 0.0 <= eval_results['summary']['valid_rate'] <= 1.0
    assert 0.0 <= eval_results['summary']['unique_rate'] <= 1.0
    assert eval_results['validity']['total'] == 2
    assert eval_results['uniqueness']['total'] == eval_results['validity']['valid']


def test_diverse_beam_sampling_returns_sequences(sampler, training_sequences):
    tokenizer = sampler.tokenizer
    if tokenizer.pad_token_id is None:
        pytest.skip('Tokenizer lacks pad_token required for beam search')

    sequences = sampler.diverse_beam_search(
        num_samples=2,
        num_beams=2,
        num_beam_groups=1,
        diversity_penalty=0.3,
        max_new_tokens=16,
        batch_size=1,
    )

    assert len(sequences) == 2
    assert all(isinstance(seq, list) and seq for seq in sequences)
    decoded = sampler.decode_sequences(sequences)
    evaluator = AIGEvaluator(tokenizer)
    metrics = evaluator.comprehensive_evaluation(decoded, training_sequences)
    print(f"Diverse beam sampling evaluation: {metrics}")

    assert metrics['summary']['total_generated'] == len(decoded)
    assert 'validity' in metrics
    assert 'uniqueness' in metrics


def test_training_sequences_are_valid(training_sequences):
    evaluator = AIGEvaluator()
    results = evaluator.comprehensive_evaluation(training_sequences)
    print(f"Training sequence evaluation: {results}")

    assert results['validity']['total'] == len(training_sequences)
    assert results['validity']['valid'] == len(training_sequences)


def test_evaluator_handles_manual_sequences():
    evaluator = AIGEvaluator()

    valid_sequence = (
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
        "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_INV "
        "<sepg> IDX_1 IDX_2 EDGE_REG <sepg> IDX_2 IDX_3 EDGE_REG <eog>"
    )

    missing_marker_sequence = (
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PO IDX_1 <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>"
    )

    dangling_edge_sequence = (
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_PO IDX_2 <eoc> "
        "<bog> <sepg> IDX_1 IDX_2 EDGE_REG <sepg> IDX_0 IDX_99 EDGE_REG <eog>"
    )

    orphan_po_sequence = (
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
        "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_INV "
        "<sepg> IDX_1 IDX_2 EDGE_REG <eog>"
    )

    sequences = [
        valid_sequence,
        missing_marker_sequence,
        dangling_edge_sequence,
        orphan_po_sequence,
    ]

    results = evaluator.comprehensive_evaluation(sequences)
    print(f"Manual sequence evaluation: {results}")

    assert results['summary']['total_generated'] == len(sequences)
    assert results['validity']['valid'] == 1
    assert results['validity']['invalid'] == len(sequences) - 1

    invalid_reasons = " ".join(results['validity']['invalid_reasons'])
    assert "Missing <eoc>" in invalid_reasons
    assert "unknown node 'IDX_99'" in invalid_reasons
    assert "POs missing drivers" in invalid_reasons


def test_evaluator_novelty_detects_memorized_sequences(training_sequences):
    evaluator = AIGEvaluator()

    memorized = training_sequences[0]
    novel_sequence = (
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
        "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_INV "
        "<sepg> IDX_1 IDX_2 EDGE_INV <sepg> IDX_2 IDX_3 EDGE_REG <eog>"
    )

    generated = [memorized, novel_sequence]

    novelty = evaluator.evaluate_novelty(generated, training_sequences)
    print(f"Novelty evaluation: {novelty}")

    assert novelty['total'] == len(generated)
    assert novelty['memorized'] == 1
    assert novelty['novel'] == 1
    assert 0.0 <= novelty['novelty_rate'] <= 1.0

