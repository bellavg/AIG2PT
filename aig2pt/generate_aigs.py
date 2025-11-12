#!/usr/bin/env python3
"""
Comprehensive sampling script for AIG generation.
Supports multinomial sampling and diverse beam search.
"""

import argparse
import os
import json
import pickle
from pathlib import Path
import torch

from sampling_and_evaluation import (
    AIGSampler,
    AIGEvaluator,
    load_model_and_tokenizer
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate AIGs using various sampling strategies')

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., results/aig-12L/ckpt.pt)')
    parser.add_argument('--tokenizer', type=str, default='dataset/tokenizer',
                       help='Path to tokenizer directory')
    parser.add_argument('--output_dir', type=str, default='generated_aigs',
                       help='Output directory for generated AIGs')

    # Sampling strategy
    parser.add_argument('--method', type=str, default='multinomial',
                       choices=['multinomial', 'diverse_beam', 'both'],
                       help='Sampling method to use')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of AIGs to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum tokens per sequence')

    # Multinomial sampling parameters
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k filtering (keep top k tokens)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Nucleus sampling (keep tokens with cumulative prob >= p)')

    # Diverse beam search parameters
    parser.add_argument('--num_beams', type=int, default=10,
                       help='Number of beams for beam search')
    parser.add_argument('--num_beam_groups', type=int, default=5,
                       help='Number of diverse groups')
    parser.add_argument('--diversity_penalty', type=float, default=0.5,
                       help='Diversity penalty for beam search')

    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on generated sequences')
    parser.add_argument('--training_data', type=str, default=None,
                       help='Path to training data for novelty evaluation')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint,
        args.tokenizer,
        device=args.device
    )

    # Create sampler
    sampler = AIGSampler(model, tokenizer, device=args.device)

    results = {}

    # Multinomial sampling
    if args.method in ['multinomial', 'both']:
        print(f"\n{'='*60}")
        print(f"Multinomial Sampling")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  Samples: {args.num_samples}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-k: {args.top_k}")
        print(f"  Top-p: {args.top_p}")

        multinomial_sequences = sampler.multinomial_sample(
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size
        )

        # Decode
        multinomial_texts = sampler.decode_sequences(multinomial_sequences)

        # Save
        output_file = os.path.join(args.output_dir, 'multinomial_sequences.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(multinomial_sequences, f)
        print(f"Saved to {output_file}")

        # Save as text too
        text_file = os.path.join(args.output_dir, 'multinomial_sequences.txt')
        with open(text_file, 'w') as f:
            f.write('\n'.join(multinomial_texts))

        results['multinomial'] = multinomial_texts

    # Diverse beam search
    if args.method in ['diverse_beam', 'both']:
        print(f"\n{'='*60}")
        print(f"Diverse Beam Search")
        print(f"{'='*60}")
        print(f"Parameters:")
        print(f"  Samples: {args.num_samples}")
        print(f"  Num beams: {args.num_beams}")
        print(f"  Beam groups: {args.num_beam_groups}")
        print(f"  Diversity penalty: {args.diversity_penalty}")

        beam_sequences = sampler.diverse_beam_search(
            num_samples=args.num_samples,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size
        )

        # Decode
        beam_texts = sampler.decode_sequences(beam_sequences)

        # Save
        output_file = os.path.join(args.output_dir, 'diverse_beam_sequences.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(beam_sequences, f)
        print(f"Saved to {output_file}")

        # Save as text
        text_file = os.path.join(args.output_dir, 'diverse_beam_sequences.txt')
        with open(text_file, 'w') as f:
            f.write('\n'.join(beam_texts))

        results['diverse_beam'] = beam_texts

    # Evaluation
    if args.evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluation")
        print(f"{'='*60}")

        evaluator = AIGEvaluator(tokenizer)

        for method_name, sequences in results.items():
            print(f"\n{method_name.upper()} Results:")
            print(f"{'-'*40}")

            # Load training data if provided
            training_sequences = None
            if args.training_data:
                with open(args.training_data, 'r') as f:
                    training_sequences = f.read().split('\n')

            eval_results = evaluator.comprehensive_evaluation(
                sequences,
                training_sequences
            )

            # Print summary
            summary = eval_results['summary']
            print(f"Total generated: {summary['total_generated']}")
            print(f"Valid rate: {summary['valid_rate']:.2%}")
            print(f"Unique rate: {summary['unique_rate']:.2%}")
            if summary['novel_rate'] is not None:
                print(f"Novel rate: {summary['novel_rate']:.2%}")

            # Save detailed results
            eval_file = os.path.join(args.output_dir, f'{method_name}_evaluation.json')
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Detailed results saved to {eval_file}")

    print(f"\n{'='*60}")
    print(f"Generation complete! Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

