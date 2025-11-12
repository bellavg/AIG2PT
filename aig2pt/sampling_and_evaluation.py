# AIG2PT Sampling and Evaluation Module
"""
Comprehensive sampling and evaluation for AIG generation.

Supports:
- Multinomial sampling (temperature-based)
- Diverse beam search
- Top-k and Top-p (nucleus) sampling
- Evaluation metrics (validity, uniqueness, novelty)
"""

import os
import json
import pickle
import logging
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from core.model import GPT, GPTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIG2PT_Sampler")


class AIGSampler:
    """Handles different sampling strategies for AIG generation."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def multinomial_sample(self, num_samples=100, temperature=1.0, top_k=None,
                          top_p=None, max_new_tokens=512, batch_size=32):
        """
        Multinomial sampling with optional top-k and top-p filtering.

        Args:
            num_samples: Number of sequences to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Nucleus sampling - keep tokens with cumulative probability >= p
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation

        Returns:
            List of generated token sequences
        """
        logger.info(f"Starting multinomial sampling: {num_samples} samples, T={temperature}, top_k={top_k}, top_p={top_p}")

        # Start token
        start_token_id = self.tokenizer.convert_tokens_to_ids('<boc>')

        all_sequences = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_sequences))

            # Initialize with start token
            generated = torch.full((current_batch_size, 1), start_token_id,
                                 dtype=torch.long, device=self.device)

            for step in range(max_new_tokens):
                # Get logits for next token
                logits, _ = self.model(generated)
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Check for end token (<eog>)
                eog_token_id = self.tokenizer.convert_tokens_to_ids('<eog>')
                if (next_token == eog_token_id).all():
                    break

            all_sequences.extend(generated.cpu().tolist())
            logger.info(f"Batch {batch_idx + 1}/{num_batches} complete")

        return all_sequences[:num_samples]

    @torch.no_grad()
    def diverse_beam_search(self, num_samples=100, num_beams=5, num_beam_groups=5,
                           diversity_penalty=0.5, max_new_tokens=512, batch_size=8):
        """
        Diverse beam search for generating varied AIG sequences.

        Args:
            num_samples: Number of sequences to generate
            num_beams: Total number of beams
            num_beam_groups: Number of diverse groups (must divide num_beams)
            diversity_penalty: Penalty for similar sequences across groups
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for generation

        Returns:
            List of generated token sequences
        """
        logger.info(f"Starting diverse beam search: {num_samples} samples, beams={num_beams}, groups={num_beam_groups}, penalty={diversity_penalty}")

        if num_beams % num_beam_groups != 0:
            raise ValueError(f"num_beams ({num_beams}) must be divisible by num_beam_groups ({num_beam_groups})")

        # Convert model to HuggingFace format for .generate()
        try:
            hf_model = self.model.to_hf()
            hf_model.eval()
            hf_model.to(self.device)
        except:
            logger.warning("Could not convert to HF format, using custom implementation")
            return self._diverse_beam_search_custom(num_samples, num_beams, num_beam_groups,
                                                    diversity_penalty, max_new_tokens, batch_size)

        start_token = '<boc>'
        all_sequences = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_sequences))

            inputs = self.tokenizer([start_token] * current_batch_size, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                num_return_sequences=num_beam_groups,  # Return one from each group
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids('<eog>'),
                early_stopping=True
            )

            all_sequences.extend(outputs.cpu().tolist())
            logger.info(f"Batch {batch_idx + 1}/{num_batches} complete")

        return all_sequences[:num_samples]

    def _diverse_beam_search_custom(self, num_samples, num_beams, num_beam_groups,
                                    diversity_penalty, max_new_tokens, batch_size):
        """Custom diverse beam search implementation if HF conversion fails."""
        logger.info("Using custom diverse beam search implementation")

        # Simplified version - just use multinomial with different temperatures per group
        all_sequences = []
        samples_per_group = num_samples // num_beam_groups

        for group_idx in range(num_beam_groups):
            # Vary temperature per group for diversity
            temp = 0.5 + (group_idx * 0.3)
            sequences = self.multinomial_sample(
                num_samples=samples_per_group,
                temperature=temp,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size
            )
            all_sequences.extend(sequences)

        return all_sequences[:num_samples]

    def decode_sequences(self, token_sequences):
        """Convert token IDs to text sequences."""
        return [self.tokenizer.decode(seq, skip_special_tokens=False)
                for seq in token_sequences]


class AIGEvaluator:
    """Evaluate generated AIG sequences."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def evaluate_validity(self, sequences):
        """
        Check validity of generated sequences.

        Returns:
            dict with validity metrics
        """
        valid_count = 0
        invalid_reasons = []

        for seq in sequences:
            is_valid, reason = self._check_sequence_validity(seq)
            if is_valid:
                valid_count += 1
            else:
                invalid_reasons.append(reason)

        validity_rate = valid_count / len(sequences) if sequences else 0

        return {
            'total': len(sequences),
            'valid': valid_count,
            'invalid': len(sequences) - valid_count,
            'validity_rate': validity_rate,
            'invalid_reasons': invalid_reasons[:10]  # Sample of reasons
        }

    def _check_sequence_validity(self, sequence):
        """Check if a single sequence is valid."""
        # Check for required markers
        required_markers = ['<boc>', '<eoc>', '<bog>', '<eog>']
        for marker in required_markers:
            if marker not in sequence:
                return False, f"Missing {marker}"

        # Check order
        try:
            boc_idx = sequence.index('<boc>')
            eoc_idx = sequence.index('<eoc>')
            bog_idx = sequence.index('<bog>')
            eog_idx = sequence.index('<eog>')

            if not (boc_idx < eoc_idx < bog_idx < eog_idx):
                return False, "Markers out of order"
        except ValueError as e:
            return False, f"Marker error: {e}"

        return True, None

    def evaluate_uniqueness(self, sequences):
        """Calculate uniqueness rate."""
        unique_seqs = set(sequences)
        uniqueness_rate = len(unique_seqs) / len(sequences) if sequences else 0

        return {
            'total': len(sequences),
            'unique': len(unique_seqs),
            'duplicates': len(sequences) - len(unique_seqs),
            'uniqueness_rate': uniqueness_rate
        }

    def evaluate_novelty(self, generated_sequences, training_sequences):
        """Calculate novelty compared to training set."""
        training_set = set(training_sequences)
        novel_count = sum(1 for seq in generated_sequences if seq not in training_set)
        novelty_rate = novel_count / len(generated_sequences) if generated_sequences else 0

        return {
            'total': len(generated_sequences),
            'novel': novel_count,
            'memorized': len(generated_sequences) - novel_count,
            'novelty_rate': novelty_rate
        }

    def comprehensive_evaluation(self, generated_sequences, training_sequences=None):
        """Run all evaluations."""
        results = {
            'validity': self.evaluate_validity(generated_sequences),
            'uniqueness': self.evaluate_uniqueness(generated_sequences)
        }

        if training_sequences:
            results['novelty'] = self.evaluate_novelty(generated_sequences, training_sequences)

        # Summary
        results['summary'] = {
            'total_generated': len(generated_sequences),
            'valid_rate': results['validity']['validity_rate'],
            'unique_rate': results['uniqueness']['uniqueness_rate'],
            'novel_rate': results.get('novelty', {}).get('novelty_rate', None)
        }

        return results


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device='cuda'):
    """Load trained model and tokenizer."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer


# Example usage
if __name__ == '__main__':
    print("AIG2PT Sampling and Evaluation Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("  1. Multinomial sampling (temperature, top-k, top-p)")
    print("  2. Diverse beam search")
    print("  3. Comprehensive evaluation metrics")
    print("\nExample usage:")
    print("""
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        'results/aig-12L/ckpt.pt',
        'aig2pt/dataset/tokenizer'
    )
    
    # Create sampler
    sampler = AIGSampler(model, tokenizer)
    
    # Generate with multinomial sampling
    sequences = sampler.multinomial_sample(
        num_samples=100,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    
    # Generate with diverse beam search
    diverse_sequences = sampler.diverse_beam_search(
        num_samples=100,
        num_beams=10,
        num_beam_groups=5,
        diversity_penalty=0.5
    )
    
    # Evaluate
    evaluator = AIGEvaluator()
    results = evaluator.comprehensive_evaluation(
        sampler.decode_sequences(sequences)
    )
    print(results)
    """)

