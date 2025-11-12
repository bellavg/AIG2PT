"""Utilities for sampling from AIG2PT checkpoints and evaluating generated sequences."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

import networkx as nx

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .core.model import GPT, GPTConfig

logger = logging.getLogger(__name__)


def seq_to_nxgraph(seq_str: str, parsing_mode: str = 'robust') -> nx.DiGraph:
    """Convert a whitespace-separated token sequence into a NetworkX DiGraph."""

    tokens = seq_str.split()
    try:
        ctx_start = tokens.index('<boc>') + 1
        ctx_end = tokens.index('<eoc>')
        bog_start = tokens.index('<bog>') + 1
        eog_end = tokens.index('<eog>')
    except ValueError:
        return nx.DiGraph()

    ctx_tokens = tokens[ctx_start:ctx_end + 1]
    edge_tokens = [t for t in tokens[bog_start:eog_end] if t != '<sepg>']

    graph = nx.DiGraph()
    node_map = {}
    node_data = {}

    idx_pattern = re.compile(r'^IDX_(\d+)$')
    node_type_pattern = re.compile(r'^NODE_(CONST0|PI|AND|PO)$')
    edge_type_pattern = re.compile(r'^(EDGE_REG|EDGE_INV)$')

    current_node_idx = None
    current_node_type = None
    node_counter = 0
    processed_tokens = set()

    for token in ctx_tokens:
        if token == '<sepc>':
            current_node_idx = None
            current_node_type = None
            continue

        node_match = node_type_pattern.fullmatch(token)
        idx_match = idx_pattern.fullmatch(token)

        if node_match:
            current_node_type = node_match.group(0)
        elif idx_match:
            current_node_idx = idx_match.group(0)

        if current_node_type and current_node_idx:
            if current_node_idx not in processed_tokens:
                try:
                    node_index = int(idx_pattern.match(current_node_idx).group(1))
                except Exception:
                    node_index = node_counter

                node_map[current_node_idx] = node_index
                node_data[node_index] = {
                    'type': current_node_type,
                    'token': current_node_idx,
                }
                processed_tokens.add(current_node_idx)
                node_counter = max(node_counter, node_index + 1)

            current_node_idx = None
            current_node_type = None

    graph.add_nodes_from(
        (
            idx,
            node_data.get(idx, {'type': 'UNKNOWN', 'token': None}),
        )
        for idx in range(node_counter)
    )
    nx.set_node_attributes(graph, node_data)

    if parsing_mode == 'strict':
        if len(edge_tokens) % 3 != 0:
            return graph

        for i in range(0, len(edge_tokens), 3):
            src_token, dst_token, edge_token = edge_tokens[i:i + 3]
            src_idx = node_map.get(src_token)
            dst_idx = node_map.get(dst_token)
            edge_type = edge_token if edge_type_pattern.fullmatch(edge_token) else 'UNKNOWN_EDGE'

            if (
                src_idx is not None
                and dst_idx is not None
                and graph.has_node(src_idx)
                and graph.has_node(dst_idx)
            ):
                graph.add_edge(src_idx, dst_idx, type=edge_type)

    elif parsing_mode == 'robust':
        edge_idx = 0
        while edge_idx < len(edge_tokens):
            is_triplet = False
            if (
                edge_idx + 2 < len(edge_tokens)
                and idx_pattern.fullmatch(edge_tokens[edge_idx])
                and idx_pattern.fullmatch(edge_tokens[edge_idx + 1])
                and re.match(r'^EDGE_', edge_tokens[edge_idx + 2])
            ):
                is_triplet = True

            if is_triplet:
                src_token, dst_token, edge_token = edge_tokens[edge_idx:edge_idx + 3]
                src_idx = node_map.get(src_token)
                dst_idx = node_map.get(dst_token)
                edge_type = edge_token if edge_type_pattern.fullmatch(edge_token) else 'UNKNOWN_EDGE'

                if (
                    src_idx is not None
                    and dst_idx is not None
                    and graph.has_node(src_idx)
                    and graph.has_node(dst_idx)
                ):
                    graph.add_edge(src_idx, dst_idx, type=edge_type)

                edge_idx += 3
            else:
                edge_idx += 1

    return graph


class AIGSampler:
    """Helper class that wraps common sampling strategies for AIG2PT."""

    def __init__(self, model: GPT, tokenizer, device: str = 'cuda') -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def multinomial_sample(
        self,
        num_samples: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: int = 512,
        batch_size: int = 32,
    ) -> List[List[int]]:
        """Draw samples using multinomial sampling with optional top-k/top-p filtering."""
        logger.info(
            "Starting multinomial sampling: samples=%d temperature=%.3f top_k=%s top_p=%s",
            num_samples,
            temperature,
            top_k,
            top_p,
        )

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        start_token_id = self.tokenizer.convert_tokens_to_ids('<boc>')
        eog_token_id = self.tokenizer.convert_tokens_to_ids('<eog>')

        all_sequences: List[List[int]] = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            remaining = num_samples - len(all_sequences)
            if remaining <= 0:
                break
            current_batch_size = min(batch_size, remaining)

            generated = torch.full(
                (current_batch_size, 1),
                start_token_id,
                dtype=torch.long,
                device=self.device,
            )

            for _ in range(max_new_tokens):
                logits, _ = self.model(generated)
                logits = logits[:, -1, :] / temperature

                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    cutoff = values[..., -1, None]
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float('-inf')), logits)

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if eog_token_id is not None and torch.all(next_token == eog_token_id):
                    break

            all_sequences.extend(generated.cpu().tolist())
            logger.info("Multinomial batch %d/%d complete", batch_idx + 1, num_batches)

        return all_sequences[:num_samples]

    @torch.no_grad()
    def diverse_beam_search(
        self,
        num_samples: int = 100,
        num_beams: int = 5,
        num_beam_groups: int = 5,
        diversity_penalty: float = 0.5,
        max_new_tokens: int = 512,
        batch_size: int = 8,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """Generate sequences using Hugging Face beam search with diversity penalties."""
        logger.info(
            "Starting diverse beam search: samples=%d num_beams=%d groups=%d penalty=%.3f",
            num_samples,
            num_beams,
            num_beam_groups,
            diversity_penalty,
        )

        if num_beams <= 0:
            raise ValueError("num_beams must be positive")
        if num_beam_groups <= 0 or num_beams % num_beam_groups != 0:
            raise ValueError("num_beams must be divisible by num_beam_groups")

        try:
            hf_model = self.model.to_hf()
            hf_model.eval()
            hf_model.to(self.device)
        except Exception as exc:  # pragma: no cover - rare failure path
            logger.warning(
                "Falling back to multinomial-based approximation because to_hf() failed: %s",
                exc,
            )
            return self._diverse_beam_search_custom(
                num_samples,
                num_beams,
                num_beam_groups,
                diversity_penalty,
                max_new_tokens,
                batch_size,
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad_token for beam search")
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<eog>')
        if eos_token_id is None:
            raise ValueError("Tokenizer is missing '<eog>' token required for beam search")

        all_sequences: List[List[int]] = []
        while len(all_sequences) < num_samples:
            remaining = num_samples - len(all_sequences)
            current_batch_size = min(batch_size, remaining, num_beams)

            inputs = self.tokenizer(
                ['<boc>'] * current_batch_size,
                return_tensors='pt',
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            try:
                do_sample = temperature != 1.0
                generation_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_new_tokens': max_new_tokens,
                    'num_beams': num_beams,
                    'num_return_sequences': current_batch_size,
                    'num_beam_groups': num_beam_groups,
                    'diversity_penalty': diversity_penalty,
                    'early_stopping': True,
                    'pad_token_id': pad_id,
                    'eos_token_id': eos_token_id,
                    'do_sample': do_sample,
                }
                if do_sample:
                    generation_kwargs['temperature'] = max(temperature, 1e-5)

                outputs = hf_model.generate(**generation_kwargs)
            except Exception as exc:  # pragma: no cover - passthrough to fallback
                logger.warning(
                    "Beam search generation failed (falling back to custom sampler): %s",
                    exc,
                )
                return self._diverse_beam_search_custom(
                    num_samples,
                    num_beams,
                    num_beam_groups,
                    diversity_penalty,
                    max_new_tokens,
                    batch_size,
                )

            all_sequences.extend(outputs.cpu().tolist())
            logger.info("Beam search generated %d/%d sequences", len(all_sequences), num_samples)

        return all_sequences[:num_samples]

    def _diverse_beam_search_custom(
        self,
        num_samples: int,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        max_new_tokens: int,
        batch_size: int,
    ) -> List[List[int]]:
        """Fallback strategy that reuses multinomial sampling with varied temperatures."""
        logger.info(
            "Using fallback multinomial strategy for diverse beam search (beams=%d groups=%d)",
            num_beams,
            num_beam_groups,
        )
        all_sequences: List[List[int]] = []
        samples_per_group = max(1, num_samples // max(1, num_beam_groups))

        for group_idx in range(num_beam_groups):
            temp = max(0.1, 0.5 + group_idx * max(diversity_penalty, 0.1))
            sequences = self.multinomial_sample(
                num_samples=samples_per_group,
                temperature=temp,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            all_sequences.extend(sequences)

        if len(all_sequences) < num_samples:
            extra = self.multinomial_sample(
                num_samples=num_samples - len(all_sequences),
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            all_sequences.extend(extra)

        return all_sequences[:num_samples]

    def decode_sequences(self, token_sequences: Sequence[Sequence[int]]) -> List[str]:
        """Convert token ID sequences into whitespace-delimited token strings."""
        return [self.tokenizer.decode(seq, skip_special_tokens=False) for seq in token_sequences]


class AIGEvaluator:
    """Compute validity, uniqueness, and novelty statistics for generated sequences."""

    def __init__(
        self,
        tokenizer=None,
        min_and_nodes: int = 1,
        min_po_nodes: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_and_nodes = max(0, min_and_nodes)
        self.min_po_nodes = max(1, min_po_nodes)

    def evaluate_validity(self, sequences: Sequence[str]):
        valid_count = 0
        invalid_reasons = []
        valid_sequences: List[str] = []
        validity_mask: List[bool] = []

        for seq in sequences:
            is_valid, reason = self._check_sequence_validity(seq)
            validity_mask.append(is_valid)
            if is_valid:
                valid_count += 1
                valid_sequences.append(seq)
            else:
                if reason is not None:
                    invalid_reasons.append(reason)

        total = len(sequences)
        validity_rate = valid_count / total if total else 0.0
        return {
            'total': total,
            'valid': valid_count,
            'invalid': total - valid_count,
            'validity_rate': validity_rate,
            'invalid_reasons': invalid_reasons[:10],
            'valid_sequences': valid_sequences,
            'validity_mask': validity_mask,
        }

    def _check_sequence_validity(self, sequence: str):
        required_markers = ['<boc>', '<eoc>', '<bog>', '<eog>']
        for marker in required_markers:
            if marker not in sequence:
                return False, f"Missing {marker}"

        try:
            tokens = sequence.split()
            boc_idx = tokens.index('<boc>')
            eoc_idx = tokens.index('<eoc>')
            bog_idx = tokens.index('<bog>')
            eog_idx = tokens.index('<eog>')
        except ValueError as exc:
            return False, f"Marker error: {exc}"

        if not (boc_idx < eoc_idx < bog_idx < eog_idx):
            return False, "Markers out of order"

        node_section = tokens[boc_idx + 1:eoc_idx]
        edge_section = tokens[bog_idx + 1:eog_idx]

        if not node_section:
            return False, "Node section is empty"
        if len(node_section) % 3 != 0:
            return False, "Node section must consist of <sepc> NODE_* IDX_* triplets"

        allowed_node_types = {'NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO'}
        allowed_edge_types = {'EDGE_REG', 'EDGE_INV'}
        idx_pattern = re.compile(r'^IDX_\d+$')

        node_ids = set()
        po_ids = set()

        for i in range(0, len(node_section), 3):
            sep_token, node_type, node_idx = node_section[i:i + 3]

            if sep_token != '<sepc>':
                return False, f"Unexpected token '{sep_token}' in node section"
            if node_type not in allowed_node_types:
                return False, f"Unknown node type '{node_type}'"
            if not idx_pattern.fullmatch(node_idx):
                return False, f"Malformed node identifier '{node_idx}'"
            if node_idx in node_ids:
                return False, f"Duplicate node identifier '{node_idx}'"

            node_ids.add(node_idx)
            if node_type == 'NODE_PO':
                po_ids.add(node_idx)

        if edge_section and len(edge_section) % 4 != 0:
            return False, "Edge section must consist of <sepg> IDX_src IDX_dst EDGE_* quartets"

        po_targets_with_inputs = set()
        for i in range(0, len(edge_section), 4):
            seg = edge_section[i:i + 4]
            if len(seg) < 4:
                return False, "Incomplete edge specification"
            sep_token, src_idx, dst_idx, edge_type = seg

            if sep_token != '<sepg>':
                return False, f"Unexpected token '{sep_token}' in edge section"
            if not idx_pattern.fullmatch(src_idx):
                return False, f"Malformed edge source identifier '{src_idx}'"
            if not idx_pattern.fullmatch(dst_idx):
                return False, f"Malformed edge destination identifier '{dst_idx}'"
            if src_idx not in node_ids:
                return False, f"Edge references unknown node '{src_idx}'"
            if dst_idx not in node_ids:
                return False, f"Edge references unknown node '{dst_idx}'"
            if edge_type not in allowed_edge_types:
                return False, f"Unknown edge type '{edge_type}'"

            if dst_idx in po_ids:
                po_targets_with_inputs.add(dst_idx)

        if po_ids:
            missing_po_drivers = sorted(po_ids - po_targets_with_inputs)
            if missing_po_drivers:
                missing_str = ', '.join(missing_po_drivers)
                return False, f"POs missing drivers: {missing_str}"

        graph = seq_to_nxgraph(sequence, parsing_mode='strict')
        if graph.number_of_nodes() == 0:
            return False, "Failed to construct graph from sequence"

        node_attrs = dict(graph.nodes(data=True))
        if not node_attrs:
            return False, "Parsed graph has no nodes"

        node_labels = {
            node: data.get('token') or f'IDX_{node}'
            for node, data in node_attrs.items()
        }

        invalid_nodes = [
            node_labels[node]
            for node, data in node_attrs.items()
            if data.get('type') not in allowed_node_types
        ]
        if invalid_nodes:
            return False, f"Unsupported node types detected: {', '.join(sorted(invalid_nodes))}"

        const0_nodes = [node for node, data in node_attrs.items() if data.get('type') == 'NODE_CONST0']

        po_nodes = [node for node, data in node_attrs.items() if data.get('type') == 'NODE_PO']
        if not po_nodes:
            return False, "AIG must include at least one NODE_PO"

        unknown_edges = [
            (node_labels.get(src), node_labels.get(dst))
            for src, dst, data in graph.edges(data=True)
            if data.get('type') == 'UNKNOWN_EDGE'
        ]
        if unknown_edges:
            edge_desc = ', '.join(f"{src}->{dst}" for src, dst in unknown_edges)
            return False, f"Unknown edge types detected on edges: {edge_desc}"

        if not nx.is_directed_acyclic_graph(graph):
            return False, "Graph contains cycles"

        for node_idx, data in graph.nodes(data=True):
            node_type = data.get('type')
            label = node_labels[node_idx]
            indeg = graph.in_degree(node_idx)
            outdeg = graph.out_degree(node_idx)

            if node_type == 'NODE_CONST0':
                if indeg != 0:
                    return False, f"NODE_CONST0 '{label}' must have indegree 0"
            elif node_type == 'NODE_PI':
                if indeg != 0:
                    return False, f"NODE_PI '{label}' must have indegree 0"
            elif node_type == 'NODE_AND':
                if indeg != 2:
                    return False, f"NODE_AND '{label}' must have exactly two inputs"
            elif node_type == 'NODE_PO':
                if outdeg != 0:
                    return False, f"NODE_PO '{label}' must have outdegree 0"
                if indeg == 0:
                    return False, f"NODE_PO '{label}' must have at least one input"

        pi_nodes = [node for node, data in node_attrs.items() if data.get('type') == 'NODE_PI']
        if not pi_nodes and not const0_nodes:
            return False, "AIG must include at least one input source"

        and_nodes = [node for node, data in node_attrs.items() if data.get('type') == 'NODE_AND']
        if len(and_nodes) < self.min_and_nodes:
            return False, (
                f"AIG must include at least {self.min_and_nodes} NODE_AND nodes (found {len(and_nodes)})"
            )

        if len(po_nodes) < self.min_po_nodes:
            return False, (
                f"AIG must include at least {self.min_po_nodes} NODE_PO nodes (found {len(po_nodes)})"
            )

        non_const_isolates = [
            node_labels[node]
            for node in graph.nodes
            if graph.in_degree(node) == 0
            and graph.out_degree(node) == 0
            and node_attrs.get(node, {}).get('type') != 'NODE_CONST0'
        ]
        if non_const_isolates:
            return False, "Isolated non-constant nodes detected: " + ', '.join(sorted(non_const_isolates))

        source_nodes = const0_nodes + pi_nodes
        for po_node in po_nodes:
            if not any(nx.has_path(graph, src, po_node) for src in source_nodes):
                return False, (
                    f"NODE_PO '{node_labels[po_node]}' is not driven by any input source"
                )

        for pi_node in pi_nodes:
            if not any(nx.has_path(graph, pi_node, po_node) for po_node in po_nodes):
                return False, f"NODE_PI '{node_labels[pi_node]}' does not drive any PO"

        return True, None

    def evaluate_uniqueness(
        self,
        sequences: Sequence[str],
        validity_mask: Optional[Sequence[bool]] = None,
    ):
        if validity_mask is None:
            validity_info = self.evaluate_validity(sequences)
            valid_sequences = validity_info['valid_sequences']
        else:
            valid_sequences = [seq for seq, is_valid in zip(sequences, validity_mask) if is_valid]

        unique_seqs = set(valid_sequences)
        total = len(valid_sequences)
        uniqueness_rate = len(unique_seqs) / total if total else 0.0
        return {
            'total': total,
            'unique': len(unique_seqs),
            'duplicates': total - len(unique_seqs),
            'uniqueness_rate': uniqueness_rate,
            'valid_sequences': valid_sequences,
        }

    def evaluate_novelty(
        self,
        generated_sequences: Sequence[str],
        training_sequences: Sequence[str],
        generated_validity_mask: Optional[Sequence[bool]] = None,
    ):
        if generated_validity_mask is None:
            generated_validity = self.evaluate_validity(generated_sequences)
            generated_valid = generated_validity['valid_sequences']
        else:
            generated_valid = [
                seq for seq, is_valid in zip(generated_sequences, generated_validity_mask) if is_valid
            ]

        training_validity = self.evaluate_validity(training_sequences)
        training_valid_set = set(training_validity['valid_sequences'])

        total = len(generated_valid)
        novel_count = sum(1 for seq in generated_valid if seq not in training_valid_set)
        novelty_rate = novel_count / total if total else 0.0
        return {
            'total': total,
            'novel': novel_count,
            'memorized': total - novel_count,
            'novelty_rate': novelty_rate,
            'reference_total': len(training_sequences),
            'reference_valid': len(training_valid_set),
        }

    def comprehensive_evaluation(
        self,
        generated_sequences: Sequence[str],
        training_sequences: Optional[Sequence[str]] = None,
    ):
        validity = self.evaluate_validity(generated_sequences)
        uniqueness = self.evaluate_uniqueness(
            generated_sequences,
            validity_mask=validity['validity_mask'],
        )

        results = {
            'validity': validity,
            'uniqueness': uniqueness,
        }

        if training_sequences:
            results['novelty'] = self.evaluate_novelty(
                generated_sequences,
                training_sequences,
                generated_validity_mask=validity['validity_mask'],
            )

        results['summary'] = {
            'total_generated': len(generated_sequences),
            'valid_rate': results['validity']['validity_rate'],
            'unique_rate': results['uniqueness']['uniqueness_rate'],
            'novel_rate': results.get('novelty', {}).get('novelty_rate'),
            'valid_considered': results['validity']['valid'],
        }
        return results


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device: str = 'cuda', weights_only: bool = False):
    """Load a GPT checkpoint and tokenizer for sampling/evaluation."""
    logger.info("Loading model checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(unwanted_prefix):
            cleaned_state_dict[key[len(unwanted_prefix):]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    logger.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer

