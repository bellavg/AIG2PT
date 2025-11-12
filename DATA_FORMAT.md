g # G2PT Data Format Analysis

## Overview
G2PT (Graph-to-Pretrained-Transformer) expects graph data to be converted into a **sequential text format** with special tokens.

## Data Format

### Input: PyTorch Geometric Graph Format
Each graph has:
- `x`: Node features (one-hot encoded node types)
- `edge_index`: Edge connectivity (2 x num_edges tensor)
- `edge_attr`: Edge features (one-hot encoded edge types)

### Output: Tokenized Text Sequence

The graph is serialized into a text sequence with the following structure:

```
<boc> <sepc> ATOM_TYPE_0 IDX_0 <sepc> ATOM_TYPE_1 IDX_1 ... <eoc> <bog> <sepg> IDX_i IDX_j EDGE_TYPE <sepg> IDX_k IDX_l EDGE_TYPE ... <eog>
```

**Structure breakdown:**
1. **Node Context Section** (between `<boc>` and `<eoc>`):
   - `<boc>`: Begin of context
   - For each node: `<sepc> NODE_TYPE IDX_i`
     - `<sepc>`: Separator for node
     - `NODE_TYPE`: Type of node (e.g., `ATOM_C`, `ATOM_N`, or just `NODE` for generic graphs)
     - `IDX_i`: Unique identifier for the node (e.g., `IDX_0`, `IDX_1`, etc.)
   - `<eoc>`: End of context

2. **Edge/Graph Section** (between `<bog>` and `<eog>`):
   - `<bog>`: Begin of graph
   - For each edge: `<sepg> IDX_source IDX_dest EDGE_TYPE`
     - `<sepg>`: Separator for edge
     - `IDX_source`: Source node identifier
     - `IDX_dest`: Destination node identifier
     - `EDGE_TYPE`: Type of edge (e.g., `BOND_SINGLE`, `BOND_DOUBLE`, or just `EDGE` for generic graphs)
   - `<eog>`: End of graph

### Example for a Simple Graph

For a graph with 3 nodes (all type "NODE") and 2 edges (all type "EDGE"):
```
<boc> <sepc> NODE IDX_0 <sepc> NODE IDX_1 <sepc> NODE IDX_2 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE <sepg> IDX_1 IDX_2 EDGE <eog>
```

## For AIG Graphs

For AIG (And-Inverter Graph) format, we need to adapt this:

### Node Types (for AIGs):
- `PI`: Primary Input
- `CONST`: Constant (usually 0)
- `AND`: AND gate
- (Note: Primary Outputs are not nodes, but references to other nodes)

### Edge Types (for AIGs):
- `FWD`: Forward (non-inverting) edge
- `NOT`: Inverting edge (represents NOT gate)

### Example AIG Sequence:
```
<boc> <sepc> CONST IDX_0 <sepc> PI IDX_1 <sepc> PI IDX_2 <sepc> AND IDX_3 <eoc> <bog> <sepg> IDX_1 IDX_3 FWD <sepg> IDX_2 IDX_3 NOT <eog>
```

## Storage Format

G2PT uses **numpy memmap** for efficient large-scale data storage:

### Binary Files (.bin):
- `xs.bin`: Node features (shape: [num_graphs, max_nodes])
- `edge_indices.bin`: Edge indices (shape: [num_graphs, 2, max_edges])
- `edge_attrs.bin`: Edge attributes (shape: [num_graphs, max_edges])

### Data Types:
- All stored as `int16` (with `-100` as padding value)
- Node features: integer indices (0, 1, 2, ...) representing node types
- Edge indices: integer node indices
- Edge attributes: integer indices (0, 1, 2, ...) representing edge types

### Shapes:
Different datasets have different shapes based on their graph sizes:
- **tree**: `{'x': (256, 64), 'edge_index': (256, 2, 126), 'edge_attr': (256, 126)}`
- **moses**: `{'x': (1419512, 27), 'edge_index': (1419512, 2, 62), 'edge_attr': (1419512, 62)}`

## Pipeline Flow

```
Raw .aig files
    ↓
Parse with aigverse → Extract (inputs, outputs, gates)
    ↓
Convert to PyG format → Data(x=node_features, edge_index=edges, edge_attr=edge_types)
    ↓
Serialize to text sequence → "<boc> ... <eoc> <bog> ... <eog>"
    ↓
Tokenize with HuggingFace tokenizer → input_ids, attention_mask, labels
    ↓
Train GPT model with next-token prediction
```

## Tokenizer Requirements

G2PT uses a **custom HuggingFace tokenizer** that includes:
- Special tokens: `<boc>`, `<eoc>`, `<bog>`, `<eog>`, `<sepc>`, `<sepg>`, `<pad>`, `<eos>`
- Node type tokens: `NODE`, `ATOM_C`, `ATOM_N`, etc. (or for AIGs: `PI`, `AND`, `CONST`)
- Edge type tokens: `EDGE`, `BOND_SINGLE`, etc. (or for AIGs: `FWD`, `NOT`)
- Index tokens: `IDX_0`, `IDX_1`, `IDX_2`, ... up to max graph size

## Key Observations

1. **Order Matters**: G2PT supports two ordering strategies:
   - `bfs`: Breadth-first search ordering of edges
   - `deg`: Degree-based ordering (minimum degree first)

2. **Node Permutation**: Random node permutation is applied for data augmentation

3. **Padding**: All sequences are padded to `max_length` (block_size) for batching

4. **For AIG2PT**: We need to:
   - Create a tokenizer with AIG-specific vocabulary
   - Convert `.aig` files to this text format
   - Store in the same binary format for efficient loading
   - Use the same training pipeline

