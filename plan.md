# AIG2PT: Action Plan

This plan outlines the steps to develop and evaluate AIG2PT, a foundation model for logic synthesis.

## Development Note
- **Local Environment:** Small sample `.aig` files in `raw_data/` for testing and development
- **Server Environment:** Full dataset will be stored on server due to size
- **Strategy:** Develop and validate pipeline locally, then scale to server for full training

## Phase 1: Data Generation

### Local Development (Current)
- [x] **1.0. Setup Local Test Data**
    - [x] Small sample `.aig` files available in `raw_data/adder/` for pipeline testing
    - [ ] **Data Format Understanding** (see `DATA_FORMAT.md`)
        - [ ] G2PT expects graphs as text sequences: `<boc> NODE_INFO <eoc> <bog> EDGE_INFO <eog>`
        - [ ] For AIGs: Node types = {`PI`, `CONST`, `AND`}, Edge types = {`FWD`, `NOT`}
        - [ ] Final storage: Binary `.bin` files (memmap) for efficient loading
    - [ ] **Create AIG Tokenizer**
        - [x] Set up local `conda` env (`env.yml`) and install tokenizer dependencies
        - [ ] Run `pytest tests/test_setup.py` to validate `dataset/setup.py`
        - [x] Build vocabulary: special tokens + node types + edge types + IDX tokens
        - [x] Save tokenizer config for HuggingFace AutoTokenizer
        - [x] Run `python -m aig2pt.dataset.setup` to update stats and generate tokenizer files
    - [ ] **Data Processing Pipeline**
        - [x] Parse `.aig` files with `aigverse`
        - [x] Convert to text sequences and tokenize (prepare_aigs)
        - [x] Auto-tune block size (`prepare_aigs` updates `aig.yaml` + tokenizer config)
        - [ ] Serialize to PyTorch-ready `.bin` files for server-scale runs (optional rewrite)
    - [x] Verify data processing pipeline works on local samples (`pytest tests/test_prepare_aigs.py`)
    - [ ] Create scripts that can scale to server environment
    - [ ] **Training Pipeline Bring-up**
        - [x] Adapt `train.py` configs for `aig_prepared/` splits + new `block_size`
            - [x] Point `processed_data_dir`/`tokenizer_path` to local assets (expect 512 block)
            - [x] Ensure vocab size matches regenerated tokenizer
        - [x] Run a local smoke training pass on the prepared splits
        - [ ] Capture loss curves / sanity metrics for documentation
    - [ ] **Sampling & Evaluation Prep**
        - [ ] Wire `sampling_and_evaluation.py` to the AIG tokenizer + checkpoints
        - [ ] Ensure V.U.N. metric hooks run locally end-to-end
        - [ ] Document any metric gaps or follow-ups for baselines
    - [ ] **Training Readiness Checklist**
        - [x] Confirm `aig2pt/dataset/aig_prepared/` metadata matches expected block size
        - [ ] Review GPU/CPU resource requirements for smoke run
        - [ ] Outline logging/metrics capture (stdout, wandb optional)
        - [ ] Make ready for server deployment multi-GPU
        - [ ] prep server env
    - [ ] **Dependency Audit**
        - [x] `pip install` core runtime libs (`torch`, `transformers`, `tqdm`, `aigverse`, `numpy`, etc.)
        - [ ] Capture environment freeze (update `env.yml` / `requirements.txt` if needed)

### Near-Term Generation Roadmap
- [ ] **Prepare Unconditional Random Generation Corpus**
    - [ ] Finalize sampling script to draw randomized AIG variants for V.U.N. evaluation
    - [ ] Materialize corpus in text/token format using the updated tokenizer pipeline
- [ ] **Run V.U.N. Evaluation for AIG2PT**
    - [ ] Generate samples from current model checkpoint(s)
    - [ ] Compute Validity, Uniqueness, and Novelty metrics on the random corpus
- [ ] **Run V.U.N. Evaluation for Baselines**
    - [ ] Identify baseline generators to compare against (e.g., random heuristics, ABC flows)
    - [ ] Produce baseline sample sets and score them with the same V.U.N. tooling
- [ ] **Gate for Additional Datasets** *(waiting on optimization scripts to finish on AIGs)*
    - [ ] Resume D2/D3/D4 dataset preparation once optimization pipeline lands
    - [ ] Backfill processed artifacts onto server when scripts are ready

### Server Production (Future)
- _Note: Revisit these once the V.U.N. evaluations land and the optimization scripts become available._
- [ ] **1.1. The "Pre-training" Dataset (For Unconditional Generation)**
    - [ ] Use the refined sampling pipeline to scale random corpus generation to server scale.
    - [ ] Convert all resulting `.aig` files into graph-sequence format.

- [ ] **1.2. The "Property Prediction" Dataset (For QoR Analysis)**
    - [ ] Take every AIG from your D1 deliverable.
    - [ ] Use a script to run a standard synthesis tool (e.g., `abc -c "ps"`) and extract its area (nodes) and delay (levels).
    - [ ] **Deliverable (D2):** A `ppa_labels.csv` file mapping each AIG sequence (or a hash of it) to its `{area, delay}` (on server).
- [ ] **1.3. The "Functional Mapping" Dataset (TT <-> AIG)**
    - [ ] Iterate through the original (small) seed circuits.
    - [ ] For each circuit, use `abc` to generate its truth table and convert the circuit to your AIG sequence format.
    - [ ] **Deliverable (D3):** A `functional_pairs.txt` file (on server).
- [ ] **1.4. The "Supervised Resynthesis" Dataset (AIG -> Optimized AIG)**
    - [ ] For each synthesis run, save pairs of (input_AIG, optimized_AIG) where the optimized version is smaller but functionally equivalent.
    - [ ] Use synthesis tools to generate smaller/optimized versions of circuits.
    - [ ] Ensure you run a formal verification check (e.g., `abc &cec`) to guarantee functional equivalence before saving the pair.
    - [ ] **Deliverable (D4):** A `resynthesis_pairs.txt` file with (larger_input_graph, smaller_equivalent_graph) pairs (on server).

## Phase 2: Model Training & Evaluation

### Local Development & Validation
- [ ] **2.0. Pipeline Development**
    - [ ] Test data loading with local sample `.aig` files
    - [ ] Validate model architecture on small dataset
    - [ ] Test sampling and V.U.N. metric calculation
    - [ ] Ensure all scripts are server-ready (paths, dependencies, etc.)

### Server Training (Full Scale)
- [ ] **2.1. Task 0: Pre-train the Foundation Model**
    - [ ] **Model:** Decoder-only AIG2PT model.
    - [ ] **Data:** D1 (`pretrain_corpus.txt`) on server.
    - [ ] **Task:** Standard next-token prediction.
    - [ ] **Validation:** Use V.U.N. metrics (Validity, Uniqueness, Novelty) to prove the model has learned AIG grammar.
- [ ] **2.2. Task 1: Fine-Tune for Property Prediction**
    - [ ] **Model:** Load the pre-trained model from 2.1. Add a simple regression head on top.
    - [ ] **Data:** D2 (`ppa_labels.csv`) on server.
    - [ ] **Task:** Graph-level classification/regression.
    - [ ] **Validation:** Report Mean Absolute Error (MAE) on area and delay prediction.
- [ ] **2.3. Task 2 & 3: Fine-Tune for Functional Mapping**
    - [ ] **Model:** Create a new Encoder-Decoder (Seq2Seq) Transformer. Initialize the decoder's weights from the pre-trained model (2.1).
    - [ ] **Data:** D3 (`functional_pairs.txt`) on server.
    - [ ] **Tasks:**
        - [ ] Synthesis (TT->AIG): Train with TT as input, AIG as output.
        - [ ] Simulation (AIG->TT): Train with AIG as input, TT as output.
    - [ ] **Validation:** Report "% Functional Equivalence" on a test set.
- [ ] **2.4. Task 4: Fine-Tune for Supervised Resynthesis (Graph-to-Graph Optimization)**
    - [ ] **Model:** Use the same Encoder-Decoder architecture from 2.3.
    - [ ] **Data:** D4 (`resynthesis_pairs.txt`) - pairs of (input_AIG, smaller_optimized_AIG) on server.
    - [ ] **Task:** Graph-to-graph translation. Given an input AIG, generate a smaller functionally equivalent AIG.
    - [ ] **Training:** The model learns from ground truth pairs where the output graph is smaller than the input.
    - [ ] **Validation:**
        - [ ] Primary: "% Functional Equivalence" (verify outputs match input function).
        - [ ] Secondary: Size reduction metrics - histogram showing the size improvement (nodes/gates reduced).
        - [ ] Tertiary: Compare generated optimized AIGs to the ground truth optimized AIGs.

## Phase 3: Paper Structure

- [ ] **Introduction:** The problem is fragmented ML models. Your solution is a versatile foundation model.
- [ ] **Related Work:** Add missing ML baselines.
- [ ] **Methodology:**
    - [ ] 3.1. Pre-training AIG2PT.
    - [ ] 3.2. Fine-Tuning for Property Prediction.
    - [ ] 3.3. Fine-Tuning for Functional Mapping.
    - [ ] 3.4. Fine-Tuning for Supervised Resynthesis.
- [ ] **Experiments:**
    - [ ] 4.1. Pre-training Validation (Unconditional Gen, V.U.N. results).
    - [ ] 4.2. Downstream Task: Property Prediction Results.
    - [ ] 4.3. Downstream Task: Functional Mapping Results.
    - [ ] 4.4. Downstream Task: Supervised Resynthesis Results.
- [ ] **Conclusion:** You've demonstrated the first true foundation model for AIGs, unifying generation, analysis, synthesis, and optimization under one framework.

