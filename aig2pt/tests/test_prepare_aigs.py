"""Integration test for the AIG preparation pipeline."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
def test_prepare_aigs_pipeline(tmp_path):
    """Ensure `aig2pt.dataset.prepare_aigs` runs end-to-end locally."""

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "aig2pt" / "configs" / "aig.yaml"
    tokenizer_config_path = repo_root / "aig2pt" / "dataset" / "tokenizer" / "tokenizer_config.json"

    original_config_text = config_path.read_text()
    config_data = yaml.safe_load(original_config_text)
    original_tokenizer_config = tokenizer_config_path.read_text()

    output_dir = tmp_path / "prepared_aigs"
    config_data["processed_data_dir"] = str(output_dir)

    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False))

    try:
        subprocess.run(
            [sys.executable, "-m", "aig2pt.dataset.prepare_aigs"],
            cwd=str(repo_root),
            check=True,
        )

        meta_path = output_dir / "data_meta.json"
        assert meta_path.exists(), "Metadata file not generated"

        metadata = json.loads(meta_path.read_text())
        assert metadata, "Metadata is empty"
        assert metadata.get("train", {}).get("num_graphs", 0) > 0, "Train split missing graphs"

        overall = metadata.get("overall", {})
        assert overall.get("max_sequence_length", 0) > 0, "Overall max sequence length missing"
        recommended_block = overall.get("recommended_block_size")
        assert recommended_block, "Recommended block size missing from metadata"

        updated_config = yaml.safe_load(config_path.read_text())
        assert updated_config.get("block_size") == recommended_block

        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        assert tokenizer_config.get("model_max_length") == recommended_block

    finally:
        config_path.write_text(original_config_text)
        tokenizer_config_path.write_text(original_tokenizer_config)
        if output_dir.exists():
            shutil.rmtree(output_dir)
