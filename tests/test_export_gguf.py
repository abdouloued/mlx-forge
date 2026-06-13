import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.export_gguf import generate_modelfile, export_gguf


def test_modelfile_contains_from_directive():
    mf = generate_modelfile(gguf_path="model.gguf", system_prompt="You are helpful.")
    assert "FROM ./model.gguf" in mf


def test_modelfile_contains_system_prompt():
    mf = generate_modelfile(gguf_path="model.gguf", system_prompt="You are a tool-calling assistant.")
    assert "You are a tool-calling assistant." in mf


def test_modelfile_contains_parameter_stop():
    mf = generate_modelfile(gguf_path="model.gguf", system_prompt="test")
    assert "PARAMETER" in mf


def test_modelfile_no_system_prompt():
    mf = generate_modelfile(gguf_path="model.gguf")
    assert "FROM ./model.gguf" in mf
    assert "SYSTEM" not in mf


def test_export_gguf_calls_subprocess(tmp_path):
    fused_dir = tmp_path / "fused"
    fused_dir.mkdir()
    out_gguf = tmp_path / "model.gguf"
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "convert_hf_to_gguf.py").write_text("")

    with patch("core.export_gguf.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        export_gguf(
            fused_path=str(fused_dir),
            output_gguf=str(out_gguf),
            llama_cpp_dir=str(llama_cpp_dir),
            quantization="Q4_K_M",
        )
        assert mock_run.call_count >= 1


def test_export_gguf_raises_on_convert_failure(tmp_path):
    fused_dir = tmp_path / "fused"
    fused_dir.mkdir()
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    (llama_cpp_dir / "convert_hf_to_gguf.py").write_text("")

    with patch("core.export_gguf.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        with pytest.raises(RuntimeError, match="GGUF conversion failed"):
            export_gguf(
                fused_path=str(fused_dir),
                output_gguf=str(tmp_path / "model.gguf"),
                llama_cpp_dir=str(llama_cpp_dir),
                quantization="Q4_K_M",
            )
