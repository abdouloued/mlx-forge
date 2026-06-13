from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console

console = Console()


def generate_modelfile(gguf_path: str, system_prompt: Optional[str] = None) -> str:
    """Generate an Ollama Modelfile for the given GGUF file."""
    gguf_name = Path(gguf_path).name
    lines = [f"FROM ./{gguf_name}", ""]
    if system_prompt:
        lines += [f'SYSTEM """{system_prompt}"""', ""]
    lines += [
        "PARAMETER stop <|im_end|>",
        "PARAMETER stop <|eot_id|>",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
    ]
    return "\n".join(lines)


def export_gguf(
    fused_path: str,
    output_gguf: str,
    llama_cpp_dir: str,
    quantization: str = "Q4_K_M",
    system_prompt: Optional[str] = None,
) -> None:
    """
    Convert fused safetensors → GGUF and write an Ollama Modelfile.

    Requires llama.cpp built locally at `llama_cpp_dir`. Verify
    convert_hf_to_gguf.py path and llama-quantize binary name against
    your llama.cpp version before use.
    """
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    raw_gguf = Path(output_gguf).with_suffix(".f16.gguf")

    cmd_convert = [
        sys.executable, str(convert_script),
        fused_path,
        "--outfile", str(raw_gguf),
        "--outtype", "f16",
    ]
    console.print(f"[bold yellow]Converting to GGUF:[/] {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed with exit code {result.returncode}")

    quantize_bin = Path(llama_cpp_dir) / "llama-quantize"
    cmd_quantize = [str(quantize_bin), str(raw_gguf), output_gguf, quantization]
    console.print(f"[bold yellow]Quantizing:[/] {' '.join(cmd_quantize)}")
    result = subprocess.run(cmd_quantize, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed with exit code {result.returncode}")

    modelfile_path = Path(output_gguf).parent / "Modelfile"
    modelfile_path.write_text(generate_modelfile(output_gguf, system_prompt))
    console.print(f"[bold yellow]Modelfile written:[/] {modelfile_path}")
    console.print(
        f"\n[bold]To register with Ollama:[/]\n"
        f"  cd {Path(output_gguf).parent}\n"
        f"  ollama create <your-model-name> -f Modelfile\n"
        f"  ollama run <your-model-name>"
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export fused model to GGUF for Ollama")
    parser.add_argument("--fused-path", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument("--llama-cpp-dir", required=True)
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()
    export_gguf(
        fused_path=args.fused_path,
        output_gguf=args.output_gguf,
        llama_cpp_dir=args.llama_cpp_dir,
        quantization=args.quantization,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
