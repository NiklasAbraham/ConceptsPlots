#!/usr/bin/env python3
"""
Script to compile all standalone TikZ plots in the single_plots directory.
"""

import os
import subprocess
from pathlib import Path


def compile_tex_file(tex_file, output_dir=None):
    """
    Compile a single .tex file using pdflatex.

    Parameters:
    tex_file: Path to the .tex file
    output_dir: Directory for output (default: same as tex_file)
    """
    tex_path = Path(tex_file)

    if not tex_path.exists():
        print(f"Error: File {tex_file} does not exist")
        return False

    # Change to the directory containing the tex file
    work_dir = tex_path.parent
    tex_name = tex_path.name

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "pdflatex",
            "-output-directory",
            str(output_path),
            "-interaction=nonstopmode",
            tex_name,
        ]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", tex_name]

    try:
        # Run pdflatex twice to resolve references
        result1 = subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=60
        )

        result2 = subprocess.run(
            cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=60
        )

        if result1.returncode == 0 and result2.returncode == 0:
            print(f"✓ Successfully compiled {tex_name}")
            return True
        else:
            print(f"✗ Error compiling {tex_name}")
            if result1.stderr:
                print(f"  Error: {result1.stderr}")
            if result2.stderr:
                print(f"  Error: {result2.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout compiling {tex_name}")
        return False
    except Exception as e:
        print(f"✗ Exception compiling {tex_name}: {e}")
        return False


def main():
    """Main function to compile all plots."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Find all .tex files starting with numbers (001-008)
    tex_files = sorted(script_dir.glob("[0-9][0-9][0-9]_*.tex"))

    if not tex_files:
        print("No numbered .tex files found in the current directory")
        return

    print(f"Found {len(tex_files)} .tex files to compile\n")

    success_count = 0
    failed_files = []

    for tex_file in tex_files:
        print(f"Compiling {tex_file.name}...")
        if compile_tex_file(tex_file):
            success_count += 1
        else:
            failed_files.append(tex_file.name)
        print()

    # Summary
    print("=" * 50)
    print("Compilation Summary:")
    print(f"  Total files: {len(tex_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    # Clean up auxiliary files
    print("\nCleaning up auxiliary files...")
    aux_extensions = [".aux", ".log", ".out"]
    for ext in aux_extensions:
        for aux_file in script_dir.glob(f"*{ext}"):
            try:
                aux_file.unlink()
                print(f"  Removed {aux_file.name}")
            except Exception as e:
                print(f"  Could not remove {aux_file.name}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
