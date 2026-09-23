"""Ejecuta las celdas en orden con IPython y guarda sus salidas, sin servidor.

Uso: python ejecutar_notebook.py --data-dir Data
También puede abrirse el notebook normalmente en VS Code o Jupyter.
"""
import argparse
import os
from pathlib import Path
import time
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    root = Path(__file__).resolve().parent
    path = root / "notebooks" / "01_exploracion_inicial.ipynb"
    os.environ["TAXI_DATA_DIR"] = str(data_dir)
    os.chdir(root)
    nb = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    start = time.perf_counter()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        print(f"Celda {count}: {cell.source.splitlines()[0][:90]}", flush=True)
        with capture_output() as captured:
            execution = shell.run_cell(cell.source, store_history=True)
        outputs = []
        for stream, text in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
            if text:
                outputs.append(nbformat.v4.new_output("stream", name=stream, text=text))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        cell.execution_count = count
        cell.outputs = outputs
        if execution.error_before_exec or execution.error_in_exec:
            nbformat.write(nb, path)
            raise RuntimeError(captured.stdout + captured.stderr)
    nb.metadata["verificacion"] = {"metodo": "IPython secuencial en un proceso, sin servidor",
                                     "celdas_ejecutadas": count, "abril_evaluado": False}
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(f"Notebook ejecutado: {count} celdas, {time.perf_counter() - start:.1f} s.")


if __name__ == "__main__":
    main()
