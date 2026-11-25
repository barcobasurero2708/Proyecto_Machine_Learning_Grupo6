"""Ejecuta los notebooks en orden con IPython y guarda sus salidas, sin servidor.

Uso:
    python ejecutar_notebook.py --data-dir Data                # los cuatro notebooks
    python ejecutar_notebook.py --data-dir Data --notebooks 03_modelos.ipynb

Los notebooks 02-04 leen los resultados que deja `python main.py`; ejecútalo antes.
También pueden abrirse normalmente en VS Code o Jupyter.
"""
import argparse
import os
import time
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

NOTEBOOKS = ["01_exploracion_inicial.ipynb", "02_limpieza_features.ipynb",
             "03_modelos.ipynb", "04_analisis_errores.ipynb"]


def run(path: Path) -> int:
    nb = nbformat.read(path, as_version=4)
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        print(f"  [{path.name}] celda {count}: {cell.source.splitlines()[0][:80]}", flush=True)
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
                                     "celdas_ejecutadas": count}
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--notebooks", nargs="*", default=NOTEBOOKS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.environ["TAXI_DATA_DIR"] = str(args.data_dir.resolve())
    os.chdir(root)
    for name in args.notebooks:
        start = time.perf_counter()
        n = run(root / "notebooks" / name)
        print(f"{name}: {n} celdas en {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
