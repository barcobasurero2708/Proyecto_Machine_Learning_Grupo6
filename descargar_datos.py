"""Descarga oficial opcional. No sobrescribe archivos existentes.

Si ya tienes los cuatro archivos, basta con copiarlos a Data/ o data/raw/.
"""
import argparse
import csv
from pathlib import Path
from urllib.request import urlopen
from src.data import MONTHS, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(__file__).resolve().parent / "data" / "manifest.csv"
    with manifest.open(encoding="utf-8") as stream:
        expected = {r["archivo"]: r["sha256"] for r in csv.DictReader(stream)}
    for month in MONTHS:
        filename = f"yellow_tripdata_{month}.parquet"
        destination = args.data_dir / filename
        if not destination.exists():
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
            temporary = destination.with_suffix(".part")
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                if sha256(temporary) != expected[filename]:
                    raise ValueError(f"La fuente cambió: la huella de {filename} no coincide con la usada en esta entrega.")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(destination) != expected[filename]:
            raise ValueError(f"{filename}: contenido distinto del manifest. Revisar antes de reproducir las métricas.")
        print(f"Verificado: {filename}")


if __name__ == "__main__":
    main()
