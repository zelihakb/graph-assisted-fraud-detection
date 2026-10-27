
import argparse
import json
from getpass import getpass
from pathlib import Path
from time import perf_counter

from neo4j import GraphDatabase

from src.data.full_history import iter_scoped_paysim
from src.graph.neo4j_import import import_graph_chunk
from src.graph.schema import ensure_graph_constraints
from scripts.neo4j_real_sample import get_counts


CSV_PATH = Path("data/paysim.csv")
CHECKPOINT_PATH = Path(".local/neo4j_v2_import.json")
URI = "bolt://127.0.0.1:17687"

BATCH_SIZE = 1_000
EXPECTED_TOTAL = 2_770_409


def source_info():
    # Checkpoint'i kullandığımız CSV dosyasına bağladık.
    return {
        "path": str(CSV_PATH.resolve()),
        "size": CSV_PATH.stat().st_size,
    }


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return 0

    state = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))

    if state["source"] != source_info():
        raise RuntimeError("Checkpoint farklı bir CSV dosyasına ait.")

    completed = state["completed"]

    if not isinstance(completed, int) or completed < 0:
        raise ValueError("Geçersiz checkpoint.")

    return completed


def save_checkpoint(completed):
    # Başarılı batch sonrasındaki ilerlemeyi atomik biçimde kaydettik.
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "source": source_info(),
        "completed": completed,
    }

    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    temporary.replace(CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit pozitif olmalı.")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    completed = load_checkpoint()

    if args.limit is not None and completed > args.limit:
        raise ValueError("Checkpoint verilen limitten ileride.")

    print(f"Checkpoint: {completed:,} işlem")
    password = getpass("V2 Neo4j password: ")

    with GraphDatabase.driver(
        URI, auth=("neo4j", password)
    ) as driver:
        driver.verify_connectivity()

        with driver.session(database="neo4j") as session:
            ensure_graph_constraints(session)

            before = get_counts(session)

            # Kayıtlı ilerlemenin veritabanında mevcut olduğunu kontrol ettik.
            if before["transactions"] < completed:
                raise RuntimeError(
                    "Veritabanı checkpoint'in gerisinde. "
                    "Aktarımı güvenli biçimde durdurduk."
                )

            if (
                before["sent"] != before["transactions"]
                or before["to"] != before["transactions"]
            ):
                raise RuntimeError("Mevcut graph ilişkileri eksik.")

            started = perf_counter()
            submitted = 0
            seen = 0

            for chunk in iter_scoped_paysim(
                CSV_PATH, chunk_size=50_000
            ):
                chunk_end = seen + len(chunk)

                # Daha önce tamamladığımız işlemleri atladık.
                if chunk_end <= completed:
                    seen = chunk_end
                    continue

                position = max(0, completed - seen)

                while position < len(chunk):
                    if args.limit is not None and completed >= args.limit:
                        break

                    take = min(BATCH_SIZE, len(chunk) - position)

                    if args.limit is not None:
                        take = min(take, args.limit - completed)

                    batch = chunk.iloc[position:position + take]

                    imported = import_graph_chunk(
                        session, batch, batch_size=BATCH_SIZE
                    )

                    if imported != len(batch):
                        raise RuntimeError("Eksik batch aktarımı.")

                    completed += imported
                    submitted += imported
                    position += imported

                    # Veritabanına yazma başarılı olduktan sonra kaydettik.
                    save_checkpoint(completed)

                    if completed % 5_000 == 0:
                        elapsed = perf_counter() - started
                        rate = submitted / elapsed if elapsed else 0

                        print(
                            f"İlerleme: {completed:,} işlem "
                            f"| Bu oturum: {rate:.0f} tx/sn"
                        )

                seen = chunk_end

                if args.limit is not None and completed >= args.limit:
                    break

            expected = (
                args.limit
                if args.limit is not None
                else EXPECTED_TOTAL
            )

            if completed != expected:
                raise RuntimeError(
                    f"Eksik aktarım: {completed:,} / {expected:,}"
                )

            counts = get_counts(session)

            if any(
                counts[key] != completed
                for key in ("transactions", "sent", "to")
            ):
                raise RuntimeError(
                    f"Graph sayıları uyuşmuyor: {counts}"
                )

            print("\nGraph:", counts)
            print(f"Checkpoint: {completed:,}")
            print("Import verification passed!")


if __name__ == "__main__":
    main()
