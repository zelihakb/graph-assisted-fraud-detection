
from scripts.train_v2_logreg import load_split
from scripts.train_v2_rf import fit_model
from src.graph.features import FEATURE_COLUMNS


def main():
    # Graph-only model için yalnızca geçmiş özelliklerini kullandık.
    train = load_split("train")
    validation = load_split("validation")

    fit_model(
        "graph_only",
        train,
        list(FEATURE_COLUMNS),
        validation,
    )

    print("\nGRAPH-ONLY RF VALIDATION COMPLETED!")
    print("Test split was not accessed.")


if __name__ == "__main__":
    main()
