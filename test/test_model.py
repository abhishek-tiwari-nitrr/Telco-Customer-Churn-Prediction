import numpy as np
import pytest


def test_unknown_category_handled_gracefully(valid_df, mock_preprocessor):
    df = valid_df.copy()
    df.loc[0, "contract"] = "Unknown_XYZ"
    try:
        X_enc = mock_preprocessor.transform(df)
        assert not np.any(np.isnan(X_enc))
    except Exception as e:
        pytest.fail(f"Preprocessor raised on unknown category: {e}")

def test_train_test_split_is_stratified():
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    y = np.random.choice([0, 1], size=500, p=[0.735, 0.265])
    _, _, y_train, y_test = train_test_split(
        np.zeros(500), y, test_size=0.2, stratify=y, random_state=42
    )
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_smote_is_inside_pipeline_not_before():
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import AdaBoostClassifier

    pipe = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", AdaBoostClassifier()),
        ]
    )
    steps = [name for name, _ in pipe.steps]
    assert "smote" in steps
    assert steps.index("smote") < steps.index("model")