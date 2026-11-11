# Code Review — Vertebral Column Condition (KNN)

Reviewer-facing deep-dive on the correctness findings in this repo (vertebral
column; 3 per-class CSVs stacked; dirty labels normalized via schema
`label_aliases`; `LabelEncoder`; scaled KNN; macro metrics + one-vs-rest ROC-AUC;
FastAPI `/predict`) and the fixes that landed. Each finding gives **Where**,
**Root cause**, **Fix** (before/after from the actual commit), and the
**regression test** that locks it. Commits referenced: `9aed679`, `7dc8ddb`,
`4752d53`.

The headline is finding #1: a **confirmed** probability-mislabeling bug at
serving. The ML core (split-before-fit, train-only scaling, held-out macro
metrics) was audited and found correct — see "Verified correct, not changed".

---

## 1. CONFIRMED BUG — per-class probabilities labeled by the wrong class order

**Where:** `src/vertebral/pipeline/prediction.py` — `PredictionPipeline.predict()`
(commit `4752d53`).

**Root cause:** the response zips `predict_proba(df)[0]` with
`label_encoder.classes_`:

```python
proba = self.model.predict_proba(df)[0]
class_names = [str(c) for c in self.label_encoder.classes_]
probabilities = {name: round(float(p), 4) for name, p in zip(class_names, proba)}
```

`predict_proba` returns columns ordered by **`model.classes_`** (the integer
codes the *classifier* was actually trained on). `label_encoder.classes_` is the
full set of class names the *encoder* ever saw, in sorted order. These two only
coincide when **every** class is present in the training split. If a class is
absent from training (small per-class files, an aggressive split, or a filtered
dataset), `model.classes_` is a *subset* — say `[0, 2]` — while
`label_encoder.classes_` is still `["Normal", "Type_H", "Type_S"]`. The zip then
pairs proba column 0 with "Normal" (ok) but column 1 (the model's class `2` =
"Type_S") with "Type_H" — **every probability after the gap is mislabeled**, and
the returned dict even advertises a class the model can't predict. The scalar
`prediction` field is fine (it decodes a single index correctly); only the
per-class `probabilities` map is wrong.

**Fix:** decode the **model's own** class order and zip against that:

```python
# before
        class_names = [str(c) for c in self.label_encoder.classes_]

# after
        # Label each probability by decoding the MODEL's own class order
        # (model.classes_), not label_encoder.classes_ — these only coincide
        # when every class is present in training, so decode the model's order
        # to stay correct even if a class is absent.
        class_names = [
            str(c) for c in self.label_encoder.inverse_transform(self.model.classes_)
        ]
```

`label_encoder.inverse_transform(model.classes_)` maps each proba column's
integer code back to its name, guaranteeing column ↔ label alignment regardless
of which classes made it into training.

**Trade-off / assumption:** this relies on `model.classes_` being the
*encoder-space* integer codes — which holds because the pipeline is trained on the
`LabelEncoder`-transformed target, so `model.classes_ ⊆ {0,1,2}` and
`inverse_transform` is always valid. If a future refactor trained the classifier
on raw string labels, `inverse_transform` would be wrong and should be dropped;
the two must be kept in the same space.

**Regression test:** `tests/test_prediction.py` —
`test_probabilities_labeled_by_model_class_order`. It builds a tiny in-memory KNN
trained on only two of the three classes (codes `0` and `2`, "Type_H" absent)
with an encoder that knows all three, then asserts the probability keys are
exactly the two present classes, correctly named — the old code would have
emitted "Type_H" and mislabeled the second column:

```python
le = LabelEncoder().fit(["Normal", "Type_H", "Type_S"])
model = KNeighborsClassifier(n_neighbors=1).fit(X, [0, 2])  # Type_H absent
...
out = p.predict({c: 0 for c in FEATURES})
assert set(out["probabilities"]) == {"Normal", "Type_S"}
assert "Type_H" not in out["probabilities"]
assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
```

---

## 2. Model artifact loaded twice; OvR ROC-AUC silently omitted on failure

**Where:** `src/vertebral/components/model_evaluation.py` — `evaluate()` and
`_log_to_mlflow()` (commit `9aed679`).

**Root cause (a):** the model was deserialized in `evaluate()` and again in
`_log_to_mlflow()` — two disk reads.

**Root cause (b):** the multiclass one-vs-rest ROC-AUC was wrapped in
`try/except` that, on failure (e.g. a class absent from the test split, which
makes OvR AUC undefined), logged a warning and **omitted the key entirely**. A
downstream consumer couldn't tell "AUC not computable this run" from "someone
forgot to write it" — the two look identical when the key is simply missing.

**Fix:** load once and pass `model` through; and in the `except`, record
`roc_auc_ovr: null` so the key is always present with an explicit sentinel. `None`
metrics are filtered before `mlflow.log_metrics`.

```python
# before
        except Exception as exc:  # pragma: no cover - metric best-effort
            logger.warning("Could not compute ROC-AUC: %s", exc)

        ...
        self._log_to_mlflow(metrics)

    def _log_to_mlflow(self, metrics: dict) -> None:
        ...
            mlflow.log_metrics(metrics)
            try:
                model = load_object(self.config.model_file)   # second disk read

# after
        except Exception as exc:  # pragma: no cover - metric best-effort
            logger.warning("Could not compute ROC-AUC: %s", exc)
            metrics["roc_auc_ovr"] = None

        ...
        self._log_to_mlflow(metrics, model)

    def _log_to_mlflow(self, metrics: dict, model) -> None:
        ...
            # MLflow metrics must be numeric; drop any None (e.g. undefined AUC).
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            try:
                mlflow.sklearn.log_model(model, name="model")
```

**Regression test:** the end-to-end test asserts `accuracy` and `f1` are in
`[0, 1]` on the real three-class data (all classes present, so AUC is computed);
the `null` branch is the documented sentinel for the degenerate case.

---

## 3. `predict()` raised a bare `KeyError` on a missing feature

**Where:** `src/vertebral/pipeline/prediction.py` — `predict()` (commit `4752d53`).

**Root cause:** a measurement dict missing one of the six features hit a
`KeyError` inside the scaler's `ColumnTransformer`, surfacing as an opaque 500.

**Fix:** validate the expected feature columns (the six `numeric_columns`) up
front with a clear `ValueError`, then reindex to the training order.

```python
# before
        df = pd.DataFrame([features])
        pred_idx = int(self.model.predict(df)[0])

# after
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required feature field(s): {missing}")
        # Reindex to the training feature order (extra keys ignored).
        df = pd.DataFrame([features]).reindex(columns=list(self.feature_columns))
        pred_idx = int(self.model.predict(df)[0])
```

**Regression test:** `tests/test_prediction.py` —
`test_missing_feature_raises_clear_error`, bypassing `__init__` so no model is
needed; it drops `S_Degree` and asserts the error names it.

---

## 4. `LabelEncoder` fit on the full target before the split — comment only (leakage-free)

**Where:** `src/vertebral/components/data_transformation.py` — `transform()`
(commit `7dc8ddb`).

**Observation:** `label_encoder.fit_transform(df[target])` runs on the **whole**
target column before `train_test_split`. At a glance this looks like the classic
"fit on all data" leakage smell.

**Decision — comment, not a refactor.** It is genuinely leakage-free: a
`LabelEncoder` only maps class *names* to sorted integer codes — a deterministic
relabeling that learns **nothing** from the feature distribution or from the row
split, and never touches test *features*. The transforms that *do* learn from
data (the imputer and scaler) are already fit on the training split only, below.
Refactoring to fit the encoder on train only would add no correctness and would
*introduce* a failure mode: an unseen label in the test split would crash
`inverse_transform`. So we documented the reasoning in place:

```python
        # NOTE: this fits on the full target BEFORE the split, which is leakage-
        # free here: LabelEncoder only maps class *names* to sorted integer codes
        # (a deterministic relabeling), learning nothing from the feature
        # distribution or the row split. It does not peek at test *features*. The
        # scaler/imputer (which do learn from data) are fit on the train split
        # only, below. ...
```

**Regression test:** none — documentation-only; the end-to-end test confirms the
pipeline trains and the encoder round-trips at serving.

---

## Verified correct, not changed (the ML core)

- **Split before fit, scale on train only.** `transform()` encodes the label,
  splits, then fits the scaler/imputer `ColumnTransformer` on the training split
  only — essential for a distance-based KNN and **leakage-free**.
- **Stratified, reproducible split.** `stratify=df[target]` with a fixed
  `random_state` keeps the three classes balanced and the split deterministic.
- **Label hygiene before validation.** Dirty label variants are normalized via
  schema `label_aliases` in ingestion, *before* the validation gate checks the
  canonical class set — so the gate sees clean labels and the encoder learns the
  intended three classes.
- **Held-out macro metrics.** Evaluation reads `test.csv` and computes
  macro-averaged precision/recall/f1 (correct for a balanced multiclass problem)
  plus OvR ROC-AUC on probabilities.
- **Scalar prediction decoding was already correct.** `predict()` decodes the
  single predicted index via `label_encoder.inverse_transform([pred_idx])` — only
  the per-class probability *map* was mislabeled (finding #1).
