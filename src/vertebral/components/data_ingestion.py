"""Stage 1 - Data ingestion.

The dataset ships as one CSV per class (Normal / Type_H / Type_S), each with an
identical header including the Class column. Ingestion stacks them row-wise into
a single combined CSV for the downstream stages.
"""

import pandas as pd

from vertebral.entity.config_entity import DataIngestionConfig
from vertebral.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self) -> str:
        frames = []
        for f in self.config.source_files:
            if not f.exists():
                raise FileNotFoundError(f"Source data not found: {f}")
            frames.append(pd.read_csv(f))

        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined = self._normalise_labels(combined)
        combined.to_csv(self.config.local_data_file, index=False)
        logger.info(
            "Ingested %d rows from %d files -> %s",
            len(combined),
            len(self.config.source_files),
            self.config.local_data_file,
        )
        return str(self.config.local_data_file)

    def _normalise_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map dirty label variants (typos/case) to the canonical classes.

        The raw files contain a few mislabelled rows (e.g. 'Nrmal', 'type_h',
        'tp_s'); we clean them here so validation can gate on canonical labels.
        Any label not covered by the alias map is left unchanged so validation
        still flags genuinely unknown classes.
        """
        target = self.config.target_column
        aliases = self.config.label_aliases
        if not aliases or target not in df.columns:
            return df

        def clean(value: object) -> object:
            key = str(value).strip().lower()
            return aliases.get(key, value)

        before = df[target].copy()
        df[target] = df[target].map(clean)
        n_fixed = int((before.astype(str) != df[target].astype(str)).sum())
        if n_fixed:
            logger.info("Normalised %d dirty target label(s).", n_fixed)
        return df
