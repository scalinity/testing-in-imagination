"""Private locations from the untracked .env; the baseline work directory is read-only."""
from dataclasses import dataclass
import os
from pathlib import Path

from tim.paths import OWN, ROOT

REQUIRED = ("CDM_DATA_DIR", "CDM_WORK_DIR", "TIM_WORK_DIR", "TIM_ADMISSION_DIR")


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


@dataclass(frozen=True)
class Settings:
    source: str
    data_dir: Path
    baseline_dir: Path
    work_dir: Path
    admission_dir: Path
    runtime: str = "blackwell"

    @property
    def synthetic(self) -> bool:
        return self.source == "synthetic"

    def baseline(self):
        """Replication settings for the baseline work directory (used only through readers)."""
        from replication.config import Settings as ReplicationSettings
        return ReplicationSettings(source=self.source, data_dir=self.data_dir,
                                   work_dir=self.baseline_dir, runtime=self.runtime)

    @classmethod
    def load(cls, env_file: Path | None = None, environ: dict | None = None):
        from replication.config import read_env
        values = read_env(env_file or OWN / ".env") | dict(os.environ if environ is None else environ)
        missing = [name for name in REQUIRED if not values.get(name)]
        if missing:
            raise ValueError("set " + ", ".join(missing) + " in the untracked .env")
        source = values.get("CDM_DATA_SOURCE", "synthetic")
        if source not in ("synthetic", "mimic"):
            raise ValueError("CDM_DATA_SOURCE must be synthetic or mimic")
        runtime = values.get("CDM_RUNTIME", "blackwell")
        if runtime not in ("pinned", "blackwell"):
            raise ValueError("CDM_RUNTIME must be pinned or blackwell")
        data, baseline, work, admissions = (Path(values[name]).expanduser().resolve() for name in REQUIRED)
        root = ROOT.resolve()
        for path in (data, baseline, work, admissions):
            if _inside(path, root):
                raise ValueError("data, baseline and runtime directories must stay outside the repository")
        for path in (work, admissions):
            if _inside(path, baseline) or _inside(baseline, path):
                raise ValueError("TIM directories must not overlap the read-only baseline work directory")
        if _inside(admissions, work) or _inside(work, admissions):
            raise ValueError("the admission directory must not overlap TIM_WORK_DIR")
        return cls(source=source, data_dir=data, baseline_dir=baseline, work_dir=work,
                   admission_dir=admissions, runtime=runtime)
