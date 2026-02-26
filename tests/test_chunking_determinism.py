from ami_mom_pipeline.config import AppConfig
from ami_mom_pipeline.pipeline import normalize_text


def test_normalize_text_deterministic():
    s = " uh  We   should   do this  "
    assert normalize_text(s) == "We should do this"


def test_default_config_loads():
    cfg = AppConfig.load(None)
    assert cfg.pipeline.seed == 42

