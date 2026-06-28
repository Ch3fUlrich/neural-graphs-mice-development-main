from ngmd import config as config_module
from ngmd.config import get_config_section, load_config


def test_load_config_reads_yaml_and_applies_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  data_dir: from_yaml
  models_dir: yaml_models
network_summaries:
  subsample_size: 42
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("NGMD_DATA", "from_env")
    monkeypatch.setenv("NGMD_MODELS", "env_models")

    config = load_config(config_path)

    assert config["paths"]["data_dir"] == "from_env"
    assert config["paths"]["models_dir"] == "env_models"
    assert config["network_summaries"]["subsample_size"] == 42


def test_get_config_section_overlays_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
network_summaries:
  subsample_size: null
""".strip(),
        encoding="utf-8",
    )

    section = get_config_section(
        "network_summaries",
        defaults={"num_subsample_trials": 20, "subsample_size": 250},
        config_path=config_path,
    )

    assert section == {"num_subsample_trials": 20, "subsample_size": None}


def test_default_config_path_can_be_found_from_child_directory(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    notebooks = repo / "notebooks"
    config_dir = repo / "configs"
    notebooks.mkdir(parents=True)
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "paths:\n  data_dir: local_data\n", encoding="utf-8"
    )
    monkeypatch.chdir(notebooks)
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATH",
        tmp_path / "missing" / "config.yaml",
    )

    assert load_config()["paths"]["data_dir"] == "local_data"
    assert config_module.Path.cwd() == notebooks
