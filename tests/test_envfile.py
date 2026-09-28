from masdar.envfile import load_env_file


def test_sets_only_unset_names_and_reports_names_not_values(tmp_path, monkeypatch):
    monkeypatch.delenv("MASDAR_TEST_A", raising=False)
    monkeypatch.delenv("MASDAR_TEST_Q", raising=False)
    monkeypatch.setenv("MASDAR_TEST_B", "from-environment")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "MASDAR_TEST_A=alpha # trailing note\n"
        "export MASDAR_TEST_B=from-file\n"
        "MASDAR_TEST_Q=\"quoted value\"\n"
        "not a line\n"
        "MASDAR_TEST_EMPTY=\n",
        encoding="utf-8",
    )
    loaded = load_env_file(env)
    import os

    assert os.environ["MASDAR_TEST_A"] == "alpha"
    assert os.environ["MASDAR_TEST_B"] == "from-environment"
    assert os.environ["MASDAR_TEST_Q"] == "quoted value"
    assert "MASDAR_TEST_EMPTY" not in os.environ
    assert sorted(loaded) == ["MASDAR_TEST_A", "MASDAR_TEST_Q"]
    monkeypatch.delenv("MASDAR_TEST_A")
    monkeypatch.delenv("MASDAR_TEST_Q")


def test_a_missing_file_is_nothing(tmp_path):
    assert load_env_file(tmp_path / "absent.env") == []
