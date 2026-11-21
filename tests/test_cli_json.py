"""--json output: serializer and pure-JSON stdout guarantee."""

import json
import sys

import numpy as np
import pandas as pd

import fantasy_intel.cli as cli


def test_to_jsonable_handles_frames_numpy_and_nan():
    df = pd.DataFrame([{"a": np.int64(3), "b": np.nan, "c": np.float32(1.5)}])
    out = cli.to_jsonable({"frame": df, "arr": np.array([1, 2]), 7: np.bool_(True), "inf": float("inf")})
    assert out == {"frame": [{"a": 3, "b": None, "c": 1.5}], "arr": [1, 2], "7": True, "inf": None}
    json.dumps(out)


def test_json_mode_keeps_stdout_pure_when_libraries_print(monkeypatch, capfd):
    class NoisyAdvisor:
        def get_waiver_moves(self, **kwargs):
            print("2026 done.")  # what nfl_data_py does on refresh
            print("Downcasting floats.")
            return {"moves": [], "score": np.float64(1.25)}

    monkeypatch.setattr(cli, "FantasyAdvisor", NoisyAdvisor)
    monkeypatch.setattr(sys, "argv", ["fantasy-intel", "waiver-moves", "--json"])
    cli.main()
    out, err = capfd.readouterr()
    assert json.loads(out) == {"moves": [], "score": 1.25}
    assert "2026 done." in err and "Downcasting floats." in err
