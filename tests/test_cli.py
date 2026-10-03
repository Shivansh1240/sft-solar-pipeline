import json

import numpy as np

from sft.cli import main


def test_verify_single_check(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert main(["verify", "--quick", "--check", "dipole_normalisation", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data[0]["name"] == "dipole_normalisation" and data[0]["passed"]
    assert "1/1 checks passed" in capsys.readouterr().out


def test_run_synthetic_writes_outputs(tmp_path):
    npz, png = tmp_path / "run.npz", tmp_path / "run.png"
    args = "run --cycles 1 --n-bmr 200 --n-lat 60 --dt 2".split()
    assert main([*args, "--out", str(npz), "--plot", str(png)]) == 0
    d = np.load(npz)
    assert d["butterfly"].shape == (60, len(d["t_yr"]))
    assert png.stat().st_size > 10_000


def test_run_catalogue(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("time_yr,lat_deg,flux_mx,tilt_deg,cycle\n0.1,15,3e21,8,25\n0.4,-12,3e21,6,25\n")
    assert main(["run", "--catalogue", str(csv), "--years", "1.0", "--n-lat", "60"]) == 0
