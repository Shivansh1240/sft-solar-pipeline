"""Every named verification check must pass (quick grids keep this fast)."""

import pytest

from sft import verification


@pytest.mark.parametrize("name", sorted(verification.CHECKS))
def test_check_passes(name):
    result = verification.CHECKS[name](True)
    assert result.passed, f"{name}: {result.metric} = {result.value:.3g}, needs {result.criterion}"


def test_fitted_order_recovers_power_law():
    h = [1.0, 0.5, 0.25, 0.125]
    assert verification.fitted_order(h, [3.0 * x**2 for x in h]) == pytest.approx(2.0)


def test_results_serialise_to_json():
    import json

    res = verification.CHECKS["dipole_normalisation"](True)
    json.dumps(res.to_dict())
