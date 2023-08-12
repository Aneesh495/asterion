import numpy as np
import pytest
import torch

from asterion.contracts.geometry import Geometry
from asterion.models.potential import ModelConfig, make_model


@pytest.fixture
def geometry():
    return Geometry(
        np.array([6, 1, 1, 8]),
        np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [-0.2, 1.0, 0.4], [0.0, -0.4, 1.3]]),
        "fixture",
    )


@pytest.fixture(scope="session", autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture(params=["invariant", "equivariant"])
def model(request):
    torch.manual_seed(911)
    return make_model(ModelConfig(kind=request.param, width=4, depth=2, radial=4)).double()
