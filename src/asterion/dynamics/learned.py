import numpy as np

from asterion.contracts.geometry import Chemistry, Geometry, collate
from asterion.dynamics.analytic import ForceResult
from asterion.graphs.radius import NeighborCache


class LearnedEvaluator:
    def __init__(self, model, chemistry=Chemistry()):
        self.model = model.eval()
        self.chemistry = chemistry
        self.cache = NeighborCache(model.config.cutoff)

    def __call__(self, positions: np.ndarray, numbers: np.ndarray):
        geometry = Geometry(numbers, positions, "dynamics", self.chemistry).validate()
        parameter = next(self.model.parameters())
        batch = collate([geometry], dtype=parameter.dtype, device=parameter.device)
        batch.positions.requires_grad_(True)
        edges = self.cache(batch)
        energy, force = self.model(batch, edges=edges)
        return ForceResult(
            float(energy.item()),
            force.detach().cpu().numpy(),
            neighbor_rebuilds=self.cache.rebuilds,
        )
