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


class CommitteeEvaluator:
    """The committee mean remains the gradient of its mean scalar energy."""

    def __init__(self, committee, calibration, chemistry=Chemistry()):
        if tuple(calibration.committee) != tuple(committee.identities):
            from asterion.contracts.errors import IntegrityError

            raise IntegrityError("dynamics calibration belongs to another committee")
        self.committee = committee
        self.calibration = calibration
        self.chemistry = chemistry

    def __call__(self, positions, numbers):
        geometry = Geometry(numbers, positions, "committee-dynamics", self.chemistry).validate()
        output = self.committee.infer([geometry])
        bound = float(self.calibration.bounds(output.force_score)[0])
        return ForceResult(
            float(output.energy_mean[0]),
            output.forces_mean,
            uncertainty=bound,
            uncertainty_available=True,
            uncertainty_quantity="calibrated-mae-bound",
        )
