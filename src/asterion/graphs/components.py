import torch

from asterion.contracts.errors import ContractError


def components(batch, edges, cutoff):
    """Deterministic union-find over active edges, including singleton components."""
    atoms = len(batch.numbers)
    parent = list(range(atoms))
    size = [1] * atoms

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    if edges.source.shape != edges.target.shape or edges.source.ndim != 1:
        raise ContractError("component edge axes disagree")
    if edges.distance.shape != edges.source.shape or not torch.isfinite(edges.distance).all():
        raise ContractError("component distance axes or finite values violated")
    if edges.source.numel() and (
        min(edges.source.min().item(), edges.target.min().item()) < 0
        or max(edges.source.max().item(), edges.target.max().item()) >= atoms
    ):
        raise ContractError("component edge index outside atom axis")
    if not torch.equal(batch.graph[edges.source], batch.graph[edges.target]):
        raise ContractError("component edge crosses a graph boundary")
    active = edges.distance.detach() < cutoff
    source = edges.source[active].detach().cpu().tolist()
    target = edges.target[active].detach().cpu().tolist()
    for first, last in zip(source, target):
        a, b = root(first), root(last)
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    groups: dict[int, list[int]] = {}
    for index in range(atoms):
        groups.setdefault(root(index), []).append(index)
    return tuple(tuple(group) for group in sorted(groups.values(), key=lambda group: group[0]))
