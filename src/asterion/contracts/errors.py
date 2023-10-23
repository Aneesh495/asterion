class AsterionError(RuntimeError):
    """An invalid input or incomplete scientific obligation."""


class ContractError(AsterionError):
    pass


class IntegrityError(AsterionError):
    pass


class ResourceError(AsterionError):
    pass


class CancellationError(ResourceError):
    """An owned operation received an explicit operating-system cancellation."""


class LabelAccessError(AsterionError):
    pass
