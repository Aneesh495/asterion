class AsterionError(RuntimeError):
    """An invalid input or incomplete scientific obligation."""


class ContractError(AsterionError):
    pass


class IntegrityError(AsterionError):
    pass


class ResourceError(AsterionError):
    pass


class LabelAccessError(AsterionError):
    pass
