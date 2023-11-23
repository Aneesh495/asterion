import hashlib

from asterion.artifacts.io import canonical


def state_fingerprint(model):
    """Bind the executed precision and every parameter/buffer value without serialization code."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
