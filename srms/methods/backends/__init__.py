from srms.methods.backends import srm

BACKENDS = {
    "srm": srm,
    # "ntfields", "mlp", "kan" — not yet implemented.
}

__all__ = ["BACKENDS", "srm"]
