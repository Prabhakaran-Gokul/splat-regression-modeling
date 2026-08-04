from srms.methods.backends import mlp, srm

BACKENDS = {
    "srm": srm,
    "mlp": mlp,
    # "ntfields", "kan" — not yet implemented.
}

__all__ = ["BACKENDS", "srm", "mlp"]
