from srms.methods.backends import mlp, mlp_raw, mlp_resnet, srm

BACKENDS = {
    "srm": srm,
    "mlp": mlp,
    "mlp_raw": mlp_raw,  # ablation: mlp.py without Fourier features, see mlp_raw.py
    "mlp_resnet": mlp_resnet,  # ablation: mlp.py + residual hidden layers, see mlp_resnet.py
    # "ntfields", "kan" — not yet implemented.
}

__all__ = ["BACKENDS", "srm", "mlp", "mlp_raw", "mlp_resnet"]
