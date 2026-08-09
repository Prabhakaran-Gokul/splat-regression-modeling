from srms.environments.sphere import SphereEnvironment
from srms.environments.torus import TorusEnvironment

ENVIRONMENTS = {
    "torus": TorusEnvironment,
    "sphere": SphereEnvironment,
}

__all__ = ["ENVIRONMENTS", "TorusEnvironment", "SphereEnvironment"]
