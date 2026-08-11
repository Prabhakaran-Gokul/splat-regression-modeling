from srms.environments.hyperbolic import HyperbolicEnvironment
from srms.environments.so3 import SO3Environment
from srms.environments.sphere import SphereEnvironment
from srms.environments.torus import TorusEnvironment

ENVIRONMENTS = {
    "torus": TorusEnvironment,
    "sphere": SphereEnvironment,
    "so3": SO3Environment,
    "hyperbolic": HyperbolicEnvironment,
}

__all__ = ["ENVIRONMENTS", "HyperbolicEnvironment", "SO3Environment", "SphereEnvironment", "TorusEnvironment"]
