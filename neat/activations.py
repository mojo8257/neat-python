"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones.

This version adds explicit implementations for the activation
functions requested for CPPNs in the project, namely:
    * abs      – absolute‐value non‑linearity (already present)
    * sigmoid  – logistic curve (already present)
    * gauss    – Gaussian (already present)
    * linear   – identity mapping (new alias `linear` → `linear_activation`)
    * sin      – sine (already present)
    * step     – binary step (new)
    * ramp     – saturated linear ramp (new)
    * tanh     – hyperbolic tangent (already present)

The ActivationFunctionSet now registers the new aliases so that they can
be referenced directly in configuration files, e.g. ``activation = step``.
 """

import math
import types


# --------------------------------------------------------------------------------------
# Core activation primitives (existing + new)
# --------------------------------------------------------------------------------------

def sigmoid_activation(z):
    """Scaled logistic curve with soft saturation outside ±60/5."""
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    """Scaled tanh for slightly steeper slope than the raw function."""
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)


# ------------------------------ NEW FUNCTIONS ------------------------------------------

def linear_activation(z):
    """Simple identity mapping (alias for *linear*)."""
    return z


def step_activation(z):
    """Binary step: returns 1 when *z* > 0, else 0."""
    return 1.0 if z > 0.0 else 0.0


def ramp_activation(z):
    """Saturated linear ramp clipped to [-1, 1].

    Values grow linearly in the interval [-1, 1] and saturate outside.
    This implementation matches the *ramp* function used in many NEAT
    reference implementations.
    """
    if z < -1.0:
        return -1.0
    if z > 1.0:
        return 1.0
    return z

# --------------------------------------------------------------------------------------
# Additional existing activation functions (unchanged)
# --------------------------------------------------------------------------------------


def relu_activation(z):
    return z if z > 0.0 else 0.0


def elu_activation(z):
    return z if z > 0.0 else math.exp(z) - 1


def lelu_activation(z):
    leaky = 0.005
    return z if z > 0.0 else leaky * z


def selu_activation(z):
    lam = 1.0507009873554805
    alpha = 1.6732632423543772
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)


def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log1p(math.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    try:
        return 1.0 / z
    except ArithmeticError:  # divide‑by‑zero or overflow
        return 0.0


def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


# --------------------------------------------------------------------------------------
# Utility helpers and activation function registry
# --------------------------------------------------------------------------------------

class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    if not isinstance(function, (types.BuiltinFunctionType,
                                 types.FunctionType,
                                 types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")

    # Exactly one positional argument expected.
    if function.__code__.co_argcount != 1:
        raise InvalidActivationFunction("A single‑argument function is required.")


class ActivationFunctionSet(object):
    """Registry of activation functions available to genomes.

    Users may add their own functions at runtime via :py:meth:`add`.
    """

    def __init__(self):
        self.functions = {}

        # ---- Core set ----
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)

        # ---- Newly added / aliases requested for CPPNs ----
        self.add('linear', linear_activation)   # explicit alias
        self.add('step', step_activation)
        self.add('ramp', ramp_activation)

        # ---- Additional pre‑existing functions ----
        self.add('relu', relu_activation)
        self.add('elu', elu_activation)
        self.add('lelu', lelu_activation)
        self.add('selu', selu_activation)
        self.add('softplus', softplus_activation)
        self.add('identity', identity_activation)  # synonym of linear
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)

    # --------------------------- Registry API ---------------------------

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        try:
            return self.functions[name]
        except KeyError:
            raise InvalidActivationFunction(f"No such activation function: {name!r}")

    def is_valid(self, name):
        return name in self.functions
