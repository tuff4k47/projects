"""
integrator.py

The core math engine for the Integration Calculator. Kept separate from
project.py (the CLI) and app.py (the Flask web UI) so both interfaces share
exactly the same logic instead of duplicating it.

What lives here:
  1. Safe parsing: user text becomes a SymPy expression through a whitelist
     (no eval of arbitrary Python), so the web app cannot be used to run code.
  2. A timeout wrapper that runs heavy work in a killable subprocess, so one
     pathological integral cannot freeze the program or the web server.
  3. Symbolic integration (indefinite and definite) with a numeric fallback
     (mpmath quadrature, which also handles infinite bounds).
  4. Verification: antiderivatives are checked by differentiating, definite
     integrals are cross-checked against independent numerical quadrature.
  5. Step-by-step derivations built from SymPy's rule tree, so they can name
     u-substitution and integration by parts, not just the basic rules.
  6. Numerical methods (left/right/midpoint Riemann, trapezoid, Simpson) with
     convergence tables and an estimate of each method's order of accuracy.
  7. Plots, and analyze_problem(), the single entry point both interfaces use.
"""

import ast
import base64
import io
import math
import multiprocessing
import re
import time

import mpmath
import numpy as np
import sympy
from sympy import integrate, simplify, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    stringify_expr,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

DEFAULT_TIMEOUT = 10  # seconds a calculation may run before it is stopped


# ===========================================================================
# 1. Safe parsing
# ===========================================================================

class ExpressionError(ValueError):
    """Raised when user input is not an allowed mathematical expression."""


MAX_INPUT_LENGTH = 300
_MAX_AST_NODES = 600
_MAX_INT_LITERAL = 10 ** 12

_FUNCTION_NAMES = [
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc", "atan2",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh",
    "exp", "log", "sqrt", "cbrt", "root",
    "Abs", "sign", "floor", "ceiling", "frac", "Max", "Min",
    "gamma", "erf", "erfc", "factorial", "binomial", "re", "im",
]
_FUNCTIONS = {name: getattr(sympy, name) for name in _FUNCTION_NAMES if hasattr(sympy, name)}
_FUNCTIONS.update({
    "ln": sympy.log,
    "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan,
    "arcsinh": sympy.asinh, "arccosh": sympy.acosh, "arctanh": sympy.atanh,
    "abs": sympy.Abs, "ceil": sympy.ceiling, "max": sympy.Max, "min": sympy.Min,
})
_CONSTANTS = {"pi": sympy.pi, "E": sympy.E, "e": sympy.E, "oo": sympy.oo, "I": sympy.I}

# Names the SymPy parser itself emits when it rewrites "2x" or "x^2".
_CONSTRUCTORS = {
    "Integer": sympy.Integer, "Float": sympy.Float,
    "Rational": sympy.Rational, "Symbol": sympy.Symbol,
}

# No "__builtins__" entry here on purpose: it is added as an empty dict at
# parse time, so eval() has no access to open, __import__, and so on.
_SAFE_NAMESPACE = {**_FUNCTIONS, **_CONSTANTS, **_CONSTRUCTORS}

_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub,
)
_VARIABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def _validate_tree(tree, allowed_names, callable_names):
    """
    Walk the parsed code and reject anything that is not plain arithmetic
    over whitelisted names: no attribute access, subscripts, lambdas,
    comprehensions, imports, or calls to anything not on the whitelist.
    """
    string_ok = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("Symbol", "Float") and node.args:
            string_ok.add(id(node.args[0]))

    for count, node in enumerate(ast.walk(tree), start=1):
        if count > _MAX_AST_NODES:
            raise ExpressionError("That expression is too complicated.")
        if isinstance(node, _ALLOWED_NODE_TYPES):
            continue
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and id(node) in string_ok and len(value) <= 64:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and abs(value) <= _MAX_INT_LITERAL:
                continue
            raise ExpressionError("That number or text is not allowed.")
        if isinstance(node, ast.Name):
            if node.id == "Function":
                raise ExpressionError(
                    "Unknown function. Allowed functions include sin, cos, tan, exp, log, "
                    "sqrt, Abs, floor, ceiling, frac, Max, Min, atan, sinh, cosh and more."
                )
            if node.id not in allowed_names:
                raise ExpressionError(f"'{node.id}' is not allowed here.")
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in callable_names):
                raise ExpressionError("That function call is not allowed.")
            continue
        if isinstance(node, ast.keyword):
            if node.arg != "precision":
                raise ExpressionError("That syntax is not allowed.")
            continue
        raise ExpressionError(
            "That syntax is not allowed. Use plain math such as x**2 + 3*x or sin(x)*cos(x)."
        )


def safe_parse(text, local_dict=None):
    """
    Parse math text into a SymPy expression without running arbitrary code.

    SymPy's own parse_expr() ends in eval(), which is unsafe on a public web
    page. Here the text is first rewritten by SymPy's transformations (so
    "2x^2" still works), then the resulting code is checked against a
    whitelist before it is evaluated with no builtins available.
    """
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("Please enter an expression.")
    text = text.strip()
    if len(text) > MAX_INPUT_LENGTH:
        raise ExpressionError(f"Expressions are limited to {MAX_INPUT_LENGTH} characters.")
    if "__" in text:
        raise ExpressionError("Double underscores are not allowed.")

    local = dict(local_dict or {})
    namespace = dict(_SAFE_NAMESPACE)
    try:
        code = stringify_expr(text, local, dict(namespace), TRANSFORMATIONS)
        tree = ast.parse(code.strip(), mode="eval")
    except Exception:
        raise ExpressionError(f"Could not understand '{text}'.") from None

    allowed_names = set(namespace) | {key for key in local if isinstance(key, str)}
    callable_names = set(_FUNCTIONS) | {"Integer", "Float", "Rational", "Symbol"}
    _validate_tree(tree, allowed_names, callable_names)

    try:
        value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}, **namespace}, local)
    except Exception as error:
        raise ExpressionError(f"Could not evaluate '{text}': {error}") from None
    if not isinstance(value, sympy.Expr):
        raise ExpressionError(f"'{text}' is not a mathematical expression.")
    return value


def parse_expression(expression_str, variable_str="x"):
    """
    Turn a user-typed string and a variable name into a SymPy expression and
    a SymPy symbol. Raises ExpressionError (a ValueError) on invalid input.
    """
    variable_str = (variable_str or "x").strip() or "x"
    if not _VARIABLE_RE.match(variable_str) or "__" in variable_str:
        raise ExpressionError(
            "The variable must be a short name made of letters, digits and underscores, like x or t."
        )
    if variable_str != "e" and (variable_str in _SAFE_NAMESPACE or variable_str == "Function"):
        raise ExpressionError(f"'{variable_str}' is a reserved name; pick another variable.")
    var = symbols(variable_str)
    expr = safe_parse(expression_str, {variable_str: var})
    return expr, var


def parse_bound(text):
    """Parse an integration bound such as 0, -pi, 1/2 or oo."""
    return safe_parse(text)


def _as_bound(value):
    """Accept a string (parsed safely) or a number/SymPy object as a bound."""
    if isinstance(value, str):
        return safe_parse(value)
    return sympy.sympify(value)


# ===========================================================================
# 2. Timeouts
# ===========================================================================

class CalculationTimeout(Exception):
    """Raised when a calculation runs past its time limit and is stopped."""


def _limit_memory(limit_bytes=3 * 1024 ** 3):
    """Best effort: cap the worker's memory so a runaway power tower cannot eat the machine."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        pass


def _worker(connection, func, args, kwargs):
    try:
        _limit_memory()
        connection.send(("ok", func(*args, **kwargs)))
    except BaseException as error:  # noqa: BLE001 - everything is relayed to the parent
        try:
            connection.send(("error", error))
        except Exception:
            connection.send(("error", RuntimeError(f"{type(error).__name__}: {error}")))
    finally:
        connection.close()


_context = None


def _get_context():
    """
    forkserver (where available) starts workers from a clean, pre-loaded
    process: fast, and safe even though Flask serves requests on threads.
    """
    global _context
    if _context is None:
        method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        _context = multiprocessing.get_context(method)
        if method == "forkserver":
            _context.set_forkserver_preload(["integrator"])
    return _context


def run_with_timeout(func, *args, timeout=DEFAULT_TIMEOUT, **kwargs):
    """
    Run func(*args, **kwargs) in a separate process and return its result.
    If it takes longer than `timeout` seconds the process is killed and
    CalculationTimeout is raised. Exceptions raised by func are re-raised here.
    func and its arguments/results must be picklable (module-level functions,
    SymPy objects, strings, numbers).
    """
    context = _get_context()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, func, args, kwargs), daemon=True)
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            process.kill()
            process.join(2)
            raise CalculationTimeout(
                f"The calculation took longer than {timeout} seconds and was stopped."
            )
        try:
            status, payload = receiver.recv()
        except EOFError:
            raise RuntimeError("The calculation stopped unexpectedly (it may have run out of memory).") from None
    finally:
        receiver.close()
        process.join(2)
        if process.is_alive():
            process.kill()
    if status == "error":
        raise payload
    return payload


def warm_up():
    """Start the worker machinery once, so the first real request is not slower."""
    run_with_timeout(math.sqrt, 4, timeout=30)


# ===========================================================================
# 3. Integration (symbolic with numeric fallback)
# ===========================================================================

def compute_indefinite_integral(expr, var):
    """
    Returns (result, is_exact). is_exact is False if SymPy could not find a
    closed-form antiderivative (result will contain an unevaluated Integral).
    """
    result = integrate(expr, var)
    is_exact = not result.has(sympy.Integral)
    if is_exact:
        result = simplify(result)
    return result, is_exact


def _to_mp(value):
    if value == sympy.oo:
        return mpmath.inf
    if value == -sympy.oo:
        return -mpmath.inf
    return mpmath.mpf(float(value))


def numeric_integral(expr, var, lower, upper):
    """
    Independent numerical value of a definite integral using mpmath's
    tanh-sinh quadrature, which also copes with infinite bounds and
    endpoint singularities. Returns (value, error_estimate) as floats.
    Raises ValueError if the result is complex or does not converge.
    """
    lower, upper = _as_bound(lower), _as_bound(upper)
    if not (lower.is_number and upper.is_number):
        raise ValueError("Bounds must be numeric values.")
    f = sympy.lambdify(var, expr, modules=["mpmath"])
    a, b = _to_mp(lower), _to_mp(upper)
    points = [a, b]
    if a == -mpmath.inf and b == mpmath.inf:
        points = [a, mpmath.mpf(0), b]
    elif expr.has(sympy.floor, sympy.ceiling, sympy.frac) and math.isfinite(a) and math.isfinite(b):
        # floor/ceiling/frac jump at integers; integrate each unit piece separately
        # so the quadrature never straddles a jump.
        first, last = math.ceil(a) , math.floor(b)
        if last - first <= 4000:
            points = [a] + [mpmath.mpf(k) for k in range(first, last + 1) if a < k < b] + [b]
    try:
        with mpmath.workdps(20):
            value, error = mpmath.quad(f, points, error=True, maxdegree=8)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError) as problem:
        raise ValueError(f"Numerical integration failed: {problem}") from None
    value = mpmath.mpmathify(value)
    if abs(mpmath.im(value)) > 1e-9 * (1 + abs(mpmath.re(value))):
        raise ValueError("The integral is not a real number.")
    real_value, error = float(mpmath.re(value)), float(error)
    if not math.isfinite(real_value):
        raise ValueError("The integral does not have a finite value.")
    if error > 1e-6 * max(1.0, abs(real_value)):
        raise ValueError("Numerical integration did not converge.")
    return real_value, error


def compute_definite_integral(expr, var, lower, upper):
    """
    Returns (exact_result, numeric_result, is_exact) for the definite
    integral of expr from lower to upper. When SymPy finds no closed form,
    numeric_result comes from numerical quadrature (including infinite
    bounds); if even that fails, numeric_result stays unevaluated.
    Raises ValueError if the bounds are not numeric.
    """
    lower, upper = _as_bound(lower), _as_bound(upper)
    if not (lower.is_number and upper.is_number):
        raise ValueError(
            f"Bounds must be numeric values (e.g. 0, 1, pi), not '{lower}' and '{upper}'."
        )
    exact_result = integrate(expr, (var, lower, upper))
    is_exact = not exact_result.has(sympy.Integral)
    if is_exact:
        numeric_result = exact_result.evalf()
        exact_result = simplify(exact_result)
    else:
        try:
            numeric_result = sympy.Float(numeric_integral(expr, var, lower, upper)[0])
        except ValueError:
            numeric_result = exact_result.evalf()
    return exact_result, numeric_result, is_exact


def _format_numeric(value):
    """
    Format a SymPy value as a decimal string, handling infinities (a divergent
    integral), NaN, and complex or still-symbolic values.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return f"undefined ({value})"
    if as_float != as_float:
        return "undefined (the integral may not exist over this interval)"
    if not math.isfinite(as_float):
        return "∞ (the integral diverges)" if as_float > 0 else "-∞ (the integral diverges)"
    return f"{as_float:.6f}"


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_indefinite_result(expr, var, result, is_exact):
    if is_exact:
        return f"∫ {expr} d{var} = {result} + C"
    return (
        f"SymPy could not find a closed-form antiderivative for {expr}. "
        f"Try a definite integral instead to get a numeric answer."
    )


def format_definite_result(expr, var, lower, upper, exact_result, numeric_result, is_exact):
    header = f"∫ from {lower} to {upper} of {expr} d{var}"
    numeric_str = _format_numeric(numeric_result)
    if is_exact:
        return f"{header} = {exact_result}  ≈ {numeric_str}"
    return f"{header} has no closed form; numeric approximation ≈ {numeric_str}"


# ===========================================================================
# 4. Verification
# ===========================================================================

_SPOT_POINTS = [0.37, 0.91, 1.53, 2.27, 3.14, -0.62, -1.7]


def _spot_check(difference, original, var):
    """Evaluate `difference` at several points; returns (True/False/None, points_used)."""
    used = 0
    for point in _SPOT_POINTS:
        try:
            difference_value = complex(difference.subs(var, point).evalf())
            original_value = complex(original.subs(var, point).evalf())
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not (cmath_finite(difference_value) and cmath_finite(original_value)):
            continue
        used += 1
        if abs(difference_value) > 1e-8 * (1 + abs(original_value)):
            return False, used
    return (True if used >= 3 else None), used


def cmath_finite(value):
    return math.isfinite(value.real) and math.isfinite(value.imag)


def verify_antiderivative(expr, var, antiderivative):
    """
    Check an antiderivative by differentiating it and comparing with the
    original integrand. Returns {"verified": True/False/None, "method": str}.
    """
    if antiderivative.has(sympy.Integral):
        return {"verified": None, "method": "no closed form to verify"}
    difference = sympy.diff(antiderivative, var) - expr
    if simplify(difference) == 0:
        return {"verified": True, "method": "differentiated the answer and got back the original function"}
    ok, used = _spot_check(difference, expr, var)
    if ok is True:
        return {"verified": True,
                "method": f"differentiated the answer and matched the original at {used} test points"}
    if ok is False:
        return {"verified": False, "method": "the derivative of the answer did not match the original function"}
    return {"verified": None, "method": "could not be checked automatically"}


def verify_definite(expr, var, lower, upper, exact_result, numeric_result, is_exact):
    """Cross-check a definite integral against independent numerical quadrature."""
    value = _safe_float(numeric_result)
    if value is None:
        return {"verified": None, "method": "not applicable (no finite value)"}
    if not is_exact:
        return {"verified": None, "method": "computed numerically (there is no closed form to compare against)"}
    try:
        check, _ = numeric_integral(expr, var, lower, upper)
    except ValueError:
        return {"verified": None, "method": "could not be cross-checked numerically"}
    if abs(check - value) <= 1e-6 * (1 + abs(check)):
        return {"verified": True, "method": "cross-checked against independent numerical quadrature"}
    return {"verified": False, "method": "the exact value disagreed with numerical quadrature"}


# ===========================================================================
# 5. Step-by-step derivation
# ===========================================================================

_KNOWN_FUNCTION_RULES = {
    sympy.sin: "-cos({v})",
    sympy.cos: "sin({v})",
    sympy.exp: "exp({v})",
    sympy.log: "{v}*log({v}) - {v}",
}

_MAX_STEPS = 60
_MAX_DEPTH = 8


def _pretty(value):
    return str(value).replace("_u", "u")


def generate_steps(expr, var):
    """
    Return a list of human-readable strings explaining how the integral is
    found. Nested steps are indented by two spaces per level.

    The explanation comes from SymPy's rule tree (sympy.integrals.manualintegrate),
    so it can show u-substitution, integration by parts, rewriting and
    trigonometric substitution as well as the sum, constant-multiple and
    power rules. If that finds no applicable rule, a simpler pattern-matching
    explanation is used instead.
    """
    steps = _rule_based_steps(expr, var)
    if steps is None:
        steps = []
        _explain(expr, var, steps)
    final, is_exact = compute_indefinite_integral(expr, var)
    if is_exact:
        steps.append(f"Final answer: ∫ {expr} d{var} = {final} + C")
    else:
        steps.append(f"No elementary closed form was found for ∫ {expr} d{var}.")
    return steps


def _rule_based_steps(expr, var):
    from sympy.integrals.manualintegrate import integral_steps
    try:
        rule = integral_steps(expr, var)
    except Exception:
        return None
    top = rule
    while type(top).__name__ == "AlternativeRule" and getattr(top, "alternatives", None):
        top = top.alternatives[0]
    if type(top).__name__ == "DontKnowRule":
        return None
    steps = []
    try:
        _explain_rule(rule, steps, 0)
    except Exception:
        return None
    if len(steps) > _MAX_STEPS:
        steps = steps[:_MAX_STEPS] + ["(further steps omitted)"]
    return steps or None


def _explain_rule(rule, steps, depth):
    if depth > _MAX_DEPTH or len(steps) > _MAX_STEPS:
        return
    name = type(rule).__name__
    variable = getattr(rule, "variable", None)
    v = _pretty(variable)
    integrand = _pretty(getattr(rule, "integrand", ""))

    def put(level, text):
        steps.append("  " * level + text)

    if name == "AlternativeRule":
        _explain_rule(rule.alternatives[0], steps, depth)
    elif name == "AddRule":
        put(depth, f"Sum rule: split ∫ {integrand} d{v} into {len(rule.substeps)} separate integrals.")
        for substep in rule.substeps:
            _explain_rule(substep, steps, depth + 1)
    elif name == "ConstantTimesRule":
        put(depth, f"Constant multiple rule: ∫ {integrand} d{v} = {_pretty(rule.constant)} · ∫ {_pretty(rule.other)} d{v}.")
        _explain_rule(rule.substep, steps, depth + 1)
    elif name == "ConstantRule":
        put(depth, f"Constant rule: ∫ {integrand} d{v} = {integrand}·{v}.")
    elif name == "PowerRule":
        base, exponent = _pretty(rule.base), rule.exp
        put(depth, f"Power rule: ∫ {base}^{exponent} d{v} = {base}^{_pretty(exponent + 1)} / {_pretty(exponent + 1)}.")
    elif name == "ReciprocalRule":
        base = _pretty(rule.base)
        put(depth, f"Reciprocal rule: ∫ 1/{base} d{v} = ln|{base}|.")
    elif name == "URule":
        du = sympy.diff(rule.u_func, variable)
        put(depth, f"u-substitution: let u = {_pretty(rule.u_func)}, so du = {_pretty(du)} d{v}.")
        put(depth + 1, f"The integral becomes ∫ {_pretty(rule.substep.integrand)} du.")
        _explain_rule(rule.substep, steps, depth + 1)
        put(depth + 1, f"Substitute back u = {_pretty(rule.u_func)}.")
    elif name == "PartsRule":
        if v == "u":
            # Inside a u-substitution the variable is itself called u, so
            # switch to f and g' to keep the formula unambiguous.
            put(depth, f"Integration by parts: ∫ f·g' d{v} = f·g − ∫ f'·g d{v}, "
                       f"with f = {_pretty(rule.u)} and g' = {_pretty(rule.dv)}.")
            put(depth + 1, f"First find g = ∫ {_pretty(rule.dv)} d{v}:")
            _explain_rule(rule.v_step, steps, depth + 2)
            put(depth + 1, f"Then evaluate ∫ f'·g d{v} = ∫ {_pretty(rule.second_step.integrand)} d{v}:")
        else:
            put(depth, f"Integration by parts: ∫ u dv = u·v − ∫ v du, "
                       f"with u = {_pretty(rule.u)} and dv = {_pretty(rule.dv)} d{v}.")
            put(depth + 1, f"First find v = ∫ {_pretty(rule.dv)} d{v}:")
            _explain_rule(rule.v_step, steps, depth + 2)
            put(depth + 1, f"Then evaluate ∫ v du = ∫ {_pretty(rule.second_step.integrand)} d{v}:")
        _explain_rule(rule.second_step, steps, depth + 2)
    elif name == "CyclicPartsRule":
        put(depth, "Integration by parts, applied repeatedly: the original integral reappears, so solve for it algebraically.")
        for part in getattr(rule, "parts_rules", []):
            put(depth + 1, f"Take u = {_pretty(part.u)}, dv = {_pretty(part.dv)} d{v}.")
    elif name in ("RewriteRule", "CompleteSquareRule"):
        put(depth, f"Rewrite the integrand: {integrand} = {_pretty(rule.rewritten)}.")
        _explain_rule(rule.substep, steps, depth + 1)
    elif name == "PiecewiseRule":
        put(depth, "The integrand is piecewise; integrate each piece separately.")
        for substep, condition in rule.subfunctions:
            put(depth + 1, f"On {_pretty(condition)}:")
            _explain_rule(substep, steps, depth + 2)
    elif name == "TrigSubstitutionRule":
        put(depth, f"Trigonometric substitution: let {_pretty(getattr(rule, 'symbol', v))} = {_pretty(getattr(rule, 'func', ''))}, "
                   f"which turns the integral into ∫ {_pretty(getattr(rule, 'rewritten', ''))} d{_pretty(getattr(rule, 'theta', ''))}.")
        _explain_rule(rule.substep, steps, depth + 1)
        put(depth + 1, "Convert back to the original variable.")
    elif name == "DontKnowRule":
        put(depth, f"No textbook rule applies to ∫ {integrand} d{v}; SymPy's general algorithm handles it.")
    elif hasattr(rule, "substep"):
        put(depth, f"Applying {name}.")
        _explain_rule(rule.substep, steps, depth + 1)
    else:
        try:
            antiderivative = integrate(rule.integrand, rule.variable)
            put(depth, f"Standard rule: ∫ {integrand} d{v} = {_pretty(antiderivative)}.")
        except Exception:
            put(depth, f"Standard rule: {name}.")


def generate_basic_steps(expr, var):
    """The simple pattern-matching explanation (sum, constant multiple, power, standard functions)."""
    steps = []
    _explain(expr, var, steps)
    return steps


def _explain(expr, var, steps):
    if expr.is_Add:
        terms = expr.args
        steps.append(f"Sum rule: ∫ ({expr}) d{var} splits into {len(terms)} separate terms.")
        for term in terms:
            _explain(term, var, steps)
        return
    if expr.is_Mul:
        const, rest = expr.as_independent(var)
        if const != 1 and rest != 1:
            steps.append(f"Constant multiple rule: ∫ {expr} d{var} = {const} · ∫ {rest} d{var}.")
            _explain(rest, var, steps)
            return
        steps.append(
            f"{expr} is a product of functions of {var}; applying SymPy's general "
            f"integration algorithm (which may use substitution or integration by parts)."
        )
        return
    if expr.is_Pow and expr.base == var and expr.exp.is_number:
        n = expr.exp
        if n == -1:
            steps.append(f"Reciprocal rule: ∫ {var}^(-1) d{var} = ln|{var}|.")
        else:
            steps.append(f"Power rule: ∫ {var}^{n} d{var} = {var}^{n + 1} / {n + 1}.")
        return
    if expr == var:
        steps.append(f"Power rule (n = 1): ∫ {var} d{var} = {var}^2 / 2.")
        return
    if expr.is_number:
        steps.append(f"Constant rule: ∫ {expr} d{var} = {expr}·{var}.")
        return
    if expr.func in _KNOWN_FUNCTION_RULES and expr.args == (var,):
        antiderivative = _KNOWN_FUNCTION_RULES[expr.func].format(v=var)
        steps.append(f"Standard rule: ∫ {expr} d{var} = {antiderivative}.")
        return
    steps.append(
        f"No simple textbook rule directly matches {expr}; applying SymPy's general "
        f"integration algorithm."
    )


def format_steps(steps):
    """Number the top-level steps and bullet the nested ones, keeping indentation."""
    lines, counter = [], 0
    for step in steps:
        depth = (len(step) - len(step.lstrip(" "))) // 2
        text = step.strip()
        if depth == 0:
            counter += 1
            lines.append(f"{counter}. {text}")
        else:
            lines.append("   " * depth + f"- {text}")
    return "\n".join(lines)


def steps_with_depth(steps):
    """Return [(depth, text), ...] for templates that indent nested steps."""
    return [((len(step) - len(step.lstrip(" "))) // 2, step.strip()) for step in steps]


# ===========================================================================
# 6. Numerical methods and convergence
# ===========================================================================

METHODS = ("left", "right", "midpoint", "trapezoid", "simpson")
METHOD_LABELS = {
    "left": "Left Riemann", "right": "Right Riemann", "midpoint": "Midpoint",
    "trapezoid": "Trapezoid", "simpson": "Simpson's",
}
EXPECTED_ORDER = {"left": 1, "right": 1, "midpoint": 2, "trapezoid": 2, "simpson": 4}


def _numeric_function(expr, var):
    custom = {"frac": lambda value: value - np.floor(value)}
    return sympy.lambdify(var, expr, modules=[custom, "numpy"])


def _evaluate(f, xs, strict=True):
    """Evaluate f on an array, always returning an array; strict mode rejects non-finite values."""
    with np.errstate(all="ignore"):
        try:
            ys = np.asarray(f(xs))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as problem:
            raise ValueError(f"The function cannot be evaluated here: {problem}") from None
    if np.iscomplexobj(ys):
        ys = np.where(np.abs(ys.imag) < 1e-12, ys.real, np.nan)
    ys = np.broadcast_to(ys.astype(float), np.shape(xs)).copy()
    if strict and not np.all(np.isfinite(ys)):
        raise ValueError("The function is undefined or infinite at a point this method needs to evaluate.")
    return ys


def _finite_bounds(lower, upper):
    lower, upper = _as_bound(lower), _as_bound(upper)
    if not (lower.is_number and upper.is_number):
        raise ValueError(f"Bounds must be numeric values (e.g. 0, 1, pi), not '{lower}' and '{upper}'.")
    a, b = _safe_float(lower), _safe_float(upper)
    if a is None or b is None:
        raise ValueError(
            "Riemann, trapezoid and Simpson methods need a finite interval, so they are skipped for infinite bounds."
        )
    return a, b


def numerical_integral(expr, var, lower, upper, n, method="midpoint"):
    """
    Approximate the definite integral using n subintervals.
    method: "left", "right", "midpoint", "trapezoid" or "simpson"
    (Simpson's rule needs an even n; an odd n is increased by one).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    if n < 1:
        raise ValueError("n must be at least 1")
    a, b = _finite_bounds(lower, upper)
    f = _numeric_function(expr, var)
    n = int(n)
    dx = (b - a) / n

    if method == "left":
        return float(np.sum(_evaluate(f, a + dx * np.arange(n))) * dx)
    if method == "right":
        return float(np.sum(_evaluate(f, a + dx * np.arange(1, n + 1))) * dx)
    if method == "midpoint":
        return float(np.sum(_evaluate(f, a + dx * (np.arange(n) + 0.5))) * dx)

    if method == "simpson" and n % 2:
        n += 1
        dx = (b - a) / n
    ys = _evaluate(f, np.linspace(a, b, n + 1))
    if method == "trapezoid":
        return float(dx * (np.sum(ys) - 0.5 * (ys[0] + ys[-1])))
    return float(dx / 3 * (ys[0] + ys[-1] + 4 * np.sum(ys[1:-1:2]) + 2 * np.sum(ys[2:-1:2])))


def riemann_sum(expr, var, lower, upper, n, method="midpoint"):
    """Left, right or midpoint Riemann sum with n equal-width subintervals."""
    if method not in ("left", "right", "midpoint"):
        raise ValueError("method must be 'left', 'right', or 'midpoint'")
    return numerical_integral(expr, var, lower, upper, n, method)


def reference_value(expr, var, lower, upper):
    """
    The best available true value of a definite integral: the exact symbolic
    value when SymPy has one, otherwise high-precision numerical quadrature.
    Raises ValueError if the integral diverges or is not a real number.
    """
    lower, upper = _as_bound(lower), _as_bound(upper)
    exact = integrate(expr, (var, lower, upper))
    if exact.has(sympy.Integral):
        return numeric_integral(expr, var, lower, upper)[0]
    try:
        value = float(exact.evalf())
    except (TypeError, ValueError):
        raise ValueError("This integral does not evaluate to a real number.") from None
    if not math.isfinite(value):
        raise ValueError("This integral diverges (has no finite value).")
    return value


def riemann_convergence_table(expr, var, lower, upper, n_values=None, method="midpoint"):
    """
    Returns (true_value, rows) where rows is a list of {n, approx, error}
    showing how the sum approaches the true value as n grows.
    Raises ValueError if bounds are not finite numbers or no finite value exists.
    """
    _finite_bounds(lower, upper)
    if n_values is None:
        n_values = [10, 50, 100, 500, 1000, 5000]
    true_value = reference_value(expr, var, lower, upper)
    rows = []
    for n in n_values:
        approx = numerical_integral(expr, var, lower, upper, n, method)
        rows.append({"n": n, "approx": approx, "error": abs(approx - true_value)})
    return true_value, rows


def order_status(rows):
    """
    Estimate a method's order of accuracy p (error ~ C / n^p) from the slope of
    log(error) against log(n). Returns (order, status) where status is:
      "ok"        - errors follow a clean power law; order is the estimate
      "exact"     - errors are already at machine precision (order is None)
      "irregular" - errors do not follow a power law (jumps, kinks or noise),
                    so the fitted slope should not be trusted
    """
    points = [(row["n"], row["error"]) for row in rows if row["error"] > 1e-13]
    if len(points) < 2:
        return None, "exact"
    logs_n = np.log([n for n, _ in points])
    logs_error = np.log([error for _, error in points])
    slope, intercept = np.polyfit(logs_n, logs_error, 1)
    residual = logs_error - (slope * logs_n + intercept)
    spread = np.sum((logs_error - np.mean(logs_error)) ** 2)
    r_squared = 1.0 if spread == 0 else 1 - np.sum(residual ** 2) / spread
    order = round(float(-slope), 2)
    if r_squared < 0.9 or order <= 0.2:
        return order, "irregular"
    return order, "ok"


def estimate_order(rows):
    """The estimated order of accuracy, or None if exact or irregular (see order_status)."""
    order, status = order_status(rows)
    return order if status == "ok" else None


def compare_methods(expr, var, lower, upper, n_values=None,
                    methods=("left", "midpoint", "trapezoid", "simpson")):
    """
    Run several numerical methods for a doubling sequence of n and compare how
    fast each converges. Returns (true_value, results) where each result is
    {"method", "label", "rows", "order", "order_status"}; methods that cannot be evaluated
    (for example an endpoint where the function is infinite) are skipped.
    """
    _finite_bounds(lower, upper)
    if n_values is None:
        n_values = [10, 20, 40, 80, 160, 320, 640]
    true_value = reference_value(expr, var, lower, upper)
    results, last_problem = [], None
    for method in methods:
        try:
            rows = []
            for n in n_values:
                approx = numerical_integral(expr, var, lower, upper, n, method)
                rows.append({"n": n, "approx": approx, "error": abs(approx - true_value)})
        except ValueError as problem:
            last_problem = problem
            continue
        order, status = order_status(rows)
        results.append({"method": method, "label": METHOD_LABELS[method], "rows": rows,
                        "order": order if status == "ok" else None, "order_status": status})
    if not results:
        raise ValueError(str(last_problem))
    return true_value, results


# ===========================================================================
# 7. Plotting
# ===========================================================================

def _matplotlib_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_window(lower, upper, reach=6.0):
    """Finite bounds for drawing; an infinite bound is truncated `reach` units from the other end."""
    lower, upper = _as_bound(lower), _as_bound(upper)
    if not (lower.is_number and upper.is_number):
        raise ValueError(f"Bounds must be numeric values (e.g. 0, 1, pi), not '{lower}' and '{upper}'.")
    a, b = _safe_float(lower), _safe_float(upper)
    if a is None and b is None:
        return -reach, reach
    if a is None:
        return b - reach, b
    if b is None:
        return a, a + reach
    return a, b


def plot_integral(expr, var, lower, upper, filename=None):
    """
    Plot expr over a range comfortably containing [lower, upper], shading the
    area between them. Saves a PNG and returns the filename if filename is
    given; otherwise returns the PNG as a base64 string (used by the web UI).
    """
    plt = _matplotlib_pyplot()
    a, b = _plot_window(lower, upper)
    f = _numeric_function(expr, var)
    span = max(b - a, 1)
    xs = np.linspace(a - span * 0.5, b + span * 0.5, 600)
    ys = _evaluate(f, xs, strict=False)
    shade_xs = np.linspace(a, b, 400)
    shade_ys = _evaluate(f, shade_xs, strict=False)

    plt.figure(figsize=(7, 4.5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(xs, ys, label=f"f({var}) = {expr}")
    plt.fill_between(shade_xs, shade_ys, alpha=0.3, label=f"∫ from {lower} to {upper}")
    finite = ys[np.isfinite(ys)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
        pad = 0.15 * (high - low if high > low else 1)
        plt.ylim(min(low, 0) - pad, max(high, 0) + pad)
    plt.legend()
    plt.title("Definite Integral")
    plt.xlabel(str(var))
    plt.ylabel(f"f({var})")
    plt.tight_layout()
    return _finish_plot(plt, filename)


def plot_convergence(rows, filename=None):
    """Plot a single method's error against n on a log-log scale."""
    plt = _matplotlib_pyplot()
    ns = [row["n"] for row in rows]
    errors = [max(row["error"], 1e-16) for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.loglog(ns, errors, marker="o")
    plt.title("Riemann Sum Convergence")
    plt.xlabel("n (number of subintervals)")
    plt.ylabel("Absolute error vs. true value")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return _finish_plot(plt, filename)


def plot_method_comparison(results, filename=None):
    """Log-log plot of error against n for several methods, labelled with their estimated order."""
    plt = _matplotlib_pyplot()
    plt.figure(figsize=(7, 4.5))
    for result in results:
        ns = [row["n"] for row in result["rows"]]
        errors = [max(row["error"], 1e-16) for row in result["rows"]]
        status = result.get("order_status", "ok")
        if status == "ok" and result["order"] is not None:
            label = f"{result['label']} (order ≈ {result['order']})"
        else:
            label = f"{result['label']} ({status})"
        plt.loglog(ns, errors, marker="o", label=label)
    plt.title("Error vs. number of subintervals")
    plt.xlabel("n (number of subintervals)")
    plt.ylabel("Absolute error vs. true value")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return _finish_plot(plt, filename)


def _finish_plot(plt, filename):
    if filename:
        plt.savefig(filename)
        plt.close()
        return filename
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


# ===========================================================================
# 8. One entry point for both interfaces
# ===========================================================================

EXTRA_STAGE_LIMIT = 6  # seconds any single optional stage (steps, comparison, ...) may use


def _core_stage(expression, variable, lower, upper):
    """Parse and integrate. Returns (report, state); state keeps SymPy objects for later stages."""
    expr, var = parse_expression(expression, variable)
    lower, upper = (lower or "").strip(), (upper or "").strip()
    if bool(lower) != bool(upper):
        raise ValueError("Enter both bounds for a definite integral, or leave both blank for an indefinite one.")

    report = {"expression": str(expr), "variable": str(var),
              "steps": [], "steps_error": None, "verification": None,
              "comparison": None, "comparison_error": None,
              "function_plot": None, "comparison_plot": None, "plot_error": None}

    if not lower:
        result, is_exact = compute_indefinite_integral(expr, var)
        report.update(kind="indefinite", is_exact=is_exact,
                      result=format_indefinite_result(expr, var, result, is_exact))
        state = {"kind": "indefinite", "expr": expr, "var": var, "antiderivative": result}
    else:
        lo, hi = parse_bound(lower), parse_bound(upper)
        exact, numeric, is_exact = compute_definite_integral(expr, var, lo, hi)
        report.update(kind="definite", is_exact=is_exact, lower=str(lo), upper=str(hi),
                      numeric=_safe_float(numeric),
                      result=format_definite_result(expr, var, lo, hi, exact, numeric, is_exact))
        state = {"kind": "definite", "expr": expr, "var": var, "lower": lo, "upper": hi,
                 "exact": exact, "numeric": numeric, "is_exact": is_exact}
    return report, state


def _verify_stage(state):
    if state["kind"] == "indefinite":
        return verify_antiderivative(state["expr"], state["var"], state["antiderivative"])
    return verify_definite(state["expr"], state["var"], state["lower"], state["upper"],
                           state["exact"], state["numeric"], state["is_exact"])


def _steps_stage(state):
    return generate_steps(state["expr"], state["var"])


def _comparison_stage(state):
    true_value, methods = compare_methods(state["expr"], state["var"], state["lower"], state["upper"])
    return {"true_value": true_value, "methods": methods}


def _plot_stage(state, methods):
    """Returns (function_plot, comparison_plot, problem); one failing plot does not block the other."""
    function_plot = comparison_plot = None
    problems = []
    try:
        function_plot = plot_integral(state["expr"], state["var"], state["lower"], state["upper"])
    except Exception as problem:  # noqa: BLE001
        problems.append(f"Could not draw the function plot: {problem}")
    if methods:
        try:
            comparison_plot = plot_method_comparison(methods)
        except Exception as problem:  # noqa: BLE001
            problems.append(f"Could not draw the convergence plot: {problem}")
    return function_plot, comparison_plot, " ".join(problems) or None


def analyze_problem(expression, variable="x", lower="", upper="",
                    want_steps=True, want_comparison=True, want_plots=False, timeout=None):
    """
    Do everything for one problem and return a plain dict that the CLI and the
    web page both display. Raises ValueError (including ExpressionError) for
    bad input.

    With timeout=None everything runs in this process (handy for tests).
    With a timeout in seconds, the answer itself (parse + integrate) runs in a
    killable subprocess and raises CalculationTimeout if it takes too long.
    Each optional extra (check, steps, numerical comparison, plots) then runs
    in its own subprocess within a shared time pool, so a slow extra is
    reported as such but never costs you the answer.
    """
    isolated = timeout is not None
    pool = {"left": float(timeout or 0)}

    def run_extra(func, *args):
        if not isolated:
            return func(*args)
        limit = min(EXTRA_STAGE_LIMIT, pool["left"])
        if limit < 0.5:
            raise CalculationTimeout("The time budget for optional extras was used up.")
        started = time.monotonic()
        try:
            return run_with_timeout(func, *args, timeout=limit)
        finally:
            pool["left"] -= time.monotonic() - started

    if isolated:
        report, state = run_with_timeout(_core_stage, expression, variable, lower, upper, timeout=timeout)
    else:
        report, state = _core_stage(expression, variable, lower, upper)

    try:
        report["verification"] = run_extra(_verify_stage, state)
    except CalculationTimeout:
        report["verification"] = {"verified": None, "method": "the check took too long to run"}
    except Exception as problem:  # noqa: BLE001 - an extra must never hide the answer
        report["verification"] = {"verified": None, "method": f"the check could not be run ({type(problem).__name__})"}

    if want_steps:
        try:
            report["steps"] = run_extra(_steps_stage, state)
        except CalculationTimeout:
            report["steps_error"] = "The step-by-step explanation took too long to generate. The result above is unaffected."
        except Exception as problem:  # noqa: BLE001
            report["steps_error"] = f"The step-by-step explanation could not be generated ({type(problem).__name__})."

    if state["kind"] == "definite":
        if want_comparison:
            try:
                report["comparison"] = run_extra(_comparison_stage, state)
            except ValueError as problem:
                report["comparison_error"] = str(problem)
            except CalculationTimeout:
                report["comparison_error"] = "The numerical comparison took too long to run."
            except Exception as problem:  # noqa: BLE001
                report["comparison_error"] = f"The numerical comparison could not be run ({type(problem).__name__})."
        if want_plots:
            methods = report["comparison"]["methods"] if report["comparison"] else None
            try:
                plots = run_extra(_plot_stage, state, methods)
                report["function_plot"], report["comparison_plot"], report["plot_error"] = plots
            except CalculationTimeout:
                report["plot_error"] = "Drawing the plots took too long."
            except Exception as problem:  # noqa: BLE001
                report["plot_error"] = f"Could not draw the plots ({type(problem).__name__})."
    return report
