"""
test_project.py

Tests for the integration calculator. Run from the project folder with:

    pytest test_project.py

Sections: command-line helpers (project.py), safe parsing, timeouts,
integration and numeric fallback, verification, numerical methods, steps,
the full analysis pipeline (including all 20 sample problems), and the
Flask routes.
"""

import base64
import math
import time

import pytest
import sympy
from sympy import symbols

import app as web
from integrator import (
    METHODS,
    CalculationTimeout,
    ExpressionError,
    analyze_problem,
    compare_methods,
    compute_definite_integral,
    compute_indefinite_integral,
    estimate_order,
    format_definite_result,
    format_indefinite_result,
    format_steps,
    generate_basic_steps,
    generate_steps,
    numerical_integral,
    order_status,
    parse_bound,
    parse_expression,
    riemann_convergence_table,
    riemann_sum,
    run_with_timeout,
    steps_with_depth,
    verify_antiderivative,
    verify_definite,
)
from project import (
    QuitRequested,
    ask,
    ask_yes_no,
    format_comparison_table,
    format_report,
    interpret_yes_no,
    parse_bounds,
    save_plot,
)


def P(text, variable="x"):
    return parse_expression(text, variable)


# ---------------------------------------------------------------------------
# project.py helpers (the command-line interface)
# ---------------------------------------------------------------------------

def test_parse_bounds_blank_means_indefinite():
    assert parse_bounds("") == ("", "")
    assert parse_bounds("   ") == ("", "")


@pytest.mark.parametrize("text, expected", [
    ("0,1", ("0", "1")),
    (" -pi , pi ", ("-pi", "pi")),
    ("-oo,oo", ("-oo", "oo")),
    ("Max(0,1),2", ("Max(0,1)", "2")),   # comma inside parentheses is not a separator
])
def test_parse_bounds_splits_on_top_level_comma(text, expected):
    assert parse_bounds(text) == expected


@pytest.mark.parametrize("text", ["1", "1,2,3", ",1", "1,", "a,"])
def test_parse_bounds_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_bounds(text)


def test_interpret_yes_no():
    assert interpret_yes_no("y") is True
    assert interpret_yes_no(" YES ") is True
    assert interpret_yes_no("n") is False
    assert interpret_yes_no("") is False
    assert interpret_yes_no("", default=True) is True
    assert interpret_yes_no("maybe", default=True) is True


def test_ask_returns_stripped_input_and_quits(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "  x**2  ")
    assert ask("f(x) = ") == "x**2"
    monkeypatch.setattr("builtins.input", lambda prompt: "QUIT")
    with pytest.raises(QuitRequested):
        ask("f(x) = ")


def test_ask_yes_no_uses_default_on_blank(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert ask_yes_no("Continue?") is False
    assert ask_yes_no("Continue?", default=True) is True
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert ask_yes_no("Continue?") is True


def test_format_report_for_definite_integral():
    report = analyze_problem("x**2", "x", "0", "3", True, True, False)
    text = format_report(report)
    assert "= 9" in text
    assert "Verified" in text
    assert "Step-by-step" in text
    assert "Simpson" in text and "order" in text


def test_format_report_for_indefinite_integral():
    text = format_report(analyze_problem("x*exp(x)", "x", "", "", True, True, False))
    assert "+ C" in text
    assert "Integration by parts" in text


def test_format_comparison_table_shape():
    report = analyze_problem("sin(x)", "x", "0", "3", False, True, False)
    table = format_comparison_table(report["comparison"])
    lines = table.splitlines()
    assert lines[0].startswith("True value:")
    assert any(line.lstrip().startswith("order") for line in lines)
    assert "Midpoint" in table and "Trapezoid" in table


def test_save_plot_writes_a_png(tmp_path):
    report = analyze_problem("x**2", "x", "0", "1", False, False, True)
    path = save_plot(report["function_plot"], str(tmp_path / "plot.png"))
    with open(path, "rb") as file:
        assert file.read(4) == b"\x89PNG"


# ---------------------------------------------------------------------------
# Parsing and security
# ---------------------------------------------------------------------------

def test_parse_expression_basic():
    expr, var = P("x**2 + 3*x")
    x = symbols("x")
    assert expr == x**2 + 3 * x
    assert var == x


def test_parse_expression_implicit_multiplication_and_caret():
    expr, _ = P("2x^2")
    assert expr == 2 * symbols("x")**2


def test_parse_expression_custom_variable():
    expr, var = P("t**2 + 1", "t")
    t = symbols("t")
    assert expr == t**2 + 1
    assert var == t


@pytest.mark.parametrize("text", [
    "sin(x)*cos(x)", "e**(-x**2)", "1/(x**2 + 1)", "Abs(x + Abs(x))", "frac(floor(x)/3)",
    "Max(0, sqrt(1-x**2) - 1/2)", "atan(sqrt(x))", "ln(x)/x", "a*x + 1", "x**2/(1-x)",
])
def test_parse_accepts_ordinary_math(text):
    expr, _ = P(text)
    assert isinstance(expr, sympy.Expr)


HOSTILE_INPUTS = [
    "__import__('os').system('echo pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "x.__class__",
    "x.func",
    "open('/etc/passwd').read()",
    "exec('print(1)')",
    "eval('1+1')",
    "[c for c in (1, 2)]",
    "lambda: 1",
    "'a'*10**9",
    "x[0]",
    "getattr(x, 'func')",
    "(1).real",
    "1; import os",
    "f(3)",
    "x" * 400,
    "",
]


@pytest.mark.parametrize("text", HOSTILE_INPUTS)
def test_parse_rejects_hostile_or_invalid_input(text):
    with pytest.raises(ExpressionError):
        P(text)


def test_hostile_input_does_not_execute(tmp_path):
    target = tmp_path / "created_by_attack.txt"
    with pytest.raises(ExpressionError):
        P(f"open('{target}', 'w').write('hi')")
    assert not target.exists()


@pytest.mark.parametrize("variable", ["sin", "pi", "a b", "x,y", "__x", "Function", "1x", "x" * 40])
def test_parse_rejects_bad_variable_names(variable):
    with pytest.raises(ExpressionError):
        P("x + 1", variable)


def test_bounds_given_as_text_are_parsed_safely():
    expr, var = P("x**2")
    with pytest.raises(ValueError):
        compute_definite_integral(expr, var, "__import__('os')", "1")
    assert parse_bound("-oo") == -sympy.oo
    assert parse_bound("pi/2") == sympy.pi / 2


def test_compute_definite_integral_rejects_symbolic_bounds():
    expr, var = P("x**2")
    with pytest.raises(ValueError):
        compute_definite_integral(expr, var, "a", "b")


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

def test_run_with_timeout_returns_result():
    assert run_with_timeout(math.sqrt, 81, timeout=10) == 9.0


def test_run_with_timeout_stops_slow_calls():
    started = time.monotonic()
    with pytest.raises(CalculationTimeout):
        run_with_timeout(time.sleep, 60, timeout=1)
    assert time.monotonic() - started < 15


def test_run_with_timeout_relays_exceptions():
    with pytest.raises(ExpressionError):
        run_with_timeout(parse_expression, "((", "x", timeout=10)


def test_run_with_timeout_round_trips_sympy_objects():
    expr, var = run_with_timeout(parse_expression, "x**2 + 1", "x", timeout=10)
    assert expr == symbols("x")**2 + 1 and var == symbols("x")


def test_hanging_integral_times_out_instead_of_freezing():
    # This integrand makes SymPy's integrate() run effectively forever.
    hang = "sqrt((cos(x)*cot(x)*csc(x))/(sin(x)*tan(x)*sec(x)))"
    started = time.monotonic()
    with pytest.raises(CalculationTimeout):
        analyze_problem(hang, "x", "", "", False, False, False, timeout=3)
    assert time.monotonic() - started < 20


def test_power_tower_attack_is_stopped():
    with pytest.raises(CalculationTimeout):
        run_with_timeout(parse_expression, "9**9**9**9**9", "x", timeout=3)


# ---------------------------------------------------------------------------
# Integration, numeric fallback, formatting
# ---------------------------------------------------------------------------

def test_compute_indefinite_integral_polynomial():
    expr, var = P("x**2")
    result, is_exact = compute_indefinite_integral(expr, var)
    assert is_exact is True
    assert result == sympy.Rational(1, 3) * var**3


def test_compute_indefinite_integral_trig():
    expr, var = P("cos(x)")
    result, is_exact = compute_indefinite_integral(expr, var)
    assert is_exact is True
    assert result == sympy.sin(var)


def test_compute_indefinite_integral_no_elementary_form():
    expr, var = P("exp(sin(x))")
    _, is_exact = compute_indefinite_integral(expr, var)
    assert is_exact is False


def test_compute_definite_integral_exact():
    expr, var = P("x**2")
    exact, numeric, is_exact = compute_definite_integral(expr, var, 0, 3)
    assert is_exact is True and exact == 9
    assert abs(numeric - 9.0) < 1e-9


def test_compute_definite_integral_numeric_fallback():
    expr, var = P("exp(sin(x))")
    _, numeric, is_exact = compute_definite_integral(expr, var, 0, 1)
    assert is_exact is False
    assert abs(float(numeric) - 1.6318696084) < 1e-8


def test_infinite_bounds_use_numerical_quadrature():
    # Converges to sqrt(pi)/2 but SymPy finds no closed form.
    expr, var = P("exp(-x**2)/(1+exp(2*x))")
    _, numeric, is_exact = compute_definite_integral(expr, var, "-oo", "oo")
    assert is_exact is False
    assert abs(float(numeric) - math.sqrt(math.pi) / 2) < 1e-8


def test_floor_integrand_converges_by_splitting_at_integers():
    expr, var = P("frac(floor(x)/3)")
    _, numeric, _ = compute_definite_integral(expr, var, 0, 2026)
    assert abs(float(numeric) - 675.0) < 1e-6


def test_format_indefinite_result_exact():
    expr, var = P("x**2")
    result, is_exact = compute_indefinite_integral(expr, var)
    assert "+ C" in format_indefinite_result(expr, var, result, is_exact)


def test_format_definite_result_exact():
    expr, var = P("x")
    exact, numeric, is_exact = compute_definite_integral(expr, var, 0, 2)
    assert "2.000000" in format_definite_result(expr, var, 0, 2, exact, numeric, is_exact)


def test_format_definite_result_handles_divergent_integral():
    expr, var = P("1/x")
    exact, numeric, is_exact = compute_definite_integral(expr, var, 0, 1)
    assert "diverges" in format_definite_result(expr, var, 0, 1, exact, numeric, is_exact)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["x**2 + 3*x", "x*exp(x)", "sin(x)*cos(x)", "atan(sqrt(x))", "1/x"])
def test_verify_antiderivative_accepts_correct_answers(text):
    expr, var = P(text)
    result, _ = compute_indefinite_integral(expr, var)
    assert verify_antiderivative(expr, var, result)["verified"] is True


def test_verify_antiderivative_rejects_a_wrong_answer():
    expr, var = P("x")
    assert verify_antiderivative(expr, var, var**3)["verified"] is False


def test_verify_antiderivative_without_closed_form_is_inconclusive():
    expr, var = P("exp(sin(x))")
    result, _ = compute_indefinite_integral(expr, var)
    assert verify_antiderivative(expr, var, result)["verified"] is None


def test_verify_definite_cross_checks_numerically():
    expr, var = P("x**2")
    exact, numeric, is_exact = compute_definite_integral(expr, var, 0, 3)
    assert verify_definite(expr, var, 0, 3, exact, numeric, is_exact)["verified"] is True
    assert verify_definite(expr, var, 0, 3, exact, 8.0, is_exact)["verified"] is False


def test_verify_definite_skips_divergent_integrals():
    expr, var = P("1/x")
    exact, numeric, is_exact = compute_definite_integral(expr, var, 0, 1)
    assert verify_definite(expr, var, 0, 1, exact, numeric, is_exact)["verified"] is None


# ---------------------------------------------------------------------------
# Numerical methods and convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method, tolerance", [
    ("left", 0.05), ("right", 0.05), ("midpoint", 1e-3), ("trapezoid", 1e-3), ("simpson", 1e-6),
])
def test_every_method_approximates_the_integral(method, tolerance):
    expr, var = P("sin(x)")
    assert abs(numerical_integral(expr, var, 0, math.pi, 100, method) - 2.0) < tolerance


def test_simpson_rounds_odd_n_up_to_even():
    expr, var = P("x**3")
    assert abs(numerical_integral(expr, var, 0, 2, 5, "simpson") - 4.0) < 1e-9


def test_numerical_integral_rejects_bad_arguments():
    expr, var = P("x")
    with pytest.raises(ValueError):
        numerical_integral(expr, var, 0, 1, 10, "bogus")
    with pytest.raises(ValueError):
        numerical_integral(expr, var, 0, 1, 0, "midpoint")
    with pytest.raises(ValueError):
        numerical_integral(expr, var, 0, "oo", 10, "midpoint")


def test_riemann_sum_matches_exact_for_polynomial():
    expr, var = P("x**2")
    assert abs(riemann_sum(expr, var, 0, 3, 2000, method="midpoint") - 9.0) < 1e-3


def test_riemann_sum_invalid_method_raises():
    expr, var = P("x**2")
    with pytest.raises(ValueError):
        riemann_sum(expr, var, 0, 1, 10, method="not-a-method")
    with pytest.raises(ValueError):
        riemann_sum(expr, var, 0, 1, 10, method="simpson")  # not a Riemann sum


def test_riemann_convergence_table_error_decreases():
    expr, var = P("x**2")
    true_value, rows = riemann_convergence_table(expr, var, 0, 3, n_values=[10, 100, 1000])
    assert abs(true_value - 9.0) < 1e-9
    errors = [row["error"] for row in rows]
    assert errors[0] >= errors[1] >= errors[2]


def test_riemann_convergence_table_rejects_divergent_integral():
    expr, var = P("1/x")
    with pytest.raises(ValueError):
        riemann_convergence_table(expr, var, 0, 1)


def test_riemann_convergence_works_without_a_closed_form():
    expr, var = P("exp(sin(x))")
    true_value, rows = riemann_convergence_table(expr, var, 0, 1, n_values=[10, 100, 1000])
    assert abs(true_value - 1.6318696084) < 1e-8
    assert rows[-1]["error"] < rows[0]["error"]


def test_estimated_orders_match_theory():
    expr, var = P("sin(x)")
    _, results = compare_methods(expr, var, 0, 3)
    orders = {r["method"]: r["order"] for r in results}
    assert 0.9 < orders["left"] < 1.3
    assert 1.9 < orders["midpoint"] < 2.1
    assert 1.9 < orders["trapezoid"] < 2.1
    assert 3.8 < orders["simpson"] < 4.2


def test_simpson_is_exact_for_cubics():
    expr, var = P("x**3 - x")
    _, results = compare_methods(expr, var, 0, 2)
    simpson = next(r for r in results if r["method"] == "simpson")
    assert simpson["order_status"] == "exact" and simpson["order"] is None


def test_order_status_flags_irregular_errors():
    rows = [{"n": n, "error": e} for n, e in [(10, 1e-2), (20, 3e-2), (40, 5e-3), (80, 2e-2), (160, 1e-3)]]
    assert order_status(rows)[1] == "irregular"
    assert estimate_order(rows) is None


def test_compare_methods_skips_methods_that_hit_a_singularity():
    # 1/sqrt(x) is infinite at 0, so only the midpoint rule can be evaluated.
    expr, var = P("1/sqrt(x)")
    true_value, results = compare_methods(expr, var, 0, 1)
    assert abs(true_value - 2.0) < 1e-6
    assert [r["method"] for r in results] == ["midpoint"]


def test_compare_methods_rejects_divergent_and_infinite_cases():
    with pytest.raises(ValueError):
        compare_methods(*P("1/x"), 0, 1)
    with pytest.raises(ValueError):
        compare_methods(*P("exp(-x**2)"), 0, "oo")


def test_all_method_names_are_supported():
    expr, var = P("x")
    for method in METHODS:
        assert abs(numerical_integral(expr, var, 0, 2, 10, method) - 2.0) < 1e-9 or method in ("left", "right")


# ---------------------------------------------------------------------------
# Step-by-step derivations
# ---------------------------------------------------------------------------

def test_generate_steps_power_rule():
    steps = generate_steps(*P("x**3"))
    assert any("Power rule" in step for step in steps)
    assert any("x**4/4" in step for step in steps)


def test_generate_steps_sum_rule():
    steps = generate_steps(*P("x**2 + sin(x)"))
    assert any("Sum rule" in step for step in steps)
    assert any("Power rule" in step for step in steps)
    assert any("Standard rule" in step for step in steps)


def test_generate_steps_constant_multiple_rule():
    assert any("Constant multiple rule" in step for step in generate_steps(*P("5*x**2")))


def test_generate_steps_detects_u_substitution():
    steps = generate_steps(*P("2*x*cos(x**2)"))
    assert any("u-substitution" in step and "x**2" in step for step in steps)
    assert any("Substitute back" in step for step in steps)


def test_generate_steps_detects_integration_by_parts():
    steps = generate_steps(*P("x*exp(x)"))
    assert any("Integration by parts" in step for step in steps)


def test_generate_steps_handles_repeated_parts_and_nesting():
    steps = generate_steps(*P("x**2*sin(x)"))
    assert sum("Integration by parts" in step for step in steps) == 2
    assert max(depth for depth, _ in steps_with_depth(steps)) >= 2


def test_generate_steps_reports_when_no_closed_form_exists():
    steps = generate_steps(*P("exp(sin(x))"))
    assert any("No elementary closed form" in step for step in steps)


def test_generate_steps_reciprocal_rule():
    assert any("Reciprocal rule" in step for step in generate_steps(*P("1/x")))


def test_generate_basic_steps_fallback_still_works():
    assert any("Sum rule" in step for step in generate_basic_steps(*P("x**2 + 3*x")))


def test_format_steps_numbers_top_level_and_bullets_nested():
    text = format_steps(["Top", "  nested", "Second"])
    assert text.splitlines() == ["1. Top", "   - nested", "2. Second"]
    assert steps_with_depth(["Top", "  nested"]) == [(0, "Top"), (1, "nested")]


# ---------------------------------------------------------------------------
# The full analysis pipeline
# ---------------------------------------------------------------------------

def test_analyze_problem_definite():
    report = analyze_problem("x**2", "x", "0", "3", True, True, True)
    assert report["kind"] == "definite" and report["numeric"] == 9.0
    assert report["verification"]["verified"] is True
    assert report["steps"] and report["comparison"]
    assert base64.b64decode(report["function_plot"])[:4] == b"\x89PNG"
    assert base64.b64decode(report["comparison_plot"])[:4] == b"\x89PNG"


def test_analyze_problem_indefinite():
    report = analyze_problem("sin(x)*cos(x)", "x", "", "", False, True, False)
    assert report["kind"] == "indefinite" and "+ C" in report["result"]
    assert report["verification"]["verified"] is True
    assert report["steps"] == [] and report["comparison"] is None


def test_analyze_problem_requires_both_bounds_or_neither():
    with pytest.raises(ValueError):
        analyze_problem("x", "x", "0", "", False, False, False)


def test_analyze_problem_with_timeout_matches_in_process_result():
    local = analyze_problem("x**2", "x", "0", "3", True, True, False)
    isolated = analyze_problem("x**2", "x", "0", "3", True, True, False, timeout=20)
    assert local["result"] == isolated["result"]
    assert local["steps"] == isolated["steps"]


def test_slow_steps_do_not_cost_you_the_answer():
    # The answer is quick, but SymPy's step-by-step rules hang on this one.
    report = analyze_problem("log(log(x))*log(log(log(x)))/(x*log(x))", "x", "", "",
                             True, False, False, timeout=8)
    assert "+ C" in report["result"]
    assert report["steps"] == [] and report["steps_error"]


def test_infinite_bounds_report_a_friendly_comparison_note():
    report = analyze_problem("exp(-x**2)/(1+exp(2*x))", "x", "-oo", "oo", False, True, True)
    assert abs(report["numeric"] - math.sqrt(math.pi) / 2) < 1e-8
    assert report["comparison"] is None and "finite" in report["comparison_error"]
    assert report["function_plot"] is not None and report["plot_error"] is None


# (expression, lower, upper) for the 20 problems in sample_problems.txt
SAMPLE_PROBLEMS = [
    ("sin(x)**2025 * cos(x)**2026", "-pi", "pi"),
    ("exp(2026*exp(x) + x)", "", ""),
    ("frac(floor(x)/3)", "0", "2026"),
    ("Abs(x + Abs(x + Abs(x + Abs(x))))", "-1", "1"),
    ("1/(sqrt(x+1) - sqrt(x-1))", "", ""),
    ("sqrt(1+cosh(x))", "", ""),
    ("2**log(x)/x**2", "", ""),
    ("x**2/(1-x)", "0", "1/2"),
    ("x**2*sin(x)", "", ""),
    ("(x-1)**2/(2*exp(x)+x**2+1)", "", ""),
    ("Max(0, sqrt(1-x**2) - 1/2)", "-1", "1"),
    ("x+1", "0", "1"),
    ("cos(x)**5 - 10*cos(x)**3*sin(x)**2 + 5*cos(x)*sin(x)**4", "", ""),
    ("atan(sqrt(x))", "", ""),
    ("floor(x) + ceiling(floor(x)) + floor(frac(x)) + frac(floor(x)) + ceiling(frac(x)) + frac(ceiling(x))", "0", "1000"),
    ("sqrt((cos(x)*cot(x)*csc(x))/(sin(x)*tan(x)*sec(x)))", "", ""),
    ("exp(-x**2)/(1+exp(2*x))", "-oo", "oo"),
    ("sin(x)**2/x**2 - sin(2*x)/x", "", ""),
    ("log(log(x))*log(log(log(x)))/(x*log(x))", "", ""),
    ("cos((pi/2)*cos((pi/2)*cos(x)**2)**2)**2", "0", "pi/2"),
]
# Problems whose symbolic integration is genuinely very slow: a clean timeout is acceptable.
SLOW_SAMPLES = {1, 16}


@pytest.mark.parametrize("number", range(1, 21))
def test_sample_problem_never_crashes(number):
    expression, lower, upper = SAMPLE_PROBLEMS[number - 1]
    try:
        report = analyze_problem(expression, "x", lower, upper, False, True, False, timeout=10)
    except CalculationTimeout:
        assert number in SLOW_SAMPLES, f"sample problem {number} timed out unexpectedly"
        return
    assert report["result"]


def test_known_sample_answers():
    answers = {
        1: 0.0,                      # odd integrand over a symmetric interval
        3: 675.0,
        8: math.log(2) - 5 / 8,
        12: 1.5,
        17: math.sqrt(math.pi) / 2,
        20: math.pi / 4,
    }
    for number, expected in answers.items():
        expression, lower, upper = SAMPLE_PROBLEMS[number - 1]
        report = analyze_problem(expression, "x", lower, upper, False, False, False, timeout=20)
        assert abs(report["numeric"] - expected) < 1e-6, f"sample problem {number}"


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    web.app.config["TESTING"] = True
    return web.app.test_client()


def post(client, expression, lower="", upper="", variable="x", steps="on"):
    data = {"expression": expression, "variable": variable, "lower": lower, "upper": upper}
    if steps:
        data["show_steps"] = steps
    return client.post("/", data=data)


def test_page_loads(client):
    assert client.get("/").status_code == 200


def test_indefinite_integral_shows_result_verification_and_steps(client):
    body = post(client, "x*exp(x)").get_data(as_text=True)
    assert "+ C" in body and "Verified" in body and "Integration by parts" in body


def test_definite_integral_shows_methods_and_plots(client):
    body = post(client, "sin(x)", "0", "3").get_data(as_text=True)
    assert "Numerical methods" in body and "Simpson" in body
    assert body.count("data:image/png;base64,") == 2


@pytest.mark.parametrize("text", HOSTILE_INPUTS)
def test_hostile_input_gets_an_error_message_not_a_server_error(client, text):
    response = post(client, text)
    assert response.status_code == 200
    assert 'class="error"' in response.get_data(as_text=True)


def test_bad_bounds_and_missing_bound_give_messages(client):
    assert "numeric" in post(client, "x**2", "a", "b").get_data(as_text=True).lower()
    assert "both bounds" in post(client, "x**2", "0", "").get_data(as_text=True)


def test_divergent_integral_does_not_crash(client):
    response = post(client, "1/x", "0", "1")
    assert response.status_code == 200 and "diverges" in response.get_data(as_text=True)


def test_hanging_integral_shows_timeout_message(client, monkeypatch):
    monkeypatch.setattr(web, "DEFAULT_TIMEOUT", 2)
    body = post(client, "sqrt((cos(x)*cot(x)*csc(x))/(sin(x)*tan(x)*sec(x)))").get_data(as_text=True)
    assert "took longer than 2 seconds" in body


def test_oversized_post_is_rejected(client):
    response = client.post("/", data={"expression": "x" * 100_000})
    assert response.status_code == 413
