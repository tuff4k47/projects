"""
project.py

An interactive integration calculator (command line).

Type a function, optionally give bounds, and the program shows:
  - the indefinite integral (+ C) or the definite integral over [a, b],
  - a check of the answer (differentiating it, or cross-checking numerically),
  - an optional step-by-step derivation (sum, power and constant-multiple
    rules, u-substitution, integration by parts, ...), and
  - for definite integrals, a comparison of numerical methods (left Riemann,
    midpoint, trapezoid, Simpson) showing how fast each converges.

The mathematics lives in integrator.py, shared with the Flask web app
(app.py). Everything here is about talking to the person at the keyboard:
reading input, formatting the report, and saving plots.
"""

import base64

from integrator import (
    DEFAULT_TIMEOUT,
    CalculationTimeout,
    analyze_problem,
    format_steps,
)


class QuitRequested(Exception):
    """Raised when the user types quit/exit at any prompt."""


def main():
    print("=== Integration Calculator ===")
    print("Enter a function of one variable (default variable: x).")
    print("Examples: x**2 + 3*x, x*exp(x), 1/(x**2 + 1), exp(-x**2)")
    print("Type 'quit' at any prompt to exit.\n")

    try:
        while True:
            expression = ask("f(x) = ")
            if not expression:
                continue
            variable = ask("Variable to integrate with respect to [x]: ") or "x"
            try:
                lower, upper = parse_bounds(
                    ask("Bounds 'a,b' for a definite integral (e.g. 0,1 or -oo,oo), or Enter for indefinite: ")
                )
            except ValueError as error:
                print(f"{error}\n")
                continue

            definite = bool(lower)
            show_steps = ask_yes_no("Show step-by-step derivation?")
            compare = definite and ask_yes_no("Compare numerical methods (Riemann, trapezoid, Simpson)?")
            plots = definite and ask_yes_no("Save plots as PNG files?")

            try:
                report = analyze_problem(
                    expression, variable, lower, upper,
                    show_steps, compare, plots, timeout=DEFAULT_TIMEOUT,
                )
            except CalculationTimeout as error:
                print(f"\n{error} Try a simpler expression or narrower bounds.\n")
                continue
            except ValueError as error:
                print(f"\nError: {error}\n")
                continue
            except Exception as error:  # noqa: BLE001 - keep the prompt loop alive
                print(f"\nSomething went wrong: {type(error).__name__}: {error}\n")
                continue

            print()
            print(format_report(report))
            if report.get("function_plot"):
                print(f"\nSaved {save_plot(report['function_plot'], 'integral_plot.png')}")
            if report.get("comparison_plot"):
                print(f"Saved {save_plot(report['comparison_plot'], 'method_comparison.png')}")
            print()
    except (QuitRequested, EOFError, KeyboardInterrupt):
        print("\nGoodbye!")


def ask(prompt):
    """Read one line of input; typing quit or exit ends the program."""
    answer = input(prompt).strip()
    if answer.lower() in ("quit", "exit"):
        raise QuitRequested
    return answer


def interpret_yes_no(text, default=False):
    """Turn typed text into True/False; blank or unrecognized input gives the default."""
    text = text.strip().lower()
    if text in ("y", "yes"):
        return True
    if text in ("n", "no"):
        return False
    return default


def ask_yes_no(prompt, default=False):
    """Ask a yes/no question, showing the default in capitals."""
    hint = "[Y/n]" if default else "[y/N]"
    return interpret_yes_no(ask(f"{prompt} {hint}: "), default)


def parse_bounds(text):
    """
    Split 'a,b' into two bound strings. Commas inside parentheses are
    ignored, so Max(0,1),2 works. Blank text means an indefinite integral
    and returns ("", ""). Raises ValueError unless there are exactly two parts.
    """
    text = text.strip()
    if not text:
        return "", ""
    parts, depth, current = [], 0, ""
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    parts.append(current.strip())
    if len(parts) != 2 or not all(parts):
        raise ValueError("Please enter bounds as two values separated by a comma, e.g. 0,1")
    return parts[0], parts[1]


def format_comparison_table(comparison):
    """Lay out each method's error for every n, plus its estimated order of accuracy."""
    methods = comparison["methods"]
    labels = [method["label"] for method in methods]
    width = max(12, *(len(label) + 2 for label in labels))
    lines = [f"True value: {comparison['true_value']:.10f}", ""]
    lines.append(f"{'n':>6} | " + " | ".join(f"{label:>{width}}" for label in labels))
    lines.append("-" * len(lines[-1]))
    for index, row in enumerate(methods[0]["rows"]):
        errors = [f"{method['rows'][index]['error']:.2e}" for method in methods]
        lines.append(f"{row['n']:>6} | " + " | ".join(f"{error:>{width}}" for error in errors))
    lines.append("-" * len(lines[-1]))
    orders = [
        f"{method['order']:.2f}" if method.get("order_status", "ok") == "ok" and method["order"] is not None
        else method.get("order_status", "exact")
        for method in methods
    ]
    lines.append(f"{'order':>6} | " + " | ".join(f"{order:>{width}}" for order in orders))
    lines.append("(order p means the error shrinks like 1/n^p; higher is faster. 'irregular' = no clean power law)")
    return "\n".join(lines)


def format_report(report):
    """Turn the dict returned by analyze_problem() into text for the terminal."""
    lines = [report["result"]]

    check = report.get("verification")
    if check:
        if check["verified"] is True:
            lines.append(f"✓ Verified: {check['method']}.")
        elif check["verified"] is False:
            lines.append(f"⚠ Check failed: {check['method']}.")
        else:
            lines.append(f"(Not verified: {check['method']}.)")

    if report.get("steps"):
        lines += ["", "Step-by-step:", format_steps(report["steps"])]
    elif report.get("steps_error"):
        lines += ["", report["steps_error"]]

    if report.get("comparison"):
        lines += ["", "Numerical methods:", format_comparison_table(report["comparison"])]
    elif report.get("comparison_error"):
        lines += ["", f"Numerical methods unavailable: {report['comparison_error']}"]

    if report.get("plot_error"):
        lines += ["", report["plot_error"]]
    return "\n".join(lines)


def save_plot(encoded_png, filename):
    """Write a base64-encoded PNG (as produced by integrator's plot functions) to a file."""
    with open(filename, "wb") as file:
        file.write(base64.b64decode(encoded_png))
    return filename


if __name__ == "__main__":
    main()
