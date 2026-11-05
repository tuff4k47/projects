"""
app.py

A small Flask web front end for the same engine the command-line version
uses. Run it from inside the project folder with:

    python3 app.py            (or: flask --app app run)

then open http://127.0.0.1:5000 in a browser.

Safety: expressions are parsed through a whitelist (never eval'd as Python),
and every calculation runs in a separate process that is stopped after
DEFAULT_TIMEOUT seconds, so one slow integral cannot freeze the server.
Debug mode is off unless you set FLASK_DEBUG=1, because Flask's debugger
lets anyone who can reach it run code.
"""

import os

from flask import Flask, render_template, request

from integrator import (
    DEFAULT_TIMEOUT,
    CalculationTimeout,
    analyze_problem,
    steps_with_depth,
    warm_up,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # form posts are tiny; refuse anything larger


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "expression": "x*exp(x)",
        "variable": "x",
        "lower": "0",
        "upper": "1",
        "show_steps": True,
    }

    if request.method == "POST":
        context.update(
            expression=request.form.get("expression", "").strip(),
            variable=request.form.get("variable", "x").strip() or "x",
            lower=request.form.get("lower", "").strip(),
            upper=request.form.get("upper", "").strip(),
            show_steps=request.form.get("show_steps") == "on",
        )
        try:
            report = analyze_problem(
                context["expression"], context["variable"], context["lower"], context["upper"],
                context["show_steps"], True, True,
                timeout=DEFAULT_TIMEOUT,
            )
        except CalculationTimeout as error:
            context["error"] = f"{error} Try a simpler expression or narrower bounds."
        except ValueError as error:
            context["error"] = str(error)
        except Exception:  # noqa: BLE001 - never show a raw 500 page for bad input
            app.logger.exception("Unexpected error while computing an integral")
            context["error"] = "Something went wrong while computing that. Try a different expression."
        else:
            context["report"] = report
            context["steps"] = steps_with_depth(report["steps"])

    return render_template("index.html", **context)


if __name__ == "__main__":
    warm_up()
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
