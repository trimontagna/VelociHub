"""Run the benchmark and write its report, or write the test measurements:

    python -m velocihub.benchmark [cases ...] [--levels preprocessing post processing images chain] [--scenarios ...]
                                  [--out folder] [--compare previous] [--spicy n_k=8 eps_l=0.5] [--no-open] [--json]
    python -m velocihub.benchmark --test-workspace [folder] [cases ...]

Cases: uniform cylinder jet modes blasius turbulence (all by default). Levels: preprocessing (the background removal
test), post (the exact fields post-processed), processing (particle links and vectors processed, then post-processed),
images (particle images pre-processed with several recipes, then the PIV and the PTV; the recipes compared on harder
images), chain (images, pre-processing, PIV/PTV with and without SPICY, post-processing); all by default. The report
goes to benchmark_reports/report_<date>_<time> in the repository by default (report.reports_folder) and is opened in
the browser.
--compare: a previous report (folder or results.json) shown beside. --spicy: SPICY settings changed for all the cases
(a study of the settings; the checks keep their tolerances). --test-workspace: the particle images of the cases as
measurements, with their exact fields and a workspace (~/VelociHub/benchmarks/test_workspace by default). --json: the
messages as JSON lines (the interface reads them). Exit code 1 when a check is not as expected."""

import argparse
import json
import sys
import webbrowser


def main(argv=None):
    from .cases import CASES, LEVELS, SCENARIOS

    parser = argparse.ArgumentParser(prog="python -m velocihub.benchmark", description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help=f"cases to run: {' '.join(CASES)} (all by default)")
    parser.add_argument("--levels", nargs="+", default=None, help=f"{' '.join(LEVELS)} (all by default)")
    parser.add_argument("--scenarios", nargs="+", default=None, help=f"{' '.join(SCENARIOS)} (all by default)")
    parser.add_argument("--out", default=None, help="folder of the report")
    parser.add_argument("--compare", default=None, help="a previous report to compare with")
    parser.add_argument("--spicy", nargs="+", default=(), metavar="KEY=VALUE",
                        help="SPICY settings changed for all the cases")
    parser.add_argument("--test-workspace", nargs="?", const="", default=None, metavar="FOLDER",
                        help="write the test measurements (particle images) instead")
    parser.add_argument("--recording", default=None, metavar="FOLDER",
                        help="compare the pre-processing recipes on the images of a real recording instead")
    parser.add_argument("--frames", type=int, default=None, help="--recording: its first frames only")
    parser.add_argument("--recipes", nargs="+", default=None, help="--recording: the recipes (all by default)")
    parser.add_argument("--work", default=None, help="--recording: the folder of its runs, kept")
    parser.add_argument("--no-open", action="store_true", help="don't open the report in the browser")
    parser.add_argument("--json", action="store_true", help="messages as JSON lines")
    args = parser.parse_args(argv)

    def say(type_, **kw):
        if args.json:
            print(json.dumps({"type": type_, **kw}), flush=True)
        elif "text" in kw:
            print(kw["text"], flush=True)

    unknown = ([c for c in args.cases if c not in CASES] + [s for s in (args.scenarios or ()) if s not in SCENARIOS]
               + [lv for lv in (args.levels or ()) if lv not in LEVELS])
    if unknown:
        say("error", text=f"unknown case, scenario or level: {', '.join(unknown)}")
        return 2
    try:
        if args.test_workspace is not None:
            from .images import test_workspace
            file = test_workspace(args.test_workspace or None, args.cases or None,
                                  report=lambda text: say("progress", text=text))
            say("done", workspace=file, text=f"Test workspace written: {file}")
            return 0
        if args.recording is not None:
            from .image_level import HARD_RECIPES, RECIPES
            from .recording import compare_recording, default_folder, write_recording_report
            recipes = args.recipes or HARD_RECIPES
            bad = [r for r in recipes if r not in RECIPES]
            if bad:
                raise ValueError(f"unknown recipe {', '.join(bad)}: {', '.join(RECIPES)}")
            result = compare_recording(args.recording, args.work, recipes, args.frames,
                                       report=lambda text: say("progress", text=text),
                                       progress=lambda done, total: say("level", level="recording", done=done,
                                                                        total=total), keep=args.work is not None)
            index = write_recording_report(result, args.out or default_folder(args.recording))
            say("done", report=index, text=f"Recording compared ({result.seconds / 60:.0f} min): {index}")
            if not args.no_open:
                webbrowser.open(index)
            return 0
        from . import report
        from .regression_level import run_benchmark

        if args.spicy:
            from ..piv.regression import SPICY_PARAMETERS
            params = {p.key: p for p in SPICY_PARAMETERS}
            changes = {}
            for item in args.spicy:
                key, _, value = item.partition("=")
                if key not in params:
                    raise ValueError(f"unknown SPICY setting {key}: {', '.join(params)}")
                changes[key] = params[key].parse(value)
            for case in CASES.values():
                case.regression = {**case.regression, **changes}
        levels = tuple(args.levels or LEVELS)
        # The pre-processing level alone: the background removal test, no case
        cases = args.cases or (None if set(levels) - {"preprocessing"} else [])
        say("levels", levels=list(levels))
        result = run_benchmark(cases, args.scenarios, report=lambda text: say("progress", text=text), levels=levels,
                               progress=lambda level, done, total: say("level", level=level, done=done,
                                                                       total=total))
        say("progress", text="Writing the report")
        index = report.write_report(result, args.out or report.default_folder(), compare=args.compare)
        counts = result.counts()
        failures = result.failures()
        say("done", report=index, counts=counts, failures=len(failures),
            text=f"{counts['pass']} checks passed, {counts['fail']} failed, {counts['known']} known issues"
                 f"{', ' + str(counts['fixed']) + ' known issues now passing' if counts['fixed'] else ''}: {index}")
        if not args.no_open:
            webbrowser.open(index)
        return 1 if failures else 0
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        say("error", text=f"{type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
