"""Batch scoring runner.

    python -m worker.batch --due        every job whose time has come
    python -m worker.batch --job ID     one job, now

On AWS a scheduled ECS task runs `--due` every 15 minutes. Each job's
result is recorded on its run (and shown on the project page); a failed job
does not stop the others. Exits non-zero only when the runner itself cannot
work (no database).
"""
import argparse
import sys

from logic import deployment
from utils import db, log

_log = log.get("batch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--due", action="store_true")
    which.add_argument("--job")
    args = parser.parse_args(argv)
    log.setup()
    try:
        db.migrate()
    except Exception:
        log.event(_log, "batch_no_database", level=40, exc_info=True)
        return 1
    results = (deployment.run_due() if args.due
               else [deployment.run_job(args.job, triggered_by="manual")])
    for r in results:
        log.event(_log, "batch_job_ran", status=r["status"], rows=r["rows"])
    log.event(_log, "batch_done", jobs=len(results),
              failed=sum(r["status"] != "succeeded" for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
