"""Scheduled work: batch scoring jobs and model monitors.

    python -m worker.batch --due        every job and monitor that is due
    python -m worker.batch --job ID     one batch job, now
    python -m worker.batch --monitor ID one monitor check, now

On AWS a scheduled ECS task runs `--due` every 15 minutes. Each result is
recorded on its run (and shown on the project page); a failure does not
stop the others. Exits non-zero only when the runner itself cannot
work (no database).
"""
import argparse
import sys

from logic import deployment, monitoring, prediction_log
from utils import db, log

_log = log.get("batch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--due", action="store_true")
    which.add_argument("--job")
    which.add_argument("--monitor")
    args = parser.parse_args(argv)
    log.setup()
    try:
        db.migrate()
    except Exception:
        log.event(_log, "batch_no_database", level=40, exc_info=True)
        return 1
    if args.due:
        results = deployment.run_due() + monitoring.run_due()
        try:
            prediction_log.purge()
        except Exception:
            log.event(_log, "prediction_log_purge_failed", level=40,
                      exc_info=True)
    elif args.job:
        results = [deployment.run_job(args.job, triggered_by="manual")]
    else:
        results = [monitoring.run_monitor(args.monitor,
                                          triggered_by="manual")]
    for r in results:
        log.event(_log, "scheduled_work_ran", status=r["status"])
    log.event(_log, "batch_done", jobs=len(results),
              failed=sum(r["status"] == "failed" for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
