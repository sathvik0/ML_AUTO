"""Background training jobs: lifecycle, claiming, cancel and retry,
interrupted workers, and the SQS worker loop — on a throwaway database."""
import json
import time
from datetime import timedelta

import pandas as pd
import pytest

from logic import jobs, ml, projects
from logic.projects import AccessDenied
from utils import db
from worker import trainer

ANA = {"sub": "a", "email": "ana@example.com", "role": "analyst"}
VIC = {"sub": "v", "email": "vic@example.com", "role": "viewer"}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "objects"))
    monkeypatch.setenv("JOBS_MODE", "none")       # tests run jobs by hand
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("JOBS_QUEUE_URL", raising=False)
    monkeypatch.setattr(jobs, "PROGRESS_EVERY", 0.0)
    db.migrate()


CONFIG = {"task": "classification", "target": "Churned", "positive": "Yes",
          "target_rule": None,
          "numeric": ["Age", "Tenure_Years", "Service_Calls_12M"],
          "categorical": ["Region"],
          "prep": {"num_missing": "median", "cat_missing": "category",
                   "scale": "standard", "rare_share": 0.0,
                   "missing_flags": False, "log_target": False, "pca": 0,
                   "cat_encoding": "onehot"},
          "validation": {"test_size": 0.2, "folds": 3, "stratify": True,
                         "group": None},
          "seed": 1, "metric": None, "balance": False,
          "models": [{"key": k, "params": ml.default_params(
              "classification", k), "search": None}
              for k in ("logistic", "tree")], "k_scan": None}


@pytest.fixture
def project():
    pid = projects.create_project(ANA, "Jobs")["id"]
    projects.share(ANA, pid, VIC["email"], "viewer")
    df = pd.read_csv("sample_data/insurance_churn.csv").head(1500)
    did = projects.save_dataset(ANA, pid, df, "churn.csv", "upload")
    return {"pid": pid, "did": did}


def submit(project, user=ANA, steps=()):
    return jobs.submit_training(user, project["pid"], CONFIG, list(steps),
                                project["did"])


def test_job_trains_and_saves_a_run(project):
    jid = submit(project, steps=[{"kind": "drop_duplicates"}])
    assert jobs.get(ANA, project["pid"], jid)["status"] == "queued"
    assert jobs.run_job(jid, "w1") == "succeeded"
    j = jobs.get(ANA, project["pid"], jid)
    assert (j["progress"], j["attempts"]) == (1.0, 1)
    rid = j["result"]["run_id"]
    run = projects.load_run(ANA, project["pid"], rid)
    assert run["best"] in ("logistic", "tree")
    assert run["prep_steps"] == [{"kind": "drop_duplicates"}]
    assert run["frame"] is not None and run["importance"]


def test_viewer_cannot_submit(project):
    with pytest.raises(AccessDenied):
        submit(project, user=VIC)


def test_claimed_once(project):
    jid = submit(project)
    assert jobs.claim(jid, "w1") is True
    assert jobs.claim(jid, "w2") is False          # w1 is alive
    assert jobs.state(jid) == "busy"


def test_interrupted_job_is_claimed_again(project):
    jid = submit(project)
    assert jobs.claim(jid, "w1")
    later = jobs._now() + jobs.STALE + timedelta(seconds=5)
    assert jobs.stale_jobs(later) == [jid]
    assert jobs.state(jid, later) == "open"
    assert jobs.claim(jid, "w2", now=later) is True
    assert jobs.get(ANA, project["pid"], jid)["attempts"] == 2


def test_gives_up_after_max_attempts(project):
    jid = submit(project)
    t = jobs._now()
    for i in range(jobs.MAX_ATTEMPTS):
        t = t + jobs.STALE + timedelta(seconds=5)
        assert jobs.claim(jid, f"w{i}", now=t)
    t = t + jobs.STALE + timedelta(seconds=5)
    assert jobs.claim(jid, "w-last", now=t) is False
    j = jobs.get(ANA, project["pid"], jid)
    assert j["status"] == "failed" and "Interrupted" in j["message"]


def test_cancel_before_start(project):
    jid = submit(project)
    jobs.request_cancel(ANA, project["pid"], jid)
    assert jobs.get(ANA, project["pid"], jid)["status"] == "cancelled"
    assert jobs.run_job(jid, "w1") is None


def test_cancel_while_training(project, monkeypatch):
    jid = submit(project)
    real = jobs.report
    calls = []

    def report(job_id, fraction, text):
        calls.append(fraction)
        if len(calls) == 3:           # the user presses Cancel mid-run
            jobs.request_cancel(ANA, project["pid"], job_id)
        return real(job_id, fraction, text)
    monkeypatch.setattr(jobs, "report", report)
    assert jobs.run_job(jid, "w1") == "cancelled"
    assert projects.list_runs(ANA, project["pid"]) == []


def test_retry_makes_a_new_job(project):
    jid = submit(project)
    jobs.request_cancel(ANA, project["pid"], jid)
    again = jobs.retry(ANA, project["pid"], jid)
    assert again != jid
    assert jobs.run_job(again, "w1") == "succeeded"
    with pytest.raises(AccessDenied):
        jobs.retry(ANA, project["pid"], jobs.submit_training(
            ANA, project["pid"], CONFIG, [], project["did"]))  # still queued


def test_access_rechecked_when_it_runs(project):
    jid = submit(project)
    projects.share(ANA, project["pid"], "boss@example.com", "owner")
    projects.unshare(ANA, project["pid"], ANA["email"])  # Ana loses access
    assert jobs.run_job(jid, "w1") == "failed"
    with db.session() as s:
        assert "not shared with you" in s.get(db.Job, jid).message


class FakeSqs:
    def __init__(self):
        self.deleted, self.hidden, self.sent = [], [], []

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append(ReceiptHandle)

    def change_message_visibility(self, QueueUrl, ReceiptHandle,
                                  VisibilityTimeout):
        self.hidden.append(ReceiptHandle)

    def send_message(self, QueueUrl, MessageBody):
        self.sent.append(json.loads(MessageBody))


def msg(job_id, receipt="r1"):
    return {"ReceiptHandle": receipt, "Body": json.dumps({"job_id": job_id})}


def test_worker_runs_then_deletes(project):
    sqs = FakeSqs()
    jid = submit(project)
    assert trainer.handle(sqs, "q", msg(jid)) == "ran"
    assert sqs.deleted == ["r1"]
    assert jobs.get(ANA, project["pid"], jid)["status"] == "succeeded"


def test_worker_defers_a_job_running_elsewhere(project):
    sqs = FakeSqs()
    jid = submit(project)
    jobs.claim(jid, "other-worker")
    assert trainer.handle(sqs, "q", msg(jid)) == "deferred"
    assert sqs.deleted == [] and sqs.hidden == ["r1"]


def test_worker_drops_finished_and_malformed(project):
    sqs = FakeSqs()
    jid = submit(project)
    jobs.request_cancel(ANA, project["pid"], jid)
    assert trainer.handle(sqs, "q", msg(jid, "r2")) == "dropped"
    assert trainer.handle(sqs, "q", {"ReceiptHandle": "r3",
                                     "Body": "not json"}) == "malformed"
    assert sqs.deleted == ["r2", "r3"]


def test_queue_dispatch_sends_the_job_id(project, monkeypatch):
    sqs = FakeSqs()
    import boto3
    monkeypatch.setattr(boto3, "client", lambda name: sqs)
    monkeypatch.setenv("JOBS_QUEUE_URL", "https://sqs/queue")
    monkeypatch.delenv("JOBS_MODE")
    jid = submit(project)
    assert sqs.sent == [{"job_id": jid}]


def test_thread_dispatch_runs_locally(project, monkeypatch):
    monkeypatch.setenv("JOBS_MODE", "thread")
    jid = submit(project)
    for _ in range(120):
        if jobs.get(ANA, project["pid"], jid)["status"] not in jobs.ACTIVE:
            break
        time.sleep(0.5)
    assert jobs.get(ANA, project["pid"], jid)["status"] == "succeeded"
