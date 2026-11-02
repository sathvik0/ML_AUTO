"""Object storage for datasets, training runs and registered models.

`STORAGE_BUCKET` set: Amazon S3 (encrypted with the bucket's own KMS key,
versioned; the ECS task role is the only writer). Otherwise a local folder, `LOCAL_STORAGE_DIR` or
data/objects, for local work. Keys are the same either way:

    projects/<project>/datasets/<id>.parquet
    projects/<project>/runs/<id>.joblib
    projects/<project>/models/<name>/v<version>.joblib
"""
import functools
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class StorageError(Exception):
    pass


def _bucket():
    return os.environ.get("STORAGE_BUCKET")


@functools.lru_cache(maxsize=1)
def _s3():
    import boto3
    return boto3.client("s3")


def _local_root():
    root = Path(os.environ.get("LOCAL_STORAGE_DIR") or ROOT / "data"
                / "objects")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _local_path(key):
    root = _local_root().resolve()
    path = (root / key).resolve()
    if root not in path.parents:
        raise StorageError("key outside the storage folder")
    return path


def put(key, data: bytes):
    if _bucket():
        # No encryption header: the bucket's default (its own KMS key)
        # applies. Naming "aws:kms" without a key id would use the AWS
        # managed key instead.
        _s3().put_object(Bucket=_bucket(), Key=key, Body=data)
        return
    path = _local_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get(key) -> bytes:
    if _bucket():
        try:
            return _s3().get_object(Bucket=_bucket(), Key=key)["Body"].read()
        except Exception as e:
            raise StorageError(f"could not read {key}") from e
    path = _local_path(key)
    if not path.exists():
        raise StorageError(f"no object {key}")
    return path.read_bytes()


def delete_prefix(prefix):
    """Every object under `prefix` (a deleted project's files)."""
    if _bucket():
        pages = _s3().get_paginator("list_objects_v2").paginate(
            Bucket=_bucket(), Prefix=prefix)
        for page in pages:
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                _s3().delete_objects(Bucket=_bucket(),
                                     Delete={"Objects": keys})
        return
    root = _local_path(prefix)
    if root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        root.rmdir()
