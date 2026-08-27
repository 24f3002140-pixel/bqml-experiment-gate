import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SAFE_INT_MAX = 9007199254740991

# runId -> frozen selection
RUNS: dict[str, dict[str, Any]] = {}

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def finite(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def timestamp_ok(x):
    if not isinstance(x, str):
        return False

    if not TS_RE.fullmatch(x):
        return False

    try:
        if x.endswith("Z"):
            datetime.fromisoformat(x[:-1] + "+00:00")
        else:
            datetime.fromisoformat(x)
        return True
    except Exception:
        return False


def to_utc(x):
    if x.endswith("Z"):
        dt = datetime.fromisoformat(x[:-1] + "+00:00")
    else:
        dt = datetime.fromisoformat(x)

    return dt.astimezone(timezone.utc)


def bkey(x):
    return x.encode("utf-8")


def sorted_utf8(values):
    return sorted(values, key=bkey)


def compact(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_digest(train_ids, eval_ids, features):
    # Exact key order required by contract.
    obj = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
    }

    return hashlib.sha256(
        compact(obj).encode("utf-8")
    ).hexdigest()


# =========================================================
# SELECTION ROW VALIDATION
# =========================================================

def validate_selection(body):
    required = {
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials",
    }

    if set(body.keys()) != required:
        return False

    if body["phase"] != "select":
        return False

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return False

    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return False

    if not all(isinstance(x, str) for x in forbidden):
        return False

    if len(set(forbidden)) != len(forbidden):
        return False

    limit = body["numTrialsLimit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > SAFE_INT_MAX
    ):
        return False

    rows = body["rows"]

    if not isinstance(rows, list) or not rows:
        return False

    ids = set()

    for row in rows:

        if not isinstance(row, dict):
            return False

        expected = {
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        }

        if set(row.keys()) != expected:
            return False

        row_id = row["id"]

        if not isinstance(row_id, str) or not row_id:
            return False

        if row_id in ids:
            return False

        ids.add(row_id)

        if not isinstance(row["entity"], str):
            return False

        if not timestamp_ok(row["eventTime"]):
            return False

        if not timestamp_ok(row["predictionTime"]):
            return False

        if not safe_int(row["version"]):
            return False

        if row["split"] not in ("TRAIN", "EVAL"):
            return False

        # Important:
        # eventTime after predictionTime is leakage.
        if to_utc(row["eventTime"]) > to_utc(
            row["predictionTime"]
        ):
            return False

        features = row["features"]

        if not isinstance(features, dict):
            return False

        for name, feature in features.items():

            if not isinstance(name, str):
                return False

            if not isinstance(feature, dict):
                return False

            if set(feature.keys()) != {
                "value",
                "availableAt",
            }:
                return False

            if not timestamp_ok(
                feature["availableAt"]
            ):
                return False

    # =====================================================
    # TRIAL VALIDATION
    # =====================================================

    trials = body["trials"]

    if not isinstance(trials, list):
        return False

    trial_ids = set()

    for trial in trials:

        if not isinstance(trial, dict):
            return False

        if "trialId" not in trial:
            return False

        if "status" not in trial:
            return False

        if not safe_int(trial["trialId"]):
            return False

        if trial["trialId"] in trial_ids:
            return False

        trial_ids.add(trial["trialId"])

        if trial["status"] not in (
            "SUCCEEDED",
            "FAILED",
        ):
            return False

        if trial["status"] == "SUCCEEDED":

            if "evalMetric" not in trial:
                return False

            if not finite(trial["evalMetric"]):
                return False

        elif "evalMetric" in trial:

            if not finite(trial["evalMetric"]):
                return False

    return True


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate(rows):
    groups = {}

    for row in rows:

        key = (
            row["entity"],
            to_utc(row["eventTime"]),
        )

        old = groups.get(key)

        if old is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > old["version"]:
            groups[key] = row

        # Equal version -> smallest UTF-8 ID.
        elif row["version"] == old["version"]:
            if bkey(row["id"]) < bkey(old["id"]):
                groups[key] = row

    return list(groups.values())


# =========================================================
# SHARED FEATURES
# =========================================================

def shared_features(rows, forbidden):

    if not rows:
        return []

    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    result = []

    for name in common:

        if name in forbidden:
            continue

        eligible = True

        for row in rows:

            available = to_utc(
                row["features"][name]["availableAt"]
            )

            prediction = to_utc(
                row["predictionTime"]
            )

            # Feature is simply NOT eligible.
            # It does not invalidate the entire row.
            if available > prediction:
                eligible = False
                break

        if eligible:
            result.append(name)

    return sorted_utf8(result)


# =========================================================
# TRIAL SELECTION
# =========================================================

def select_trial(trials):

    candidates = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get("evalMetric")

        if not finite(metric):
            continue

        candidates.append(trial)

    if not candidates:
        return None

    # Highest metric.
    # Exact tie -> smallest trialId.
    candidates.sort(
        key=lambda x: (
            -float(x["evalMetric"]),
            x["trialId"],
        )
    )

    return candidates[0]


# =========================================================
# SELECT PHASE
# =========================================================

def do_select(body):

    reason_codes = []

    if len(body["trials"]) > body["numTrialsLimit"]:
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    retained = deduplicate(body["rows"])

    train_ids = sorted_utf8(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ]
    )

    eval_ids = sorted_utf8(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ]
    )

    features = shared_features(
        retained,
        set(body["forbiddenFeatures"]),
    )

    selected = select_trial(
        body["trials"]
    )

    if selected is None:
        reason_codes.append(
            "NO_SUCCESSFUL_TRIAL"
        )

    reason_codes = sorted_utf8(
        set(reason_codes)
    )

    # Failed selection has null ID and null digest.
    if reason_codes:

        return {
            "runId": body["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": features,
            "datasetDigest": None,
            "reasonCodes": reason_codes,
        }

    digest = make_digest(
        train_ids,
        eval_ids,
        features,
    )

    return {
        "runId": body["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
        "datasetDigest": digest,
        "reasonCodes": [],
    }


# =========================================================
# EVALUATION INPUT
# =========================================================

def validate_evaluation(body):

    expected = {
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes",
    }

    if set(body.keys()) != expected:
        return False

    if body["phase"] != "evaluate":
        return False

    if (
        not isinstance(body["runId"], str)
        or not body["runId"]
    ):
        return False

    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    if (
        not isinstance(body["datasetDigest"], str)
        or not HEX64_RE.fullmatch(
            body["datasetDigest"]
        )
    ):
        return False

    if not finite(body["metricFloor"]):
        return False

    if not (
        0 <= float(body["metricFloor"]) <= 1
    ):
        return False

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():

        if not isinstance(name, str) or not name:
            return False

        if not finite(floor):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    if not isinstance(body["rows"], list):
        return False

    if not safe_int(
        body["bytesProcessed"]
    ):
        return False

    if not safe_int(
        body["maxBytes"]
    ):
        return False

    return True


# =========================================================
# TEST ROWS
# =========================================================

def valid_test_rows(rows):

    for row in rows:

        if not isinstance(row, dict):
            return False

        if set(row.keys()) != {
            "label",
            "prediction",
            "slice",
        }:
            return False

        if (
            not isinstance(row["label"], int)
            or isinstance(row["label"], bool)
            or row["label"] not in (0, 1)
        ):
            return False

        if (
            not isinstance(row["prediction"], int)
            or isinstance(row["prediction"], bool)
            or row["prediction"] not in (0, 1)
        ):
            return False

        if (
            not isinstance(row["slice"], str)
            or not row["slice"]
        ):
            return False

    return True


# =========================================================
# EVALUATE PHASE
# =========================================================

def do_evaluate(body):

    reason_codes = []

    stored = RUNS.get(body["runId"])

    lineage_ok = (
        stored is not None
        and stored["successful"] is True
        and stored["response"][
            "selectedTrialId"
        ] == body["selectedTrialId"]
        and stored["response"][
            "datasetDigest"
        ] == body["datasetDigest"]
    )

    if not lineage_ok:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    rows = body["rows"]

    rows_ok = valid_test_rows(rows)

    if not rows_ok:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    bytes_ok = (
        body["bytesProcessed"]
        <= body["maxBytes"]
    )

    if not bytes_ok:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    test_metric = None
    critical_slice_pass = False

    # Only perform metric/slice checks when
    # rows exist and every row is valid.
    if rows and rows_ok:

        correct = sum(
            row["label"] == row["prediction"]
            for row in rows
        )

        test_metric = round(
            correct / len(rows),
            12,
        )

        if test_metric < float(
            body["metricFloor"]
        ):
            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        slice_pass = True

        for slice_name in sorted_utf8(
            body["requiredSlices"].keys()
        ):

            slice_rows = [
                row
                for row in rows
                if row["slice"] == slice_name
            ]

            if not slice_rows:

                reason_codes.append(
                    f"MISSING_SLICE:{slice_name}"
                )

                slice_pass = False
                continue

            correct_slice = sum(
                row["label"]
                == row["prediction"]
                for row in slice_rows
            )

            metric = round(
                correct_slice / len(slice_rows),
                12,
            )

            if metric < float(
                body["requiredSlices"][
                    slice_name
                ]
            ):

                reason_codes.append(
                    f"SLICE_FLOOR:{slice_name}"
                )

                slice_pass = False

        critical_slice_pass = slice_pass

    # Contract requirements.
    if (
        not lineage_ok
        or not rows
        or not rows_ok
    ):
        critical_slice_pass = False

    if any(
        x.startswith("MISSING_SLICE:")
        or x.startswith("SLICE_FLOOR:")
        for x in reason_codes
    ):
        critical_slice_pass = False

    admit = (
        lineage_ok
        and rows_ok
        and bool(rows)
        and test_metric is not None
        and test_metric
        >= float(body["metricFloor"])
        and critical_slice_pass
        and bytes_ok
    )

    if admit:
        decision = "admit"
    else:
        decision = "reject"

    return {
        "runId": body["runId"],
        "selectedTrialId": body[
            "selectedTrialId"
        ],
        "datasetDigest": body[
            "datasetDigest"
        ],
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": body[
            "bytesProcessed"
        ],
        "reasonCodes": sorted_utf8(
            set(reason_codes)
        ),
    }


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/bqml")
async def bqml(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    phase = body.get("phase")

    # Missing/unknown phase.
    if phase not in (
        "select",
        "evaluate",
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # -----------------------------------------------------
    # SELECT
    # -----------------------------------------------------

    if phase == "select":

        if not validate_selection(body):
            return JSONResponse(
                {"error": "INVALID_INPUT"},
                status_code=400,
            )

        run_id = body["runId"]

        fingerprint = hashlib.sha256(
            compact(body)
            .encode("utf-8")
        ).hexdigest()

        if run_id in RUNS:

            old = RUNS[run_id]

            if old["fingerprint"] == fingerprint:
                return JSONResponse(
                    old["response"]
                )

            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = do_select(body)

        successful = (
            response["selectedTrialId"]
            is not None
            and response["datasetDigest"]
            is not None
            and response["reasonCodes"] == []
        )

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
            "successful": successful,
        }

        return JSONResponse(response)

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    if phase == "evaluate":

        if not validate_evaluation(body):

            return JSONResponse(
                {
                    "runId": body.get("runId"),
                    "selectedTrialId": body.get(
                        "selectedTrialId"
                    ),
                    "datasetDigest": body.get(
                        "datasetDigest"
                    ),
                    "testMetric": None,
                    "criticalSlicePass": False,
                    "decision": "reject",
                    "bytesProcessed": body.get(
                        "bytesProcessed"
                    ),
                    "reasonCodes": [
                        "INVALID_INPUT"
                    ],
                },
                status_code=400,
            )

        return JSONResponse(
            do_evaluate(body)
        )


@app.get("/")
def root():
    return {"status": "ok"}