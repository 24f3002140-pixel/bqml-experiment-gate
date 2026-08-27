from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any
from datetime import datetime
import hashlib
import json
import math
import re

app = FastAPI()

# Stateful storage for the lifetime of the service.
RUNS: dict[str, dict[str, Any]] = {}

SAFE_INT_MAX = 9007199254740991

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def utf8_sorted(values):
    return sorted(values, key=lambda x: x.encode("utf-8"))


def is_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_INT_MAX
    )


def is_finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def valid_timestamp(value):
    if not isinstance(value, str) or not TS_RE.fullmatch(value):
        return False

    try:
        if value.endswith("Z"):
            datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            datetime.fromisoformat(value)
        return True
    except Exception:
        return False


def utc_timestamp_key(value):
    """
    Return a canonical UTC datetime for timestamp comparison/deduplication.
    """
    if value.endswith("Z"):
        dt = datetime.fromisoformat(value[:-1] + "+00:00")
    else:
        dt = datetime.fromisoformat(value)

    return dt.astimezone(__import__("datetime").timezone.utc)


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_dataset(train_ids, eval_ids, feature_names):
    # Exact key order required by the contract.
    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


def response_fingerprint(payload):
    """
    Fingerprint the selection input for replay/conflict detection.
    Compact JSON + deterministic key ordering.
    """
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def selection_input_valid(body):
    required = [
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials",
    ]

    if not all(k in body for k in required):
        return False

    if body.get("phase") != "select":
        return False

    run_id = body.get("runId")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
        return False

    forbidden = body.get("forbiddenFeatures")
    if not isinstance(forbidden, list):
        return False

    if not all(isinstance(x, str) for x in forbidden):
        return False

    limit = body.get("numTrialsLimit")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > SAFE_INT_MAX
    ):
        return False

    rows = body.get("rows")
    trials = body.get("trials")

    if not isinstance(rows, list) or not rows:
        return False

    if not isinstance(trials, list):
        return False

    # Selection rows must be TRAIN/EVAL only.
    seen_row_ids = set()

    for row in rows:
        if not isinstance(row, dict):
            return False

        for key in (
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        ):
            if key not in row:
                return False

        if not isinstance(row["id"], str) or not row["id"]:
            return False

        if row["id"] in seen_row_ids:
            return False
        seen_row_ids.add(row["id"])

        if not isinstance(row["entity"], str):
            return False

        if not valid_timestamp(row["eventTime"]):
            return False

        if not valid_timestamp(row["predictionTime"]):
            return False

        if not is_safe_int(row["version"]):
            return False

        if row["split"] not in ("TRAIN", "EVAL"):
            return False

        features = row["features"]
        if not isinstance(features, dict):
            return False

        for fname, fvalue in features.items():
            if not isinstance(fname, str):
                return False

            if not isinstance(fvalue, dict):
                return False

            if "value" not in fvalue or "availableAt" not in fvalue:
                return False

            if not valid_timestamp(fvalue["availableAt"]):
                return False

    seen_trial_ids = set()

    for trial in trials:
        if not isinstance(trial, dict):
            return False

        if "trialId" not in trial:
            return False

        if "status" not in trial:
            return False

        if trial["status"] not in ("SUCCEEDED", "FAILED"):
            return False

        if not is_safe_int(trial["trialId"]):
            return False

        if trial["trialId"] in seen_trial_ids:
            return False

        seen_trial_ids.add(trial["trialId"])

        if trial["status"] == "SUCCEEDED":
            if "evalMetric" not in trial:
                return False

            # Non-finite successful trials simply aren't eligible,
            # but malformed types are invalid input.
            if not isinstance(
                trial["evalMetric"], (int, float)
            ) or isinstance(trial["evalMetric"], bool):
                return False

    return True


def perform_selection(body):
    run_id = body["runId"]
    forbidden = set(body["forbiddenFeatures"])
    rows = body["rows"]
    trials = body["trials"]

    reason_codes = []

    if len(trials) > body["numTrialsLimit"]:
        reason_codes.append("TRIAL_LIMIT_EXCEEDED")

    # Deduplicate by [entity, UTC(eventTime)].
    groups = {}

    for row in rows:
        key = (
            row["entity"],
            utc_timestamp_key(row["eventTime"]),
        )

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        if row["version"] > current["version"]:
            groups[key] = row
        elif row["version"] == current["version"]:
            if row["id"].encode("utf-8") < current["id"].encode("utf-8"):
                groups[key] = row

    retained = list(groups.values())

    # A feature is eligible only if:
    # 1. It occurs in every retained row.
    # 2. It isn't forbidden.
    # 3. availableAt <= predictionTime in every retained row.
    common_features = None

    for row in retained:
        names = set(row["features"].keys())

        if common_features is None:
            common_features = names
        else:
            common_features &= names

    if common_features is None:
        common_features = set()

    eligible_features = []

    for fname in common_features:
        if fname in forbidden:
            continue

        eligible = True

        for row in retained:
            available_at = row["features"][fname]["availableAt"]
            prediction_time = row["predictionTime"]

            if utc_timestamp_key(available_at) > utc_timestamp_key(
                prediction_time
            ):
                eligible = False
                break

        if eligible:
            eligible_features.append(fname)

    feature_names = utf8_sorted(eligible_features)

    train_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ]
    )

    eval_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ]
    )

    # Only finite SUCCEEDED trials are eligible.
    successful = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and isinstance(trial["evalMetric"], (int, float))
            and not isinstance(trial["evalMetric"], bool)
            and math.isfinite(float(trial["evalMetric"]))
        )
    ]

    if not successful:
        reason_codes.append("NO_SUCCESSFUL_TRIAL")

    if reason_codes:
        return {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": utf8_sorted(set(reason_codes)),
        }

    # Max metric, exact tie -> smallest integer trialId.
    selected = max(
        successful,
        key=lambda t: (
            float(t["evalMetric"]),
            -t["trialId"],
        ),
    )

    digest = digest_dataset(
        train_ids,
        eval_ids,
        feature_names,
    )

    return {
        "runId": run_id,
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": [],
    }


def valid_evaluate_shape(body):
    required = [
        "phase",
        "runId",
        "selectedTrialId",
        "datasetDigest",
        "metricFloor",
        "requiredSlices",
        "rows",
        "bytesProcessed",
        "maxBytes",
    ]

    if not all(k in body for k in required):
        return False

    if body.get("phase") != "evaluate":
        return False

    if not isinstance(body["runId"], str) or not body["runId"]:
        return False

    if not is_safe_int(body["selectedTrialId"]):
        return False

    if (
        not isinstance(body["datasetDigest"], str)
        or not DIGEST_RE.fullmatch(body["datasetDigest"])
    ):
        return False

    if not is_finite_number(body["metricFloor"]):
        return False

    if not 0 <= float(body["metricFloor"]) <= 1:
        return False

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str) or not name:
            return False

        if not is_finite_number(floor):
            return False

        if not 0 <= float(floor) <= 1:
            return False

    if not isinstance(body["rows"], list):
        return False

    if not is_safe_int(body["bytesProcessed"]):
        return False

    if not is_safe_int(body["maxBytes"]):
        return False

    return True


def evaluate_run(body):
    run_id = body["runId"]

    reason_codes = []

    # First establish lineage.
    stored = RUNS.get(run_id)

    lineage_ok = (
        stored is not None
        and stored.get("success") is True
        and body["selectedTrialId"] == stored["response"]["selectedTrialId"]
        and body["datasetDigest"] == stored["response"]["datasetDigest"]
    )

    if not lineage_ok:
        reason_codes.append("INVALID_LINEAGE")

    rows = body["rows"]

    # Validate test rows.
    valid_rows = True

    for row in rows:
        if not isinstance(row, dict):
            valid_rows = False
            break

        if set(row.keys()) != {"label", "prediction", "slice"}:
            valid_rows = False
            break

        label = row["label"]
        prediction = row["prediction"]
        slice_name = row["slice"]

        if label not in (0, 1) or prediction not in (0, 1):
            valid_rows = False
            break

        if not isinstance(label, int) or isinstance(label, bool):
            valid_rows = False
            break

        if not isinstance(prediction, int) or isinstance(prediction, bool):
            valid_rows = False
            break

        if not isinstance(slice_name, str) or not slice_name:
            valid_rows = False
            break

    if not valid_rows:
        reason_codes.append("INVALID_TEST_ROW")

    # Bytes are always checked independently.
    if body["bytesProcessed"] > body["maxBytes"]:
        reason_codes.append("BYTE_LIMIT")

    test_metric = None
    critical_slice_pass = False

    # Empty or invalid rows:
    # no aggregate/slice checks.
    if rows and valid_rows:
        total_correct = sum(
            1
            for row in rows
            if row["label"] == row["prediction"]
        )

        test_metric = round(
            total_correct / len(rows),
            12,
        )

        if test_metric < float(body["metricFloor"]):
            reason_codes.append("AGGREGATE_FLOOR")

        required_slices = body["requiredSlices"]

        all_slices_pass = True

        for slice_name in utf8_sorted(required_slices.keys()):
            slice_rows = [
                row
                for row in rows
                if row["slice"] == slice_name
            ]

            if not slice_rows:
                reason_codes.append(
                    f"MISSING_SLICE:{slice_name}"
                )
                all_slices_pass = False
                continue

            correct = sum(
                1
                for row in slice_rows
                if row["label"] == row["prediction"]
            )

            slice_metric = round(
                correct / len(slice_rows),
                12,
            )

            if slice_metric < float(
                required_slices[slice_name]
            ):
                reason_codes.append(
                    f"SLICE_FLOOR:{slice_name}"
                )
                all_slices_pass = False

        critical_slice_pass = all_slices_pass

    # Contract says criticalSlicePass is false for:
    # invalid input, invalid lineage, invalid row,
    # missing slice, failed slice floor.
    if not lineage_ok or not rows or not valid_rows:
        critical_slice_pass = False

    # Also false if any slice-related failure occurred.
    if any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in reason_codes
    ):
        critical_slice_pass = False

    # Admission requires EVERYTHING to pass.
    decision = (
        "admit"
        if (
            lineage_ok
            and rows
            and valid_rows
            and test_metric is not None
            and test_metric >= float(body["metricFloor"])
            and critical_slice_pass
            and body["bytesProcessed"] <= body["maxBytes"]
        )
        else "reject"
    )

    reason_codes = utf8_sorted(set(reason_codes))

    return {
        "runId": run_id,
        "selectedTrialId": body["selectedTrialId"],
        "datasetDigest": body["datasetDigest"],
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": body["bytesProcessed"],
        "reasonCodes": reason_codes,
    }


@app.post("/bqml")
async def bqml(request: Request):
    # JSON parsing.
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

    # Unknown/missing phase -> exactly this response.
    if phase not in ("select", "evaluate"):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # ---------------------------------------------------------
    # SELECT
    # ---------------------------------------------------------
    if phase == "select":
        if not selection_input_valid(body):
            return JSONResponse(
                {"error": "INVALID_INPUT"},
                status_code=400,
            )

        run_id = body["runId"]
        fingerprint = response_fingerprint(body)

        # Identical replay.
        if run_id in RUNS:
            stored = RUNS[run_id]

            if stored["fingerprint"] == fingerprint:
                return JSONResponse(stored["response"])

            # Reusing runId with different selection input.
            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = perform_selection(body)

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
            "success": (
                response["selectedTrialId"] is not None
                and response["datasetDigest"] is not None
                and not response["reasonCodes"]
            ),
        }

        return JSONResponse(response)

    # ---------------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------------
    if phase == "evaluate":
        if not valid_evaluate_shape(body):
            return JSONResponse(
                {
                    "runId": body.get("runId"),
                    "selectedTrialId": body.get("selectedTrialId"),
                    "datasetDigest": body.get("datasetDigest"),
                    "testMetric": None,
                    "criticalSlicePass": False,
                    "decision": "reject",
                    "bytesProcessed": body.get("bytesProcessed"),
                    "reasonCodes": ["INVALID_INPUT"],
                },
                status_code=400,
            )

        return JSONResponse(evaluate_run(body))


@app.get("/")
def root():
    return {"status": "ok"}