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

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# =========================================================
# HELPERS
# =========================================================

def safe_int(v: Any) -> bool:
    return (
        isinstance(v, int)
        and not isinstance(v, bool)
        and 0 <= v <= SAFE_INT_MAX
    )


def finite_number(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def valid_timestamp(v: Any) -> bool:
    if not isinstance(v, str):
        return False

    if TIMESTAMP_RE.fullmatch(v) is None:
        return False

    try:
        if v.endswith("Z"):
            datetime.fromisoformat(v[:-1] + "+00:00")
        else:
            datetime.fromisoformat(v)
        return True
    except (ValueError, TypeError):
        return False


def parse_utc(v: str) -> datetime:
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"

    return datetime.fromisoformat(v).astimezone(timezone.utc)


def utf8_key(v: str) -> bytes:
    return v.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def has_fields(obj: Any, fields) -> bool:
    return (
        isinstance(obj, dict)
        and all(x in obj for x in fields)
    )


def unique_strings(values) -> bool:
    return len(values) == len(set(values))


# =========================================================
# DATASET DIGEST
# =========================================================

def make_dataset_digest(
    train_ids,
    eval_ids,
    feature_names,
):
    payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    raw = compact_json(payload).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# =========================================================
# SELECTION VALIDATION
# =========================================================

def validate_selection(body: Any) -> bool:

    if not isinstance(body, dict):
        return False

    required = [
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials",
    ]

    if not has_fields(body, required):
        return False

    if body["phase"] != "select":
        return False

    # runId
    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return False

    # forbiddenFeatures
    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return False

    if not all(isinstance(x, str) for x in forbidden):
        return False

    if not unique_strings(forbidden):
        return False

    # trial limit
    limit = body["numTrialsLimit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > SAFE_INT_MAX
    ):
        return False

    # rows
    rows = body["rows"]

    if not isinstance(rows, list) or len(rows) == 0:
        return False

    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            return False

        required_row = [
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        ]

        if not has_fields(row, required_row):
            return False

        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or len(row_id) == 0
        ):
            return False

        if row_id in row_ids:
            return False

        row_ids.add(row_id)

        if not isinstance(row["entity"], str):
            return False

        if not valid_timestamp(row["eventTime"]):
            return False

        if not valid_timestamp(row["predictionTime"]):
            return False

        if not safe_int(row["version"]):
            return False

        if row["split"] not in ("TRAIN", "EVAL"):
            return False

        features = row["features"]

        if not isinstance(features, dict):
            return False

        for feature_name, feature in features.items():

            if not isinstance(feature_name, str):
                return False

            if not isinstance(feature, dict):
                return False

            if not has_fields(
                feature,
                ["value", "availableAt"],
            ):
                return False

            if not valid_timestamp(
                feature["availableAt"]
            ):
                return False

    # trials
    trials = body["trials"]

    if not isinstance(trials, list):
        return False

    trial_ids = set()

    for trial in trials:

        if not isinstance(trial, dict):
            return False

        if not has_fields(
            trial,
            ["trialId", "status"],
        ):
            return False

        trial_id = trial["trialId"]

        if not safe_int(trial_id):
            return False

        if trial_id in trial_ids:
            return False

        trial_ids.add(trial_id)

        status = trial["status"]

        if status not in (
            "SUCCEEDED",
            "FAILED",
        ):
            return False

        if status == "SUCCEEDED":

            if "evalMetric" not in trial:
                return False

            if not finite_number(
                trial["evalMetric"]
            ):
                return False

    return True


# =========================================================
# ROW DEDUPLICATION
# =========================================================

def deduplicate_rows(rows):

    retained = {}

    for row in rows:

        key = (
            row["entity"],
            parse_utc(row["eventTime"]),
        )

        current = retained.get(key)

        if current is None:
            retained[key] = row
            continue

        # Highest version wins.
        if row["version"] > current["version"]:
            retained[key] = row
            continue

        # Same version -> UTF-8 smallest ID.
        if (
            row["version"] == current["version"]
            and utf8_key(row["id"])
            < utf8_key(current["id"])
        ):
            retained[key] = row

    return list(retained.values())


# =========================================================
# POINT-IN-TIME FEATURE ELIGIBILITY
# =========================================================

def get_eligible_features(
    retained_rows,
    forbidden,
):

    if not retained_rows:
        return []

    common = set(
        retained_rows[0]["features"].keys()
    )

    for row in retained_rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    eligible = []

    for name in common:

        if name in forbidden:
            continue

        ok = True

        for row in retained_rows:

            available_at = parse_utc(
                row["features"][name]["availableAt"]
            )

            prediction_time = parse_utc(
                row["predictionTime"]
            )

            # Point-in-time safety:
            # feature must exist no later than
            # the prediction timestamp.
            if available_at > prediction_time:
                ok = False
                break

        if ok:
            eligible.append(name)

    return utf8_sorted(eligible)


# =========================================================
# TRIAL SELECTION
# =========================================================

def choose_trial(trials):

    candidates = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get("evalMetric")

        if not finite_number(metric):
            continue

        candidates.append(trial)

    if not candidates:
        return None

    # Maximize metric.
    # Exact metric tie -> smallest integer trialId.
    candidates.sort(
        key=lambda t: (
            -float(t["evalMetric"]),
            t["trialId"],
        )
    )

    return candidates[0]


# =========================================================
# SELECTION
# =========================================================

def perform_selection(body):

    reason_codes = []

    if len(body["trials"]) > body["numTrialsLimit"]:
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    retained = deduplicate_rows(
        body["rows"]
    )

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

    feature_names = get_eligible_features(
        retained,
        set(body["forbiddenFeatures"]),
    )

    selected = choose_trial(
        body["trials"]
    )

    if selected is None:
        reason_codes.append(
            "NO_SUCCESSFUL_TRIAL"
        )

    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # Any selection error means no usable lineage.
    if reason_codes:

        return {
            "runId": body["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": reason_codes,
        }

    digest = make_dataset_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    return {
        "runId": body["runId"],
        "selectedTrialId": selected["trialId"],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": [],
    }


# =========================================================
# EVALUATION INPUT VALIDATION
# =========================================================

def validate_evaluation(body: Any) -> bool:

    if not isinstance(body, dict):
        return False

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

    if not has_fields(body, required):
        return False

    if body["phase"] != "evaluate":
        return False

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return False

    # Selected trial MUST be non-null safe integer.
    if not safe_int(body["selectedTrialId"]):
        return False

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return False

    floor = body["metricFloor"]

    if not finite_number(floor):
        return False

    if not 0 <= float(floor) <= 1:
        return False

    required_slices = body["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, value in required_slices.items():

        if (
            not isinstance(name, str)
            or len(name) == 0
        ):
            return False

        if not finite_number(value):
            return False

        if not 0 <= float(value) <= 1:
            return False

    rows = body["rows"]

    if not isinstance(rows, list):
        return False

    if not safe_int(body["bytesProcessed"]):
        return False

    if not safe_int(body["maxBytes"]):
        return False

    return True


# =========================================================
# TEST ROW VALIDATION
# =========================================================

def validate_test_rows(rows):

    if not isinstance(rows, list):
        return False

    for row in rows:

        if not isinstance(row, dict):
            return False

        if not has_fields(
            row,
            [
                "label",
                "prediction",
                "slice",
            ],
        ):
            return False

        label = row["label"]
        prediction = row["prediction"]
        slice_name = row["slice"]

        # bool is an int subclass in Python,
        # so explicitly reject bool.
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            return False

        if (
            not isinstance(prediction, int)
            or isinstance(prediction, bool)
            or prediction not in (0, 1)
        ):
            return False

        if (
            not isinstance(slice_name, str)
            or len(slice_name) == 0
        ):
            return False

    return True


# =========================================================
# EVALUATION
# =========================================================

def perform_evaluation(body):

    run_id = body["runId"]
    selected_trial_id = body["selectedTrialId"]
    digest = body["datasetDigest"]

    reason_codes = []

    # -----------------------------------------------------
    # FROZEN LINEAGE
    # -----------------------------------------------------

    stored = RUNS.get(run_id)

    lineage_valid = False

    if isinstance(stored, dict):

        saved = stored.get("response")

        if (
            stored.get("successful") is True
            and isinstance(saved, dict)
            and saved.get("runId") == run_id
            and saved.get("selectedTrialId")
            == selected_trial_id
            and saved.get("datasetDigest")
            == digest
            and isinstance(
                saved.get("datasetDigest"),
                str,
            )
            and DIGEST_RE.fullmatch(
                saved["datasetDigest"]
            ) is not None
            and saved.get("reasonCodes") == []
        ):
            lineage_valid = True

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    # -----------------------------------------------------
    # BYTE CHECK
    # -----------------------------------------------------

    bytes_processed = body["bytesProcessed"]
    max_bytes = body["maxBytes"]

    byte_pass = (
        bytes_processed <= max_bytes
    )

    if not byte_pass:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    # -----------------------------------------------------
    # TEST ROW CHECK
    # -----------------------------------------------------

    rows = body["rows"]

    rows_valid = validate_test_rows(rows)

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # -----------------------------------------------------
    # Empty OR invalid rows
    # -----------------------------------------------------

    if len(rows) == 0 or not rows_valid:

        reason_codes = utf8_sorted(
            set(reason_codes)
        )

        return {
            "runId": run_id,
            "selectedTrialId": selected_trial_id,
            "datasetDigest": digest,
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed": bytes_processed,
            "reasonCodes": reason_codes,
        }

    # -----------------------------------------------------
    # AGGREGATE ACCURACY
    # -----------------------------------------------------

    correct = 0

    for row in rows:
        if row["label"] == row["prediction"]:
            correct += 1

    test_metric = round(
        correct / len(rows),
        12,
    )

    aggregate_pass = (
        test_metric
        >= float(body["metricFloor"])
    )

    if not aggregate_pass:
        reason_codes.append(
            "AGGREGATE_FLOOR"
        )

    # -----------------------------------------------------
    # REQUIRED SLICE CHECKS
    # -----------------------------------------------------

    required_slices = body["requiredSlices"]

    all_required_slices_pass = True

    # Keep slice names deterministic.
    for slice_name in utf8_sorted(
        required_slices.keys()
    ):

        slice_rows = [
            row
            for row in rows
            if row["slice"] == slice_name
        ]

        # Required slice missing.
        if len(slice_rows) == 0:

            reason_codes.append(
                "MISSING_SLICE:" + slice_name
            )

            all_required_slices_pass = False
            continue

        slice_correct = sum(
            1
            for row in slice_rows
            if row["label"] == row["prediction"]
        )

        slice_metric = round(
            slice_correct / len(slice_rows),
            12,
        )

        floor = float(
            required_slices[slice_name]
        )

        # Inclusive floor.
        if slice_metric < floor:

            reason_codes.append(
                "SLICE_FLOOR:" + slice_name
            )

            all_required_slices_pass = False

    # -----------------------------------------------------
    # criticalSlicePass
    #
    # Contract:
    # false for:
    # - invalid input
    # - invalid lineage
    # - invalid test row
    # - missing required slice
    # - failed slice floor
    #
    # It does NOT represent aggregate or byte status.
    # -----------------------------------------------------

    critical_slice_pass = (
        lineage_valid
        and rows_valid
        and len(rows) > 0
        and all_required_slices_pass
    )

    # -----------------------------------------------------
    # FINAL DECISION
    #
    # Admission requires ALL gates:
    # - valid frozen lineage
    # - valid non-empty test data
    # - aggregate floor
    # - every required slice floor
    # - every required slice exists
    # - byte limit
    # -----------------------------------------------------

    admit = (
        lineage_valid
        and rows_valid
        and len(rows) > 0
        and aggregate_pass
        and all_required_slices_pass
        and byte_pass
    )

    decision = "admit" if admit else "reject"

    # -----------------------------------------------------
    # DEDUPLICATE + UTF-8 SORT REASON CODES
    # -----------------------------------------------------

    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # -----------------------------------------------------
    # EXACT OUTPUT
    # -----------------------------------------------------

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    }


# =========================================================
# /bqml
# =========================================================

@app.post("/bqml")
async def bqml(request: Request):

    # -----------------------------------------------------
    # JSON parsing
    # -----------------------------------------------------

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

    # Unknown/missing phase -> EXACT response.
    if phase not in ("select", "evaluate"):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # =====================================================
    # SELECT
    # =====================================================

    if phase == "select":

        # Invalid selection still returns the
        # selection response shape.
        if not validate_selection(body):

            run_id = body.get("runId")

            if not isinstance(run_id, str):
                run_id = ""

            return JSONResponse(
                {
                    "runId": run_id,
                    "selectedTrialId": None,
                    "trainRowIds": [],
                    "evalRowIds": [],
                    "featureNames": [],
                    "datasetDigest": None,
                    "reasonCodes": [
                        "INVALID_INPUT"
                    ],
                }
            )

        run_id = body["runId"]

        # Complete canonical input fingerprint.
        fingerprint = hashlib.sha256(
            compact_json(body).encode("utf-8")
        ).hexdigest()

        # Existing run.
        if run_id in RUNS:

            existing = RUNS[run_id]

            if (
                existing.get("fingerprint")
                == fingerprint
            ):
                # Exact replay:
                # return frozen response unchanged.
                return JSONResponse(
                    existing["response"]
                )

            # Same run ID but different selection.
            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        response = perform_selection(body)

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

    # =====================================================
    # EVALUATE
    # =====================================================

    if phase == "evaluate":

        if not validate_evaluation(body):

            return JSONResponse(
                {
                    "runId": body.get("runId"),
                    "selectedTrialId":
                        body.get("selectedTrialId"),
                    "datasetDigest":
                        body.get("datasetDigest"),
                    "testMetric": None,
                    "criticalSlicePass": False,
                    "decision": "reject",
                    "bytesProcessed":
                        body.get("bytesProcessed"),
                    "reasonCodes": [
                        "INVALID_INPUT"
                    ],
                }
            )

        return JSONResponse(
            perform_evaluation(body)
        )

    return JSONResponse(
        {"error": "INVALID_INPUT"},
        status_code=400,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {"status": "ok"}