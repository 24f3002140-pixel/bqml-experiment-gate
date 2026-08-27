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

# runId -> {
#     fingerprint: str,
#     response: dict,
#     successful: bool
# }
RUNS: dict[str, dict[str, Any]] = {}

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


# =========================================================
# BASIC HELPERS
# =========================================================

def safe_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    if not TIMESTAMP_RE.fullmatch(value):
        return False

    try:
        if value.endswith("Z"):
            datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            datetime.fromisoformat(value)

        return True
    except (ValueError, TypeError, OverflowError):
        return False


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value).astimezone(timezone.utc)


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def has_fields(obj: Any, fields) -> bool:
    return (
        isinstance(obj, dict)
        and all(field in obj for field in fields)
    )


def unique_utf8_strings(values) -> bool:
    if not isinstance(values, list):
        return False

    seen = set()

    for value in values:
        if not isinstance(value, str):
            return False

        if value in seen:
            return False

        seen.add(value)

    return True


# =========================================================
# DATASET DIGEST
# =========================================================

def make_dataset_digest(
    train_ids,
    eval_ids,
    feature_names,
):
    # IMPORTANT:
    # Exact key order required by contract.
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

    # -----------------------------------------------------
    # runId
    # -----------------------------------------------------

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return False

    # -----------------------------------------------------
    # forbiddenFeatures
    # -----------------------------------------------------

    forbidden = body["forbiddenFeatures"]

    if not unique_utf8_strings(forbidden):
        return False

    # -----------------------------------------------------
    # numTrialsLimit
    # -----------------------------------------------------

    limit = body["numTrialsLimit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > SAFE_INT_MAX
    ):
        return False

    # -----------------------------------------------------
    # rows
    # -----------------------------------------------------

    rows = body["rows"]

    if not isinstance(rows, list) or len(rows) == 0:
        return False

    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            return False

        required_row_fields = [
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        ]

        if not has_fields(row, required_row_fields):
            return False

        # ID
        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or len(row_id) == 0
        ):
            return False

        if row_id in row_ids:
            return False

        row_ids.add(row_id)

        # Entity
        if not isinstance(row["entity"], str):
            return False

        # Timestamps
        if not valid_timestamp(row["eventTime"]):
            return False

        if not valid_timestamp(row["predictionTime"]):
            return False

        # Version
        if not safe_int(row["version"]):
            return False

        # Split
        if row["split"] not in ("TRAIN", "EVAL"):
            return False

        # Features
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

            # "value" is arbitrary data.
            # NEVER parse or interpret it.

            if not valid_timestamp(
                feature["availableAt"]
            ):
                return False

    # -----------------------------------------------------
    # trials
    # -----------------------------------------------------

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

        if status not in ("SUCCEEDED", "FAILED"):
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
# DEDUPLICATION
# =========================================================

def deduplicate_rows(rows):

    groups = {}

    for row in rows:

        key = (
            row["entity"],
            parse_utc(row["eventTime"]),
        )

        old = groups.get(key)

        if old is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > old["version"]:
            groups[key] = row
            continue

        # Equal version -> UTF-8 smallest ID.
        if (
            row["version"] == old["version"]
            and utf8_key(row["id"])
            < utf8_key(old["id"])
        ):
            groups[key] = row

    return list(groups.values())


# =========================================================
# POINT-IN-TIME FEATURE ELIGIBILITY
# =========================================================

def get_eligible_features(
    rows,
    forbidden,
):

    if not rows:
        return []

    # Feature must exist in EVERY retained row.
    common = set(rows[0]["features"].keys())

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    eligible = []

    for feature_name in common:

        if feature_name in forbidden:
            continue

        feature_ok = True

        for row in rows:

            available_at = parse_utc(
                row["features"]
                [feature_name]
                ["availableAt"]
            )

            prediction_time = parse_utc(
                row["predictionTime"]
            )

            # Point-in-time safety:
            # feature cannot become available AFTER prediction time.
            if available_at > prediction_time:
                feature_ok = False
                break

        if feature_ok:
            eligible.append(feature_name)

    return utf8_sorted(eligible)


# =========================================================
# TRIAL SELECTION
# =========================================================

def choose_trial(trials):

    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get("evalMetric")

        if not finite_number(metric):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric first.
    # Exact metric tie -> smallest integer trialId.
    eligible.sort(
        key=lambda trial: (
            -float(trial["evalMetric"]),
            trial["trialId"],
        )
    )

    return eligible[0]


# =========================================================
# SELECTION
# =========================================================

def perform_selection(body):

    reason_codes = []

    # Contract failure if more trials than allowed.
    if len(body["trials"]) > body["numTrialsLimit"]:
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    retained = deduplicate_rows(
        body["rows"]
    )

    # Final IDs are sorted by UTF-8 bytes.
    train_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "TRAIN"
    ])

    eval_ids = utf8_sorted([
        row["id"]
        for row in retained
        if row["split"] == "EVAL"
    ])

    # IMPORTANT:
    # Only TRAIN/EVAL rows are used for selection.
    # There is no TEST split accepted here.
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

    # ANY selection error:
    # selectedTrialId MUST be null
    # datasetDigest MUST be null.
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
# EVALUATION VALIDATION
# =========================================================

def validate_evaluation(body):

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

    # -----------------------------------------------------
    # runId
    # -----------------------------------------------------

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        return False

    # -----------------------------------------------------
    # Frozen trial ID
    # -----------------------------------------------------

    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    # -----------------------------------------------------
    # Digest
    # -----------------------------------------------------

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return False

    # -----------------------------------------------------
    # Aggregate floor
    # -----------------------------------------------------

    metric_floor = body["metricFloor"]

    if not finite_number(metric_floor):
        return False

    if not (
        0 <= float(metric_floor) <= 1
    ):
        return False

    # -----------------------------------------------------
    # Required slices
    # -----------------------------------------------------

    required_slices = body["requiredSlices"]

    if not isinstance(
        required_slices,
        dict,
    ):
        return False

    for name, floor in required_slices.items():

        if (
            not isinstance(name, str)
            or len(name) == 0
        ):
            return False

        if not finite_number(floor):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    # -----------------------------------------------------
    # Test rows
    # -----------------------------------------------------

    if not isinstance(
        body["rows"],
        list,
    ):
        return False

    # -----------------------------------------------------
    # Cost
    # -----------------------------------------------------

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

        # Binary INTEGER only.
        # bool is deliberately rejected.
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

        # Slice must be non-empty string.
        if (
            not isinstance(slice_name, str)
            or len(slice_name) == 0
        ):
            return False

    return True


# =========================================================
# FROZEN LINEAGE
# =========================================================

def check_lineage(
    run_id,
    selected_trial_id,
    digest,
):

    stored = RUNS.get(run_id)

    if stored is None:
        return False

    if stored.get("successful") is not True:
        return False

    saved = stored.get("response")

    if not isinstance(saved, dict):
        return False

    if saved.get("runId") != run_id:
        return False

    if saved.get(
        "selectedTrialId"
    ) != selected_trial_id:
        return False

    saved_digest = saved.get(
        "datasetDigest"
    )

    if saved_digest != digest:
        return False

    if (
        not isinstance(saved_digest, str)
        or DIGEST_RE.fullmatch(saved_digest)
        is None
    ):
        return False

    # A successful selection must have no reason codes.
    if saved.get("reasonCodes") != []:
        return False

    return True


# =========================================================
# EVALUATION
# =========================================================

def perform_evaluation(body):

    run_id = body["runId"]
    selected_trial_id = body[
        "selectedTrialId"
    ]
    digest = body[
        "datasetDigest"
    ]

    rows = body["rows"]

    bytes_processed = body[
        "bytesProcessed"
    ]

    max_bytes = body[
        "maxBytes"
    ]

    reason_codes = []

    # -----------------------------------------------------
    # 1. FROZEN LINEAGE
    # -----------------------------------------------------

    lineage_valid = check_lineage(
        run_id,
        selected_trial_id,
        digest,
    )

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    # -----------------------------------------------------
    # 2. COST
    # -----------------------------------------------------

    byte_pass = (
        bytes_processed <= max_bytes
    )

    if not byte_pass:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    # -----------------------------------------------------
    # 3. TEST ROW VALIDITY
    # -----------------------------------------------------

    rows_valid = validate_test_rows(rows)

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # -----------------------------------------------------
    # IMPORTANT CONTRACT:
    #
    # Empty rows OR ANY invalid row:
    #
    # testMetric = null
    # slice checks skipped
    # aggregate check skipped
    # criticalSlicePass = false
    #
    # Lineage and byte checks still apply.
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
    # 4. AGGREGATE ACCURACY
    # -----------------------------------------------------

    correct = sum(
        1
        for row in rows
        if row["label"] == row["prediction"]
    )

    test_metric = round(
        correct / len(rows),
        12,
    )

    metric_floor = float(
        body["metricFloor"]
    )

    aggregate_pass = (
        test_metric >= metric_floor
    )

    if not aggregate_pass:
        reason_codes.append(
            "AGGREGATE_FLOOR"
        )

    # -----------------------------------------------------
    # 5. REQUIRED SLICE CHECKS
    # -----------------------------------------------------

    required_slices = body[
        "requiredSlices"
    ]

    all_required_slices_pass = True

    # Sort slice names by UTF-8 bytes.
    for slice_name in utf8_sorted(
        required_slices.keys()
    ):

        slice_rows = [
            row
            for row in rows
            if row["slice"] == slice_name
        ]

        # Required slice must exist.
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
    # 6. criticalSlicePass
    # -----------------------------------------------------
    #
    # This flag:
    #
    # - false on invalid lineage
    # - false on invalid test rows
    # - false on empty rows
    # - false on missing required slice
    # - false on failed slice floor
    #
    # It does NOT summarize:
    # - aggregate floor
    # - byte limit
    #
    # -----------------------------------------------------

    critical_slice_pass = (
        lineage_valid
        and all_required_slices_pass
    )

    # -----------------------------------------------------
    # 7. FINAL DECISION
    # -----------------------------------------------------
    #
    # Admission requires ALL gates:
    #
    # lineage
    # + valid non-empty test rows
    # + aggregate floor
    # + required slices
    # + byte limit
    #
    # -----------------------------------------------------

    admit = (
        lineage_valid
        and rows_valid
        and len(rows) > 0
        and aggregate_pass
        and all_required_slices_pass
        and byte_pass
    )

    decision = (
        "admit"
        if admit
        else "reject"
    )

    # -----------------------------------------------------
    # 8. SORT + DEDUP REASON CODES
    # -----------------------------------------------------

    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # -----------------------------------------------------
    # 9. EXACT OUTPUT SHAPE
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
# INVALID EVALUATION RESPONSE
# =========================================================

def invalid_evaluation_response(body):

    run_id = (
        body.get("runId")
        if isinstance(body, dict)
        else None
    )

    selected_trial_id = (
        body.get("selectedTrialId")
        if isinstance(body, dict)
        else None
    )

    digest = (
        body.get("datasetDigest")
        if isinstance(body, dict)
        else None
    )

    bytes_processed = (
        body.get("bytesProcessed")
        if isinstance(body, dict)
        else None
    )

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": digest,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": [
            "INVALID_INPUT"
        ],
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

    # -----------------------------------------------------
    # Phase
    # -----------------------------------------------------

    phase = body.get("phase")

    if phase not in (
        "select",
        "evaluate",
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # =====================================================
    # SELECT
    # =====================================================

    if phase == "select":

        # Malformed selection has the selection output shape.
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

        # -------------------------------------------------
        # Fingerprint COMPLETE selection input.
        # -------------------------------------------------

        fingerprint = hashlib.sha256(
            compact_json(body).encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # Stateful replay / conflict
        # -------------------------------------------------

        if run_id in RUNS:

            existing = RUNS[run_id]

            if existing["fingerprint"] == fingerprint:

                # IDENTICAL replay:
                # return frozen response unchanged.
                return JSONResponse(
                    existing["response"]
                )

            # Same runId with different selection input.
            return JSONResponse(
                {"error": "RUN_ID_CONFLICT"},
                status_code=409,
            )

        # -------------------------------------------------
        # Execute and freeze selection.
        # -------------------------------------------------

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
                invalid_evaluation_response(body)
            )

        response = perform_evaluation(body)

        return JSONResponse(response)

    # Defensive fallback.
    return JSONResponse(
        {"error": "INVALID_INPUT"},
        status_code=400,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status": "ok"
    }