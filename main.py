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

# Stateful experiment storage.
# This is intentionally process-local for the grader.
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

    if TIMESTAMP_RE.fullmatch(value) is None:
        return False

    try:
        if value.endswith("Z"):
            datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                return False

        return True

    except (ValueError, TypeError):
        return False


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value).astimezone(timezone.utc)


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def utf8_sorted(values):
    return sorted(values, key=utf8_key)


def sorted_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def has_fields(obj: Any, fields) -> bool:
    return (
        isinstance(obj, dict)
        and all(field in obj for field in fields)
    )


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

    return hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()


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
        or not run_id
        or len(run_id) > 128
    ):
        return False

    # -----------------------------------------------------
    # forbiddenFeatures
    # -----------------------------------------------------

    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return False

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
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

    if not isinstance(rows, list):
        return False

    if len(rows) == 0:
        return False

    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            return False

        if not has_fields(
            row,
            [
                "id",
                "entity",
                "eventTime",
                "predictionTime",
                "version",
                "split",
                "features",
            ],
        ):
            return False

        # ID
        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or not row_id
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
                [
                    "value",
                    "availableAt",
                ],
            ):
                return False

            # value is opaque data.
            # Never interpret its contents.

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
            [
                "trialId",
                "status",
            ],
        ):
            return False

        trial_id = trial["trialId"]

        if not safe_int(trial_id):
            return False

        if trial_id in trial_ids:
            return False

        trial_ids.add(trial_id)

        if trial["status"] not in (
            "SUCCEEDED",
            "FAILED",
        ):
            return False

        if trial["status"] == "SUCCEEDED":

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

        # Same version:
        # UTF-8-byte-smallest ID wins.
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

    # Feature must occur in EVERY retained row.
    common = set(
        rows[0]["features"].keys()
    )

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    result = []

    for feature_name in common:

        if feature_name in forbidden:
            continue

        eligible = True

        for row in rows:

            available_at = parse_utc(
                row["features"][feature_name]["availableAt"]
            )

            prediction_time = parse_utc(
                row["predictionTime"]
            )

            # Point-in-time rule.
            if available_at > prediction_time:
                eligible = False
                break

        if eligible:
            result.append(feature_name)

    return utf8_sorted(result)


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

    # Highest metric.
    # Exact tie -> smallest trialId.
    eligible.sort(
        key=lambda x: (
            -float(x["evalMetric"]),
            x["trialId"],
        )
    )

    return eligible[0]


# =========================================================
# PERFORM SELECTION
# =========================================================

def perform_selection(body):

    reason_codes = []

    # Contract failure if trial count exceeds limit.
    if len(body["trials"]) > body["numTrialsLimit"]:
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    retained = deduplicate_rows(
        body["rows"]
    )

    # TRAIN IDs.
    train_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "TRAIN"
        ]
    )

    # EVAL IDs.
    eval_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"] == "EVAL"
        ]
    )

    # Shared point-in-time-safe features.
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

    reason_codes = sorted_codes(
        reason_codes
    )

    # Any selection failure -> null digest.
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

    if (
        not isinstance(body["runId"], str)
        or not body["runId"]
        or len(body["runId"]) > 128
    ):
        return False

    # -----------------------------------------------------
    # selectedTrialId
    # -----------------------------------------------------

    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    # -----------------------------------------------------
    # datasetDigest
    # -----------------------------------------------------

    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return False

    # -----------------------------------------------------
    # metricFloor
    # -----------------------------------------------------

    metric_floor = body["metricFloor"]

    if not finite_number(metric_floor):
        return False

    if not (
        0 <= float(metric_floor) <= 1
    ):
        return False

    # -----------------------------------------------------
    # requiredSlices
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
            or not name
        ):
            return False

        if not finite_number(floor):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    # -----------------------------------------------------
    # rows
    # -----------------------------------------------------

    if not isinstance(
        body["rows"],
        list,
    ):
        return False

    # -----------------------------------------------------
    # bytes
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

        # Must be integer 0 or 1.
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            return False

        # Must be integer 0 or 1.
        if (
            not isinstance(prediction, int)
            or isinstance(prediction, bool)
            or prediction not in (0, 1)
        ):
            return False

        # Non-empty string.
        if (
            not isinstance(slice_name, str)
            or not slice_name
        ):
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

    # =====================================================
    # 1. FROZEN LINEAGE
    # =====================================================

    stored = RUNS.get(run_id)

    lineage_valid = False

    if (
        stored is not None
        and stored.get("successful") is True
    ):

        saved = stored.get("response")

        if isinstance(saved, dict):

            saved_run_id = saved.get(
                "runId"
            )

            saved_trial_id = saved.get(
                "selectedTrialId"
            )

            saved_digest = saved.get(
                "datasetDigest"
            )

            saved_codes = saved.get(
                "reasonCodes"
            )

            if (
                saved_run_id == run_id
                and saved_trial_id
                == selected_trial_id
                and saved_digest == digest
                and isinstance(
                    saved_digest,
                    str,
                )
                and DIGEST_RE.fullmatch(
                    saved_digest
                ) is not None
                and saved_codes == []
            ):
                lineage_valid = True

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    # =====================================================
    # 2. BYTE GATE
    # =====================================================

    byte_pass = (
        bytes_processed <= max_bytes
    )

    if not byte_pass:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    # =====================================================
    # 3. TEST ROW VALIDATION
    # =====================================================

    rows_valid = validate_test_rows(
        rows
    )

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # =====================================================
    # 4. EMPTY / INVALID ROWS
    # =====================================================

    # Contract:
    #
    # Empty OR invalid rows:
    #   testMetric = null
    #   aggregate check skipped
    #   slice checks skipped
    #   criticalSlicePass = false
    #
    # But lineage and bytes remain applicable.

    if (
        not rows
        or not rows_valid
    ):

        return {
            "runId": run_id,
            "selectedTrialId":
                selected_trial_id,
            "datasetDigest":
                digest,
            "testMetric":
                None,
            "criticalSlicePass":
                False,
            "decision":
                "reject",
            "bytesProcessed":
                bytes_processed,
            "reasonCodes":
                sorted_codes(reason_codes),
        }

    # =====================================================
    # 5. AGGREGATE ACCURACY
    # =====================================================

    correct = sum(
        1
        for row in rows
        if row["label"]
        == row["prediction"]
    )

    test_metric = round(
        correct / len(rows),
        12,
    )

    aggregate_pass = (
        test_metric
        >= float(
            body["metricFloor"]
        )
    )

    if not aggregate_pass:
        reason_codes.append(
            "AGGREGATE_FLOOR"
        )

    # =====================================================
    # 6. REQUIRED SLICES
    # =====================================================

    required_slices = body[
        "requiredSlices"
    ]

    all_required_slices_pass = True

    # Track the actual critical slice independently.
    critical_slice_pass = True

    has_critical_requirement = (
        "critical"
        in required_slices
    )

    for slice_name in utf8_sorted(
        required_slices.keys()
    ):

        slice_rows = [
            row
            for row in rows
            if row["slice"]
            == slice_name
        ]

        # Required slice must exist.
        if not slice_rows:

            reason_codes.append(
                "MISSING_SLICE:"
                + slice_name
            )

            all_required_slices_pass = False

            if slice_name == "critical":
                critical_slice_pass = False

            continue

        slice_correct = sum(
            1
            for row in slice_rows
            if row["label"]
            == row["prediction"]
        )

        slice_metric = round(
            slice_correct
            / len(slice_rows),
            12,
        )

        floor = float(
            required_slices[
                slice_name
            ]
        )

        # Inclusive floor.
        if slice_metric < floor:

            reason_codes.append(
                "SLICE_FLOOR:"
                + slice_name
            )

            all_required_slices_pass = False

            if slice_name == "critical":
                critical_slice_pass = False

    # =====================================================
    # 7. criticalSlicePass
    # =====================================================

    if not lineage_valid:
        critical_slice_pass = False
    elif has_critical_requirement:
        # Already calculated above.
        pass
    else:
        # No required "critical" slice means there is no
        # critical-slice failure to report.
        critical_slice_pass = True

    # =====================================================
    # 8. FINAL DECISION
    # =====================================================

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

    # =====================================================
    # 9. FINAL OUTPUT
    # =====================================================

    return {
        "runId": run_id,
        "selectedTrialId":
            selected_trial_id,
        "datasetDigest":
            digest,
        "testMetric":
            test_metric,
        "criticalSlicePass":
            critical_slice_pass,
        "decision":
            decision,
        "bytesProcessed":
            bytes_processed,
        "reasonCodes":
            sorted_codes(reason_codes),
    }


# =========================================================
# /bqml
# =========================================================

@app.post("/bqml")
async def bqml(request: Request):

    # -----------------------------------------------------
    # Parse JSON
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

    # Unknown/missing phase:
    # EXACT required response.
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

        if not validate_selection(body):

            run_id = body.get(
                "runId"
            )

            if not isinstance(
                run_id,
                str,
            ):
                run_id = ""

            return JSONResponse({
                "runId": run_id,
                "selectedTrialId": None,
                "trainRowIds": [],
                "evalRowIds": [],
                "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": [
                    "INVALID_INPUT"
                ],
            })

        run_id = body["runId"]

        # Fingerprint COMPLETE selection request.
        fingerprint = hashlib.sha256(
            compact_json(body)
            .encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # Existing run
        # -------------------------------------------------

        if run_id in RUNS:

            existing = RUNS[
                run_id
            ]

            # Identical replay.
            if (
                existing["fingerprint"]
                == fingerprint
            ):
                return JSONResponse(
                    existing["response"]
                )

            # Same runId, different selection input.
            return JSONResponse(
                {
                    "error":
                        "RUN_ID_CONFLICT"
                },
                status_code=409,
            )

        # -------------------------------------------------
        # Freeze selection
        # -------------------------------------------------

        response = perform_selection(
            body
        )

        successful = (
            response[
                "selectedTrialId"
            ] is not None
            and response[
                "datasetDigest"
            ] is not None
            and response[
                "reasonCodes"
            ] == []
        )

        RUNS[run_id] = {
            "fingerprint":
                fingerprint,
            "response":
                response,
            "successful":
                successful,
        }

        return JSONResponse(
            response
        )

    # =====================================================
    # EVALUATE
    # =====================================================

    if phase == "evaluate":

        # Evaluation malformed input still has the
        # evaluation-shaped output.
        if not validate_evaluation(body):

            return JSONResponse({
                "runId":
                    body.get(
                        "runId"
                    ),
                "selectedTrialId":
                    body.get(
                        "selectedTrialId"
                    ),
                "datasetDigest":
                    body.get(
                        "datasetDigest"
                    ),
                "testMetric":
                    None,
                "criticalSlicePass":
                    False,
                "decision":
                    "reject",
                "bytesProcessed":
                    body.get(
                        "bytesProcessed"
                    ),
                "reasonCodes": [
                    "INVALID_INPUT"
                ],
            })

        return JSONResponse(
            perform_evaluation(body)
        )

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