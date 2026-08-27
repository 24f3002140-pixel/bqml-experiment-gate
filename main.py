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

# Stateful frozen selections:
# runId -> {
#   fingerprint,
#   response,
#   successful
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

    if not TIMESTAMP_RE.fullmatch(value):
        return False

    try:
        if value.endswith("Z"):
            datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            datetime.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(value).astimezone(
        timezone.utc
    )


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


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
        and all(field in obj for field in fields)
    )


def normalize_codes(codes):
    return sorted(
        set(codes),
        key=utf8_key,
    )


# =========================================================
# DATASET DIGEST
# =========================================================

def make_dataset_digest(
    train_ids,
    eval_ids,
    feature_names,
):
    # Exact required key order.
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

    # runId
    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return False

    # forbiddenFeatures
    forbidden = body["forbiddenFeatures"]

    if not isinstance(forbidden, list):
        return False

    if not all(
        isinstance(x, str)
        for x in forbidden
    ):
        return False

    # numTrialsLimit
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

    if not isinstance(rows, list):
        return False

    if len(rows) == 0:
        return False

    seen_ids = set()

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

        # id
        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in seen_ids
        ):
            return False

        seen_ids.add(row_id)

        # entity
        if not isinstance(
            row["entity"],
            str,
        ):
            return False

        # timestamps
        if not valid_timestamp(
            row["eventTime"]
        ):
            return False

        if not valid_timestamp(
            row["predictionTime"]
        ):
            return False

        # version
        if not safe_int(
            row["version"]
        ):
            return False

        # split
        if row["split"] not in (
            "TRAIN",
            "EVAL",
        ):
            return False

        # features
        features = row["features"]

        if not isinstance(
            features,
            dict,
        ):
            return False

        for feature_name, feature in features.items():

            if not isinstance(
                feature_name,
                str,
            ):
                return False

            if not isinstance(
                feature,
                dict,
            ):
                return False

            if not has_fields(
                feature,
                [
                    "value",
                    "availableAt",
                ],
            ):
                return False

            # Feature value is opaque data.
            if not valid_timestamp(
                feature["availableAt"]
            ):
                return False

    # trials
    trials = body["trials"]

    if not isinstance(
        trials,
        list,
    ):
        return False

    seen_trial_ids = set()

    for trial in trials:

        if not isinstance(
            trial,
            dict,
        ):
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

        if not safe_int(
            trial_id
        ):
            return False

        if trial_id in seen_trial_ids:
            return False

        seen_trial_ids.add(trial_id)

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

        current = groups.get(key)

        if current is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > current["version"]:
            groups[key] = row
            continue

        # Equal version:
        # UTF-8 smallest ID wins.
        if (
            row["version"] == current["version"]
            and utf8_key(row["id"])
            < utf8_key(current["id"])
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

    # Feature must occur in every retained row.
    common = set(
        rows[0]["features"].keys()
    )

    for row in rows[1:]:
        common.intersection_update(
            row["features"].keys()
        )

    eligible = []

    for feature_name in common:

        if feature_name in forbidden:
            continue

        valid = True

        for row in rows:

            available_at = parse_utc(
                row["features"][feature_name][
                    "availableAt"
                ]
            )

            prediction_time = parse_utc(
                row["predictionTime"]
            )

            # Leakage prevention.
            if available_at > prediction_time:
                valid = False
                break

        if valid:
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

        # Only finite successful trials.
        if not finite_number(metric):
            continue

        eligible.append(
            (
                float(metric),
                trial["trialId"],
            )
        )

    if not eligible:
        return None

    # Max metric.
    # Exact tie -> smallest trial ID.
    eligible.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    return eligible[0][1]


# =========================================================
# SELECTION
# =========================================================

def perform_selection(body):

    codes = []

    if len(body["trials"]) > body["numTrialsLimit"]:
        codes.append(
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

    selected_trial_id = choose_trial(
        body["trials"]
    )

    if selected_trial_id is None:
        codes.append(
            "NO_SUCCESSFUL_TRIAL"
        )

    codes = normalize_codes(codes)

    # Any selection failure => null digest.
    if codes:
        return {
            "runId": body["runId"],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": codes,
        }

    digest = make_dataset_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    return {
        "runId": body["runId"],
        "selectedTrialId": selected_trial_id,
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

    # runId
    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
        or len(run_id) > 128
    ):
        return False

    # selectedTrialId
    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    # digest
    digest = body["datasetDigest"]

    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
    ):
        return False

    # metric floor
    metric_floor = body["metricFloor"]

    if not finite_number(
        metric_floor
    ):
        return False

    if not (
        0 <= float(metric_floor) <= 1
    ):
        return False

    # required slices
    required_slices = body[
        "requiredSlices"
    ]

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

    # rows
    if not isinstance(
        body["rows"],
        list,
    ):
        return False

    # bytes
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

def validate_test_row(row):

    if not isinstance(
        row,
        dict,
    ):
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

    # bool is deliberately rejected because bool is
    # technically an int subclass in Python.
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
        or not slice_name
    ):
        return False

    return True


def validate_test_rows(rows):

    for row in rows:
        if not validate_test_row(row):
            return False

    return True


# =========================================================
# FROZEN LINEAGE
# =========================================================

def frozen_lineage_valid(
    run_id,
    selected_trial_id,
    dataset_digest,
):

    stored = RUNS.get(run_id)

    if not isinstance(
        stored,
        dict,
    ):
        return False

    if stored.get("successful") is not True:
        return False

    frozen = stored.get("response")

    if not isinstance(
        frozen,
        dict,
    ):
        return False

    # All three lineage values must exactly match.
    if frozen.get("runId") != run_id:
        return False

    if (
        frozen.get("selectedTrialId")
        != selected_trial_id
    ):
        return False

    if (
        frozen.get("datasetDigest")
        != dataset_digest
    ):
        return False

    # Stored successful selection itself must be clean.
    if frozen.get("reasonCodes") != []:
        return False

    # Stored digest must be a valid lowercase SHA-256.
    stored_digest = frozen.get(
        "datasetDigest"
    )

    if (
        not isinstance(
            stored_digest,
            str,
        )
        or DIGEST_RE.fullmatch(
            stored_digest
        ) is None
    ):
        return False

    return True


# =========================================================
# INVALID EVALUATION RESPONSE
# =========================================================

def invalid_evaluation_response(body):

    return {
        "runId": (
            body.get("runId")
            if isinstance(
                body.get("runId"),
                str,
            )
            else body.get("runId")
        ),
        "selectedTrialId":
            body.get("selectedTrialId"),
        "datasetDigest":
            body.get("datasetDigest"),
        "testMetric":
            None,
        "criticalSlicePass":
            False,
        "decision":
            "reject",
        "bytesProcessed":
            body.get("bytesProcessed"),
        "reasonCodes":
            ["INVALID_INPUT"],
    }


# =========================================================
# EVALUATION
# =========================================================

def perform_evaluation(body):

    run_id = body["runId"]
    selected_trial_id = body[
        "selectedTrialId"
    ]
    dataset_digest = body[
        "datasetDigest"
    ]

    rows = body["rows"]

    bytes_processed = body[
        "bytesProcessed"
    ]

    max_bytes = body[
        "maxBytes"
    ]

    codes = []

    # -----------------------------------------------------
    # LINEAGE
    # -----------------------------------------------------

    lineage_ok = frozen_lineage_valid(
        run_id,
        selected_trial_id,
        dataset_digest,
    )

    if not lineage_ok:
        codes.append(
            "INVALID_LINEAGE"
        )

    # -----------------------------------------------------
    # COST
    # -----------------------------------------------------

    bytes_ok = (
        bytes_processed <= max_bytes
    )

    if not bytes_ok:
        codes.append(
            "BYTE_LIMIT"
        )

    # -----------------------------------------------------
    # TEST ROWS
    # -----------------------------------------------------

    rows_ok = validate_test_rows(
        rows
    )

    if not rows_ok:
        codes.append(
            "INVALID_TEST_ROW"
        )

    # -----------------------------------------------------
    # EMPTY OR INVALID TEST SET
    # -----------------------------------------------------

    # IMPORTANT:
    # aggregate and slice metrics are completely skipped
    # if rows are empty OR any row is invalid.
    #
    # lineage and byte checks remain active.
    #

    if (
        len(rows) == 0
        or not rows_ok
    ):

        return {
            "runId":
                run_id,
            "selectedTrialId":
                selected_trial_id,
            "datasetDigest":
                dataset_digest,
            "testMetric":
                None,
            "criticalSlicePass":
                False,
            "decision":
                "reject",
            "bytesProcessed":
                bytes_processed,
            "reasonCodes":
                normalize_codes(codes),
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

    aggregate_ok = (
        test_metric
        >= float(body["metricFloor"])
    )

    if not aggregate_ok:
        codes.append(
            "AGGREGATE_FLOOR"
        )

    # -----------------------------------------------------
    # REQUIRED SLICES
    # -----------------------------------------------------

    required_slices = body[
        "requiredSlices"
    ]

    slices_ok = True

    for slice_name in utf8_sorted(
        required_slices.keys()
    ):

        matching = [
            row
            for row in rows
            if row["slice"] == slice_name
        ]

        # Missing required slice.
        if len(matching) == 0:

            codes.append(
                "MISSING_SLICE:"
                + slice_name
            )

            slices_ok = False
            continue

        slice_correct = sum(
            1
            for row in matching
            if row["label"] == row["prediction"]
        )

        slice_metric = round(
            slice_correct / len(matching),
            12,
        )

        slice_floor = float(
            required_slices[slice_name]
        )

        # Inclusive floor.
        if slice_metric < slice_floor:

            codes.append(
                "SLICE_FLOOR:"
                + slice_name
            )

            slices_ok = False

    # -----------------------------------------------------
    # CRITICAL SLICE PASS
    # -----------------------------------------------------
    #
    # This is NOT the final admission decision.
    #
    # It is false for:
    # - invalid lineage
    # - invalid rows
    # - missing slices
    # - failed slice floors
    #
    # It deliberately does NOT include:
    # - aggregate floor
    # - byte limit
    #

    critical_slice_pass = (
        lineage_ok
        and rows_ok
        and slices_ok
    )

    # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    decision = "reject"

    if (
        lineage_ok
        and rows_ok
        and aggregate_ok
        and slices_ok
        and bytes_ok
    ):
        decision = "admit"

    # -----------------------------------------------------
    # FINAL CODES
    # -----------------------------------------------------

    codes = normalize_codes(codes)

    # -----------------------------------------------------
    # EXACT OUTPUT
    # -----------------------------------------------------

    return {
        "runId":
            run_id,
        "selectedTrialId":
            selected_trial_id,
        "datasetDigest":
            dataset_digest,
        "testMetric":
            test_metric,
        "criticalSlicePass":
            critical_slice_pass,
        "decision":
            decision,
        "bytesProcessed":
            bytes_processed,
        "reasonCodes":
            codes,
    }


# =========================================================
# ENDPOINT
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
            {
                "error": "INVALID_INPUT"
            },
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error": "INVALID_INPUT"
            },
            status_code=400,
        )

    # -----------------------------------------------------
    # Unknown/missing phase
    # -----------------------------------------------------

    phase = body.get("phase")

    if phase not in (
        "select",
        "evaluate",
    ):
        return JSONResponse(
            {
                "error": "INVALID_INPUT"
            },
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

            return JSONResponse(
                {
                    "runId":
                        run_id,
                    "selectedTrialId":
                        None,
                    "trainRowIds":
                        [],
                    "evalRowIds":
                        [],
                    "featureNames":
                        [],
                    "datasetDigest":
                        None,
                    "reasonCodes":
                        ["INVALID_INPUT"],
                }
            )

        run_id = body["runId"]

        # Complete original selection input.
        fingerprint = hashlib.sha256(
            compact_json(body).encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # Existing run
        # -------------------------------------------------

        if run_id in RUNS:

            existing = RUNS[run_id]

            # Exact replay.
            if (
                existing.get(
                    "fingerprint"
                )
                == fingerprint
            ):
                return JSONResponse(
                    existing["response"]
                )

            # Same ID, different selection input.
            return JSONResponse(
                {
                    "error":
                        "RUN_ID_CONFLICT"
                },
                status_code=409,
            )

        # -------------------------------------------------
        # New frozen selection
        # -------------------------------------------------

        response = perform_selection(
            body
        )

        successful = (
            response["selectedTrialId"]
            is not None
            and response["datasetDigest"]
            is not None
            and response["reasonCodes"] == []
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

        if not validate_evaluation(
            body
        ):
            return JSONResponse(
                invalid_evaluation_response(
                    body
                )
            )

        return JSONResponse(
            perform_evaluation(body)
        )

    # Defensive fallback.
    return JSONResponse(
        {
            "error": "INVALID_INPUT"
        },
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