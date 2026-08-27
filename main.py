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

# runId -> frozen selection state
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
    except (TypeError, ValueError):
        return False


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        dt = datetime.fromisoformat(
            value[:-1] + "+00:00"
        )
    else:
        dt = datetime.fromisoformat(value)

    return dt.astimezone(timezone.utc)


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


# =========================================================
# DATASET DIGEST
# =========================================================

def make_dataset_digest(
    train_row_ids,
    eval_row_ids,
    feature_names,
):
    payload = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
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
        if not isinstance(
            row["entity"],
            str,
        ):
            return False

        # Timestamp syntax + parseability
        if not valid_timestamp(
            row["eventTime"]
        ):
            return False

        if not valid_timestamp(
            row["predictionTime"]
        ):
            return False

        # IMPORTANT:
        # eventTime is NOT compared against predictionTime.
        # eventTime is used for deduplication only.
        #
        # Point-in-time leakage is checked through
        # feature.availableAt <= predictionTime.

        # Version
        if not safe_int(
            row["version"]
        ):
            return False

        # Split
        if row["split"] not in (
            "TRAIN",
            "EVAL",
        ):
            return False

        # Features
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

            # Feature value is arbitrary data.
            # Do NOT inspect its text.
            if not valid_timestamp(
                feature["availableAt"]
            ):
                return False

    # -----------------------------------------------------
    # trials
    # -----------------------------------------------------

    trials = body["trials"]

    if not isinstance(
        trials,
        list,
    ):
        return False

    trial_ids = set()

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
# DEDUPLICATION
# =========================================================

def deduplicate_rows(rows):

    groups = {}

    for row in rows:

        # Required key:
        # [entity, UTC(eventTime)]
        key = (
            row["entity"],
            parse_utc(
                row["eventTime"]
            ),
        )

        old = groups.get(key)

        if old is None:
            groups[key] = row
            continue

        # Highest version wins.
        if row["version"] > old["version"]:
            groups[key] = row
            continue

        # Same version -> UTF-8-smallest ID.
        if row["version"] == old["version"]:

            if (
                utf8_key(row["id"])
                < utf8_key(old["id"])
            ):
                groups[key] = row

    return list(groups.values())


# =========================================================
# FEATURE ELIGIBILITY
# =========================================================

def get_eligible_features(
    rows,
    forbidden,
):

    if not rows:
        return []

    # Feature must appear in every retained row.
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

        usable = True

        for row in rows:

            available_at = parse_utc(
                row["features"][
                    feature_name
                ]["availableAt"]
            )

            prediction_time = parse_utc(
                row["predictionTime"]
            )

            # Feature is not eligible if unavailable
            # at prediction time for ANY retained row.
            if available_at > prediction_time:
                usable = False
                break

        if usable:
            result.append(
                feature_name
            )

    return utf8_sorted(result)


# =========================================================
# TRIAL SELECTION
# =========================================================

def choose_trial(trials):

    candidates = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get(
            "evalMetric"
        )

        # Only finite successful trials.
        if not finite_number(metric):
            continue

        candidates.append(trial)

    if not candidates:
        return None

    # Highest metric first.
    # Exact metric tie -> smallest trialId.
    candidates.sort(
        key=lambda trial: (
            -float(
                trial["evalMetric"]
            ),
            trial["trialId"],
        )
    )

    return candidates[0]


# =========================================================
# SELECTION
# =========================================================

def perform_selection(body):

    reason_codes = []

    # Trial limit.
    if (
        len(body["trials"])
        > body["numTrialsLimit"]
    ):
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    # Deduplicate BEFORE split and feature processing.
    retained = deduplicate_rows(
        body["rows"]
    )

    # Only TRAIN/EVAL selection rows.
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

    # Shared point-in-time-safe features.
    features = get_eligible_features(
        retained,
        set(body["forbiddenFeatures"]),
    )

    # Select only from successful finite trials.
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

    # Selection failure.
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

    digest = make_dataset_digest(
        train_ids,
        eval_ids,
        features,
    )

    return {
        "runId": body["runId"],
        "selectedTrialId": selected[
            "trialId"
        ],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": features,
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
    if (
        not isinstance(
            body["runId"],
            str,
        )
        or not body["runId"]
    ):
        return False

    # Selected trial
    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    # Digest
    digest = body["datasetDigest"]

    if (
        not isinstance(
            digest,
            str,
        )
        or not DIGEST_RE.fullmatch(
            digest
        )
    ):
        return False

    # Metric floor
    if not finite_number(
        body["metricFloor"]
    ):
        return False

    if not (
        0 <= float(
            body["metricFloor"]
        ) <= 1
    ):
        return False

    # Required slices
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
            not isinstance(
                name,
                str,
            )
            or not name
        ):
            return False

        if not finite_number(
            floor
        ):
            return False

        if not (
            0 <= float(floor) <= 1
        ):
            return False

    # Test rows
    if not isinstance(
        body["rows"],
        list,
    ):
        return False

    # Cost
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

        # Binary integer label.
        if (
            not isinstance(
                label,
                int,
            )
            or isinstance(
                label,
                bool,
            )
            or label not in (0, 1)
        ):
            return False

        # Binary integer prediction.
        if (
            not isinstance(
                prediction,
                int,
            )
            or isinstance(
                prediction,
                bool,
            )
            or prediction not in (0, 1)
        ):
            return False

        # Non-empty slice.
        if (
            not isinstance(
                slice_name,
                str,
            )
            or not slice_name
        ):
            return False

    return True


# =========================================================
# EVALUATION
# =========================================================

def perform_evaluation(body):

    reason_codes = []

    run_id = body["runId"]

    stored = RUNS.get(
        run_id
    )

    # Exact frozen lineage match.
    lineage_valid = (
        stored is not None
        and stored["successful"]
        and stored["response"][
            "selectedTrialId"
        ]
        == body["selectedTrialId"]
        and stored["response"][
            "datasetDigest"
        ]
        == body["datasetDigest"]
    )

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    rows = body["rows"]

    rows_valid = validate_test_rows(
        rows
    )

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    bytes_valid = (
        body["bytesProcessed"]
        <= body["maxBytes"]
    )

    if not bytes_valid:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    test_metric = None
    critical_slice_pass = False

    # -----------------------------------------------------
    # Metric/slice checks only happen when:
    #   - rows are non-empty
    #   - every row is valid
    # -----------------------------------------------------

    if rows and rows_valid:

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

        # Aggregate floor.
        if (
            test_metric
            < float(
                body["metricFloor"]
            )
        ):
            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        all_slices_pass = True

        # UTF-8 order.
        for slice_name in utf8_sorted(
            body[
                "requiredSlices"
            ].keys()
        ):

            slice_rows = [
                row
                for row in rows
                if row["slice"]
                == slice_name
            ]

            # Required slice missing.
            if not slice_rows:

                reason_codes.append(
                    "MISSING_SLICE:"
                    + slice_name
                )

                all_slices_pass = False
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
                body["requiredSlices"][
                    slice_name
                ]
            )

            if slice_metric < floor:

                reason_codes.append(
                    "SLICE_FLOOR:"
                    + slice_name
                )

                all_slices_pass = False

        critical_slice_pass = (
            all_slices_pass
        )

    # Explicit contract:
    # false for invalid input/lineage/test rows,
    # empty rows, missing slices, failed slices.
    if (
        not lineage_valid
        or not rows
        or not rows_valid
    ):
        critical_slice_pass = False

    # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    admit = (
        lineage_valid
        and rows_valid
        and bool(rows)
        and test_metric is not None
        and test_metric
        >= float(
            body["metricFloor"]
        )
        and critical_slice_pass
        and bytes_valid
    )

    decision = (
        "admit"
        if admit
        else "reject"
    )

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
        "reasonCodes": utf8_sorted(
            set(reason_codes)
        ),
    }


# =========================================================
# POST /bqml
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

    if not isinstance(
        body,
        dict,
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    phase = body.get("phase")

    # Explicit contract:
    # unknown/missing phase -> HTTP 400
    # exactly {"error":"INVALID_INPUT"}
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

        # Invalid selection input is returned as a
        # selection response, not the phase-level 400.
        if not validate_selection(body):

            run_id = body.get(
                "runId",
                "",
            )

            if not isinstance(
                run_id,
                str,
            ):
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

        # Fingerprint the complete selection input.
        fingerprint = hashlib.sha256(
            compact_json(body)
            .encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # STATEFUL REPLAY / CONFLICT
        # -------------------------------------------------

        if run_id in RUNS:

            previous = RUNS[
                run_id
            ]

            # Identical replay.
            if (
                previous["fingerprint"]
                == fingerprint
            ):
                return JSONResponse(
                    previous["response"]
                )

            # Same runId, different selection input.
            return JSONResponse(
                {
                    "error":
                    "RUN_ID_CONFLICT"
                },
                status_code=409,
            )

        response = perform_selection(
            body
        )

        successful = (
            response[
                "selectedTrialId"
            ]
            is not None
            and response[
                "datasetDigest"
            ]
            is not None
            and response[
                "reasonCodes"
            ]
            == []
        )

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
            "successful": successful,
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
                {
                    "runId": body.get(
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
                    "testMetric": None,
                    "criticalSlicePass": False,
                    "decision": "reject",
                    "bytesProcessed":
                        body.get(
                            "bytesProcessed"
                        ),
                    "reasonCodes": [
                        "INVALID_INPUT"
                    ],
                }
            )

        return JSONResponse(
            perform_evaluation(
                body
            )
        )


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status": "ok"
    }