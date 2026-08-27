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
# runId -> {
#     "fingerprint": str,
#     "response": dict,
#     "successful": bool
# }
RUNS: dict[str, dict[str, Any]] = {}

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

DIGEST_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


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
            datetime.fromisoformat(
                value[:-1] + "+00:00"
            )
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
    return sorted(
        values,
        key=utf8_key,
    )


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def has_fields(
    obj: Any,
    fields,
) -> bool:
    return (
        isinstance(obj, dict)
        and all(
            field in obj
            for field in fields
        )
    )


# =========================================================
# DATASET DIGEST
# =========================================================

def make_dataset_digest(
    train_row_ids,
    eval_row_ids,
    feature_names,
):
    # Exact required shape and key order.
    payload = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
    }

    encoded = compact_json(
        payload
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


# =========================================================
# SELECTION VALIDATION
# =========================================================

def validate_selection(
    body: Any,
) -> bool:

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

    if not has_fields(
        body,
        required,
    ):
        return False

    if body["phase"] != "select":
        return False

    # -----------------------------------------------------
    # runId
    # -----------------------------------------------------

    run_id = body["runId"]

    if (
        not isinstance(
            run_id,
            str,
        )
        or not run_id
        or len(run_id) > 128
    ):
        return False

    # -----------------------------------------------------
    # forbiddenFeatures
    # -----------------------------------------------------

    forbidden = body[
        "forbiddenFeatures"
    ]

    if not isinstance(
        forbidden,
        list,
    ):
        return False

    if not all(
        isinstance(
            value,
            str,
        )
        for value in forbidden
    ):
        return False

    # -----------------------------------------------------
    # numTrialsLimit
    # -----------------------------------------------------

    limit = body[
        "numTrialsLimit"
    ]

    if (
        not isinstance(
            limit,
            int,
        )
        or isinstance(
            limit,
            bool,
        )
        or limit <= 0
        or limit > SAFE_INT_MAX
    ):
        return False

    # -----------------------------------------------------
    # rows
    # -----------------------------------------------------

    rows = body["rows"]

    if not isinstance(
        rows,
        list,
    ):
        return False

    # Selection rows must be non-empty.
    if len(rows) == 0:
        return False

    row_ids = set()

    for row in rows:

        if not isinstance(
            row,
            dict,
        ):
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

        if not has_fields(
            row,
            required_row,
        ):
            return False

        # -------------------------------------------------
        # Row ID
        # -------------------------------------------------

        row_id = row["id"]

        if (
            not isinstance(
                row_id,
                str,
            )
            or not row_id
        ):
            return False

        # IDs must be unique in the supplied array.
        if row_id in row_ids:
            return False

        row_ids.add(row_id)

        # -------------------------------------------------
        # Entity
        # -------------------------------------------------

        if not isinstance(
            row["entity"],
            str,
        ):
            return False

        # -------------------------------------------------
        # Timestamps
        # -------------------------------------------------

        if not valid_timestamp(
            row["eventTime"]
        ):
            return False

        if not valid_timestamp(
            row["predictionTime"]
        ):
            return False

        # -------------------------------------------------
        # Version
        # -------------------------------------------------

        if not safe_int(
            row["version"]
        ):
            return False

        # -------------------------------------------------
        # Split
        # -------------------------------------------------

        if row["split"] not in (
            "TRAIN",
            "EVAL",
        ):
            return False

        # -------------------------------------------------
        # Features
        # -------------------------------------------------

        features = row[
            "features"
        ]

        if not isinstance(
            features,
            dict,
        ):
            return False

        for (
            feature_name,
            feature,
        ) in features.items():

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

            # "value" is arbitrary data.
            # Never parse or interpret its text.
            if not valid_timestamp(
                feature[
                    "availableAt"
                ]
            ):
                return False

    # -----------------------------------------------------
    # trials
    # -----------------------------------------------------

    trials = body[
        "trials"
    ]

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

        trial_id = trial[
            "trialId"
        ]

        if not safe_int(
            trial_id
        ):
            return False

        if trial_id in trial_ids:
            return False

        trial_ids.add(
            trial_id
        )

        status = trial[
            "status"
        ]

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

def deduplicate_rows(
    rows,
):
    groups = {}

    for row in rows:

        # EXACT deduplication key:
        #
        # [entity, UTC(eventTime)]
        #
        key = (
            row["entity"],
            parse_utc(
                row["eventTime"]
            ),
        )

        existing = groups.get(
            key
        )

        if existing is None:
            groups[key] = row
            continue

        # Highest integer version wins.
        if (
            row["version"]
            > existing["version"]
        ):
            groups[key] = row
            continue

        # Same version:
        # UTF-8-byte-smallest ID wins.
        if (
            row["version"]
            == existing["version"]
        ):
            if (
                utf8_key(
                    row["id"]
                )
                <
                utf8_key(
                    existing["id"]
                )
            ):
                groups[key] = row

    return list(
        groups.values()
    )


# =========================================================
# POINT-IN-TIME FEATURE ELIGIBILITY
# =========================================================

def get_eligible_features(
    rows,
    forbidden,
):

    if not rows:
        return []

    # A feature must occur in EVERY retained row.
    common = set(
        rows[0][
            "features"
        ].keys()
    )

    for row in rows[1:]:

        common.intersection_update(
            row[
                "features"
            ].keys()
        )

    eligible = []

    for feature_name in common:

        # Forbidden feature.
        if feature_name in forbidden:
            continue

        feature_ok = True

        for row in rows:

            feature = row[
                "features"
            ][feature_name]

            available_at = parse_utc(
                feature[
                    "availableAt"
                ]
            )

            prediction_time = parse_utc(
                row[
                    "predictionTime"
                ]
            )

            # Point-in-time rule:
            #
            # availableAt <= predictionTime
            #
            # Equality is allowed.
            if (
                available_at
                > prediction_time
            ):
                feature_ok = False
                break

        if feature_ok:
            eligible.append(
                feature_name
            )

    # UTF-8-byte ordering.
    return utf8_sorted(
        eligible
    )


# =========================================================
# TRIAL SELECTION
# =========================================================

def choose_trial(
    trials,
):

    candidates = []

    for trial in trials:

        # Only SUCCEEDED trials.
        if (
            trial["status"]
            != "SUCCEEDED"
        ):
            continue

        metric = trial.get(
            "evalMetric"
        )

        # Only finite metrics.
        if not finite_number(
            metric
        ):
            continue

        candidates.append(
            trial
        )

    if not candidates:
        return None

    # Highest metric.
    #
    # Exact metric tie:
    # smallest integer trialId.
    candidates.sort(
        key=lambda trial: (
            -float(
                trial[
                    "evalMetric"
                ]
            ),
            trial[
                "trialId"
            ],
        )
    )

    return candidates[0]


# =========================================================
# PERFORM SELECTION
# =========================================================

def perform_selection(
    body,
):

    reason_codes = []

    # -----------------------------------------------------
    # Trial limit
    # -----------------------------------------------------

    if (
        len(body["trials"])
        > body["numTrialsLimit"]
    ):
        reason_codes.append(
            "TRIAL_LIMIT_EXCEEDED"
        )

    # -----------------------------------------------------
    # Deduplicate FIRST
    # -----------------------------------------------------

    retained = deduplicate_rows(
        body["rows"]
    )

    # -----------------------------------------------------
    # Split IDs
    # -----------------------------------------------------

    train_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"]
            == "TRAIN"
        ]
    )

    eval_ids = utf8_sorted(
        [
            row["id"]
            for row in retained
            if row["split"]
            == "EVAL"
        ]
    )

    # -----------------------------------------------------
    # Shared point-in-time-safe features
    # -----------------------------------------------------

    feature_names = (
        get_eligible_features(
            retained,
            set(
                body[
                    "forbiddenFeatures"
                ]
            ),
        )
    )

    # -----------------------------------------------------
    # Frozen trial selection
    # -----------------------------------------------------

    selected = choose_trial(
        body["trials"]
    )

    if selected is None:
        reason_codes.append(
            "NO_SUCCESSFUL_TRIAL"
        )

    # Sort + deduplicate codes.
    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # -----------------------------------------------------
    # Selection failure
    # -----------------------------------------------------

    if reason_codes:

        return {
            "runId": body[
                "runId"
            ],
            "selectedTrialId": None,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": feature_names,
            "datasetDigest": None,
            "reasonCodes": reason_codes,
        }

    # -----------------------------------------------------
    # Dataset digest
    # -----------------------------------------------------

    digest = make_dataset_digest(
        train_ids,
        eval_ids,
        feature_names,
    )

    return {
        "runId": body[
            "runId"
        ],
        "selectedTrialId": selected[
            "trialId"
        ],
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
        "datasetDigest": digest,
        "reasonCodes": [],
    }


# =========================================================
# EVALUATION INPUT VALIDATION
# =========================================================

def validate_evaluation(
    body,
):

    if not isinstance(
        body,
        dict,
    ):
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

    if not has_fields(
        body,
        required,
    ):
        return False

    if body["phase"] != "evaluate":
        return False

    # -----------------------------------------------------
    # runId
    # -----------------------------------------------------

    if (
        not isinstance(
            body["runId"],
            str,
        )
        or not body["runId"]
    ):
        return False

    # -----------------------------------------------------
    # selectedTrialId
    # -----------------------------------------------------

    if not safe_int(
        body[
            "selectedTrialId"
        ]
    ):
        return False

    # -----------------------------------------------------
    # datasetDigest
    # -----------------------------------------------------

    digest = body[
        "datasetDigest"
    ]

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

    # -----------------------------------------------------
    # metricFloor
    # -----------------------------------------------------

    if not finite_number(
        body["metricFloor"]
    ):
        return False

    if not (
        0
        <= float(
            body["metricFloor"]
        )
        <= 1
    ):
        return False

    # -----------------------------------------------------
    # requiredSlices
    # -----------------------------------------------------

    required_slices = body[
        "requiredSlices"
    ]

    if not isinstance(
        required_slices,
        dict,
    ):
        return False

    for (
        name,
        floor,
    ) in required_slices.items():

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
            0
            <= float(floor)
            <= 1
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
        body[
            "bytesProcessed"
        ]
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

def validate_test_rows(
    rows,
):

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

        label = row[
            "label"
        ]

        prediction = row[
            "prediction"
        ]

        slice_name = row[
            "slice"
        ]

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
            or label not in (
                0,
                1,
            )
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
            or prediction not in (
                0,
                1,
            )
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
# FINAL EVALUATION
# =========================================================

def perform_evaluation(
    body,
):

    reason_codes = []

    run_id = body[
        "runId"
    ]

    selected_trial_id = body[
        "selectedTrialId"
    ]

    digest = body[
        "datasetDigest"
    ]

    # =====================================================
    # FROZEN LINEAGE
    # =====================================================

    stored = RUNS.get(
        run_id
    )

    lineage_valid = False

    if stored is not None:

        saved_response = stored.get(
            "response"
        )

        if (
            stored.get(
                "successful"
            )
            is True
            and isinstance(
                saved_response,
                dict,
            )
            and saved_response.get(
                "selectedTrialId"
            )
            == selected_trial_id
            and saved_response.get(
                "datasetDigest"
            )
            == digest
        ):
            lineage_valid = True

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    # =====================================================
    # COST GATE
    # =====================================================

    bytes_processed = body[
        "bytesProcessed"
    ]

    max_bytes = body[
        "maxBytes"
    ]

    bytes_pass = (
        bytes_processed
        <= max_bytes
    )

    if not bytes_pass:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    # =====================================================
    # TEST ROWS
    # =====================================================

    rows = body[
        "rows"
    ]

    rows_valid = validate_test_rows(
        rows
    )

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # =====================================================
    # DEFAULT OUTPUT STATE
    # =====================================================

    test_metric = None

    # Contract:
    #
    # criticalSlicePass is false for:
    # - invalid input
    # - invalid lineage
    # - invalid test row
    # - empty rows
    # - missing required slice
    # - failed required slice floor
    #
    # It does NOT summarize aggregate or byte gates.
    critical_slice_pass = False

    # =====================================================
    # AGGREGATE + SLICE CHECKS
    # =====================================================

    # Empty rows or invalid rows:
    # do NOT calculate metric or slices.
    if (
        rows
        and rows_valid
    ):

        # -------------------------------------------------
        # Aggregate accuracy
        # -------------------------------------------------

        correct = sum(
            1
            for row in rows
            if row[
                "label"
            ]
            == row[
                "prediction"
            ]
        )

        test_metric = round(
            correct / len(rows),
            12,
        )

        aggregate_pass = (
            test_metric
            >= float(
                body[
                    "metricFloor"
                ]
            )
        )

        if not aggregate_pass:
            reason_codes.append(
                "AGGREGATE_FLOOR"
            )

        # -------------------------------------------------
        # Required slices
        # -------------------------------------------------

        required_slices = body[
            "requiredSlices"
        ]

        all_required_slices_pass = True

        for slice_name in utf8_sorted(
            required_slices.keys()
        ):

            slice_rows = [
                row
                for row in rows
                if row[
                    "slice"
                ]
                == slice_name
            ]

            # Required slice does not exist.
            if not slice_rows:

                reason_codes.append(
                    "MISSING_SLICE:"
                    + slice_name
                )

                all_required_slices_pass = False
                continue

            slice_correct = sum(
                1
                for row in slice_rows
                if row[
                    "label"
                ]
                == row[
                    "prediction"
                ]
            )

            slice_metric = round(
                slice_correct
                / len(slice_rows),
                12,
            )

            slice_floor = float(
                required_slices[
                    slice_name
                ]
            )

            # Inclusive floor.
            if (
                slice_metric
                < slice_floor
            ):

                reason_codes.append(
                    "SLICE_FLOOR:"
                    + slice_name
                )

                all_required_slices_pass = False

        # This ONLY summarizes required-slice gates.
        critical_slice_pass = (
            all_required_slices_pass
        )

    # Empty/invalid rows explicitly force false.
    if (
        not rows
        or not rows_valid
    ):
        critical_slice_pass = False

    # Invalid lineage also forces false.
    if not lineage_valid:
        critical_slice_pass = False

    # =====================================================
    # FINAL DECISION
    # =====================================================

    aggregate_pass = (
        test_metric is not None
        and test_metric
        >= float(
            body[
                "metricFloor"
            ]
        )
    )

    admit = (
        lineage_valid
        and rows_valid
        and bool(rows)
        and aggregate_pass
        and critical_slice_pass
        and bytes_pass
    )

    decision = (
        "admit"
        if admit
        else "reject"
    )

    # =====================================================
    # SORT + DEDUP REASON CODES
    # =====================================================

    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # =====================================================
    # EXACT EVALUATION OUTPUT
    # =====================================================

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
# POST /bqml
# =========================================================

@app.post("/bqml")
async def bqml(
    request: Request,
):

    # -----------------------------------------------------
    # JSON parsing
    # -----------------------------------------------------

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {
                "error":
                "INVALID_INPUT"
            },
            status_code=400,
        )

    if not isinstance(
        body,
        dict,
    ):
        return JSONResponse(
            {
                "error":
                "INVALID_INPUT"
            },
            status_code=400,
        )

    phase = body.get(
        "phase"
    )

    # -----------------------------------------------------
    # Missing/unknown phase
    #
    # EXACT required response.
    # -----------------------------------------------------

    if phase not in (
        "select",
        "evaluate",
    ):
        return JSONResponse(
            {
                "error":
                "INVALID_INPUT"
            },
            status_code=400,
        )

    # =====================================================
    # SELECT
    # =====================================================

    if phase == "select":

        # Selection validation failure is a response
        # containing INVALID_INPUT.
        if not validate_selection(
            body
        ):

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

        run_id = body[
            "runId"
        ]

        # Complete selection input fingerprint.
        fingerprint = hashlib.sha256(
            compact_json(
                body
            ).encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # STATEFUL RUN ID
        # -------------------------------------------------

        if run_id in RUNS:

            previous = RUNS[
                run_id
            ]

            # Identical replay.
            if (
                previous[
                    "fingerprint"
                ]
                == fingerprint
            ):
                return JSONResponse(
                    previous[
                        "response"
                    ]
                )

            # Same runId but different selection input.
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

        # Evaluation malformed input still returns the
        # evaluation-shaped response.
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
# HEALTH CHECK
# =========================================================

@app.get("/")
def health():
    return {
        "status": "ok"
    }