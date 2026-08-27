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

# Stateful storage:
# runId -> frozen successful/failed selection
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
            datetime.fromisoformat(
                value[:-1] + "+00:00"
            )
        else:
            datetime.fromisoformat(value)

        return True
    except (ValueError, TypeError):
        return False


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    return datetime.fromisoformat(
        value
    ).astimezone(timezone.utc)


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
        compact_json(
            payload
        ).encode("utf-8")
    ).hexdigest()


# =========================================================
# SELECTION INPUT VALIDATION
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

    row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            return False

        fields = [
            "id",
            "entity",
            "eventTime",
            "predictionTime",
            "version",
            "split",
            "features",
        ]

        if not has_fields(row, fields):
            return False

        row_id = row["id"]

        if (
            not isinstance(row_id, str)
            or not row_id
        ):
            return False

        if row_id in row_ids:
            return False

        row_ids.add(row_id)

        if not isinstance(
            row["entity"],
            str,
        ):
            return False

        if not valid_timestamp(
            row["eventTime"]
        ):
            return False

        if not valid_timestamp(
            row["predictionTime"]
        ):
            return False

        if not safe_int(
            row["version"]
        ):
            return False

        if row["split"] not in (
            "TRAIN",
            "EVAL",
        ):
            return False

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

            # feature["value"] is arbitrary data.
            # Do not interpret its contents.

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
# DEDUPLICATION
# =========================================================

def deduplicate_rows(rows):

    groups = {}

    for row in rows:

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
        if (
            row["version"]
            > old["version"]
        ):
            groups[key] = row
            continue

        # Same version:
        # smallest UTF-8 ID wins.
        if (
            row["version"]
            == old["version"]
            and utf8_key(
                row["id"]
            )
            < utf8_key(
                old["id"]
            )
        ):
            groups[key] = row

    return list(
        groups.values()
    )


# =========================================================
# POINT-IN-TIME FEATURE FILTER
# =========================================================

def get_eligible_features(
    rows,
    forbidden,
):

    if not rows:
        return []

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

    result = []

    for feature_name in common:

        if feature_name in forbidden:
            continue

        eligible = True

        for row in rows:

            available_at = parse_utc(
                row[
                    "features"
                ][feature_name][
                    "availableAt"
                ]
            )

            prediction_time = parse_utc(
                row[
                    "predictionTime"
                ]
            )

            if available_at > prediction_time:
                eligible = False
                break

        if eligible:
            result.append(
                feature_name
            )

    return utf8_sorted(result)


# =========================================================
# TRIAL SELECTION
# =========================================================

def choose_trial(trials):

    eligible = []

    for trial in trials:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get(
            "evalMetric"
        )

        if not finite_number(metric):
            continue

        eligible.append(trial)

    if not eligible:
        return None

    # Highest metric first.
    # Exact tie -> smallest trialId.
    eligible.sort(
        key=lambda x: (
            -float(
                x["evalMetric"]
            ),
            x["trialId"],
        )
    )

    return eligible[0]


# =========================================================
# SELECTION
# =========================================================

def perform_selection(body):

    reason_codes = []

    if (
        len(body["trials"])
        > body["numTrialsLimit"]
    ):
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
        set(
            body[
                "forbiddenFeatures"
            ]
        ),
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

    if not has_fields(
        body,
        required,
    ):
        return False

    if body["phase"] != "evaluate":
        return False

    run_id = body["runId"]

    if (
        not isinstance(run_id, str)
        or not run_id
    ):
        return False

    if not safe_int(
        body["selectedTrialId"]
    ):
        return False

    digest = body[
        "datasetDigest"
    ]

    if (
        not isinstance(digest, str)
        or not DIGEST_RE.fullmatch(
            digest
        )
    ):
        return False

    metric_floor = body[
        "metricFloor"
    ]

    if not finite_number(
        metric_floor
    ):
        return False

    if not (
        0
        <= float(metric_floor)
        <= 1
    ):
        return False

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
            0
            <= float(floor)
            <= 1
        ):
            return False

    if not isinstance(
        body["rows"],
        list,
    ):
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

        # Exactly binary integers.
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            return False

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

    run_id = body["runId"]
    selected_trial_id = body[
        "selectedTrialId"
    ]
    digest = body[
        "datasetDigest"
    ]

    reason_codes = []

    # -----------------------------------------------------
    # 1. FROZEN LINEAGE
    # -----------------------------------------------------

    stored = RUNS.get(run_id)

    lineage_valid = False

    if stored is not None:

        saved = stored.get(
            "response"
        )

        if (
            stored.get(
                "successful"
            )
            is True
            and isinstance(
                saved,
                dict,
            )
            and saved.get(
                "runId"
            )
            == run_id
            and saved.get(
                "selectedTrialId"
            )
            == selected_trial_id
            and saved.get(
                "datasetDigest"
            )
            == digest
            and isinstance(
                saved.get(
                    "datasetDigest"
                ),
                str,
            )
            and DIGEST_RE.fullmatch(
                saved.get(
                    "datasetDigest"
                )
            )
        ):
            lineage_valid = True

    if not lineage_valid:
        reason_codes.append(
            "INVALID_LINEAGE"
        )

    # -----------------------------------------------------
    # 2. BYTE GATE
    # -----------------------------------------------------

    bytes_processed = body[
        "bytesProcessed"
    ]

    max_bytes = body[
        "maxBytes"
    ]

    byte_pass = (
        bytes_processed
        <= max_bytes
    )

    if not byte_pass:
        reason_codes.append(
            "BYTE_LIMIT"
        )

    # -----------------------------------------------------
    # 3. TEST ROW VALIDITY
    # -----------------------------------------------------

    rows = body["rows"]

    rows_valid = validate_test_rows(
        rows
    )

    if not rows_valid:
        reason_codes.append(
            "INVALID_TEST_ROW"
        )

    # -----------------------------------------------------
    # Defaults
    # -----------------------------------------------------

    test_metric = None

    critical_slice_pass = False

    # -----------------------------------------------------
    # 4. EMPTY / INVALID TEST DATA
    # -----------------------------------------------------

    # Contract:
    #
    # empty rows OR any invalid row:
    #
    # testMetric = null
    # aggregate check skipped
    # slice checks skipped
    # criticalSlicePass = false
    #
    if (
        not rows
        or not rows_valid
    ):

        reason_codes = utf8_sorted(
            set(reason_codes)
        )

        return {
            "runId": run_id,
            "selectedTrialId":
                selected_trial_id,
            "datasetDigest": digest,
            "testMetric": None,
            "criticalSlicePass": False,
            "decision": "reject",
            "bytesProcessed":
                bytes_processed,
            "reasonCodes": reason_codes,
        }

    # -----------------------------------------------------
    # 5. AGGREGATE ACCURACY
    # -----------------------------------------------------

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
            body[
                "metricFloor"
            ]
        )
    )

    if not aggregate_pass:
        reason_codes.append(
            "AGGREGATE_FLOOR"
        )

    # -----------------------------------------------------
    # 6. REQUIRED SLICES
    # -----------------------------------------------------

    required_slices = body[
        "requiredSlices"
    ]

    all_slices_pass = True

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
            required_slices[
                slice_name
            ]
        )

        # Inclusive:
        # metric == floor passes.
        if slice_metric < floor:

            reason_codes.append(
                "SLICE_FLOOR:"
                + slice_name
            )

            all_slices_pass = False

    # criticalSlicePass is ONLY the slice gate.
    critical_slice_pass = (
        all_slices_pass
        and lineage_valid
    )

    # -----------------------------------------------------
    # 7. FINAL DECISION
    # -----------------------------------------------------

    decision = "reject"

    if (
        lineage_valid
        and rows_valid
        and len(rows) > 0
        and aggregate_pass
        and all_slices_pass
        and byte_pass
    ):
        decision = "admit"

    # -----------------------------------------------------
    # 8. FINAL REASON CODES
    # -----------------------------------------------------

    reason_codes = utf8_sorted(
        set(reason_codes)
    )

    # -----------------------------------------------------
    # 9. EXACT OUTPUT
    # -----------------------------------------------------

    return {
        "runId": run_id,
        "selectedTrialId":
            selected_trial_id,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass":
            critical_slice_pass,
        "decision": decision,
        "bytesProcessed":
            bytes_processed,
        "reasonCodes": reason_codes,
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
            {
                "error":
                    "INVALID_INPUT"
            },
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            {
                "error":
                    "INVALID_INPUT"
            },
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

        if not validate_selection(
            body
        ):

            return JSONResponse(
                {
                    "runId": (
                        body.get(
                            "runId"
                        )
                        if isinstance(
                            body.get(
                                "runId"
                            ),
                            str,
                        )
                        else ""
                    ),
                    "selectedTrialId":
                        None,
                    "trainRowIds": [],
                    "evalRowIds": [],
                    "featureNames": [],
                    "datasetDigest":
                        None,
                    "reasonCodes": [
                        "INVALID_INPUT"
                    ],
                }
            )

        run_id = body["runId"]

        # Fingerprint the COMPLETE selection input.
        fingerprint = hashlib.sha256(
            compact_json(
                body
            ).encode("utf-8")
        ).hexdigest()

        # -------------------------------------------------
        # Existing run
        # -------------------------------------------------

        if run_id in RUNS:

            existing = RUNS[
                run_id
            ]

            if (
                existing[
                    "fingerprint"
                ]
                == fingerprint
            ):
                # Identical replay.
                return JSONResponse(
                    existing[
                        "response"
                    ]
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

            # Evaluation contract uses
            # INVALID_INPUT as a reason code.
            return JSONResponse(
                {
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
                }
            )

        return JSONResponse(
            perform_evaluation(
                body
            )
        )

    return JSONResponse(
        {
            "error":
                "INVALID_INPUT"
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