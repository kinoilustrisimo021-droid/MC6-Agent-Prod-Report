import io
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import polars as pl
import streamlit as st


st.set_page_config(page_title="Agent Production Summary", page_icon="📊", layout="wide")

st.title("📊 Agent Production Summary")
st.caption("Optimized for large Excel uploads: reads only needed columns, aggregates in Polars, then exports a compact formatted summary.")

if "agent_prod_result" not in st.session_state:
    st.session_state.agent_prod_result = None


LEFT_COLS = ["CMS User", "Name", "Placement"]
DAY_COLS = [
    "Connected Calls",
    "Connected Calls Target",
    "Connected %",
    "RPC Count",
    "RPC Target",
    "RPC %",
    "RPC OB",
    "RPC OB Target",
    "RPC OB %",
    "PTP Count",
    "PTP Target",
    "PTP %",
    "PTP OB",
    "PTP OB Target",
    "PTP OB %",
    "Kept Count",
    "Kept Target",
    "Kept Count %",
    "KEPT OB",
    "KEPT OB Target",
    "KEPT OB %",
]

ACTIVITY_REQUIRED_ALIASES = {
    "Date": ["Date"],
    "Remark By": ["Remark By"],
    "Call Duration": ["Call Duration"],
    "Talk Time Duration": ["Talk Time Duration"],
    "Old IC": ["Old IC"],
    "Status": ["Status"],
    "PTP Amount": ["PTP Amount"],
    "Claim Paid Amount": ["Claim Paid Amount"],
}

CMS_REQUIRED_ALIASES = {
    "Name": ["Name"],
    "CMS User": ["CMS User"],
    "Placement": ["Placement"],
}

TARGET_REQUIRED_ALIASES = {
    "Placement": ["Placement"],
    "Connected Calls Target": ["Connected Calls Target"],
    "RPC Target": ["RPC Target"],
    "RPC OB Target": ["RPC OB Target"],
    "PTP Target": ["PTP Target"],
    "PTP OB Target": ["PTP OB Target"],
    "Kept Target": ["Kept Target", "Kept Count Target", "KEPT Target", "Kept_Count_Target"],
    "KEPT OB Target": ["KEPT OB Target", "Kept OB Target", "KEPT_OB_Target"],
}

OLD_IC_REQUIRED_ALIASES = {
    "Old IC": ["Old IC"],
    "Placement": ["Placement"],
    "Principal": ["Principal"],
}

TARGET_TO_DAY_COL = {
    "Connected Calls Target": "Connected Calls Target",
    "RPC Target": "RPC Target",
    "RPC OB Target": "RPC OB Target",
    "PTP Target": "PTP Target",
    "PTP OB Target": "PTP OB Target",
    "Kept Target": "Kept Target",
    "KEPT OB Target": "KEPT OB Target",
}

PERCENTAGE_COLS = {
    "Connected %": ("Connected Calls", "Connected Calls Target"),
    "RPC %": ("RPC Count", "RPC Target"),
    "RPC OB %": ("RPC OB", "RPC OB Target"),
    "PTP %": ("PTP Count", "PTP Target"),
    "PTP OB %": ("PTP OB", "PTP OB Target"),
    "Kept Count %": ("Kept Count", "Kept Target"),
    "KEPT OB %": ("KEPT OB", "KEPT OB Target"),
}

STATUS_RPC_KEYWORDS = ["RPC", "POS CLIENT", "POSITIVE CLIENT"]
STATUS_KEPT_KEYWORDS = ["CONFIRMED", "PAID", "PAYMENT"]
PTP_EXCLUSIONS = [
    "LS VIA SOCMED - T3 PTP REMINDER",
    "PTP_FF UP - KEEPS ON RINGING (KOR)",
    "PTP_FF UP - BUSY",
    "PTP_FF UP - UNCONTACTABLE",
    "PTP_FF UP - ANSWERED WILL SETTLE",
    "PTP_FF UP - ANSWERED RENEGO",
    "PYMT REMINDER (FF UP) - KEEPS ON RINGING",
    "PAYMENT REMINDER (FF UP) - BUSY",
    "PAYMENT REMINDER (FF UP) - UNCONTACTABLE",
    "PYMT REMINDER (FF UP) - ANSWERED WILL SETTLE",
    "PAYMENT REMINDER (FF UP) - ANSWERED RENEGO",
    "PAYMENT REMINDER (FF UP) - KEEPS ON RINGING",
    "PAYMENT REMINDER (FF UP) - WILL SETTLE",
    "LETTER SENT VIA SOCMED - T3 PTP REMINDER",
    "PAYMENT REMINDER - KEEPS ON RINGING",
    "PAYMENT REMINDER - BUSY",
    "PAYMENT REMINDER - UNCONTACTABLE",
    "PAYMENT REMINDER - ANSWERED WILL SETTLE",
    "PAYMENT REMINDER - ANSWERED RENEGO",
    "LS VIA SOC MED - PTP REMINDER",
    "PTP FF UP - CLIENT ANSWERED AND WILL SETTLE",
    "PTP FF UP - NO ANSWER_SENT REMINDER",
    "PAYMENT REMINDER_FF UP - ANSWERED WILL SETTLE",
    "PAYMENT REMINDER_FF UP - UNCONTACTABLE",
    "PTP FF UP - NO ANSWER_SENT PAYMENT REMINDER",
    "PAYMENT REMINDER (FF UP) - KEEPS ON RINGING (",
    "PAYMENT REMINDER (FF UP) - ANSWERED WILL SETT",
]


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_key(value):
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value).upper())


def resolve_columns(header_columns, aliases_map):
    normalized = {normalize_key(col): col for col in header_columns}
    resolved = {}
    for target_name, aliases in aliases_map.items():
        found = None
        for alias in aliases:
            found = normalized.get(normalize_key(alias))
            if found:
                break
        if not found:
            raise ValueError(f"Missing required column: {target_name}")
        resolved[target_name] = found
    return resolved


def read_excel_selected(file_name, file_bytes, aliases_map, preview_rows=None):
    preview_scan = pl.read_excel(
        io.BytesIO(file_bytes),
        engine="calamine",
        read_options={"n_rows": 10},
        infer_schema_length=0,
    )
    resolved = resolve_columns(preview_scan.columns, aliases_map)
    usecols = list(dict.fromkeys(resolved.values()))
    schema_overrides = {col: pl.Utf8 for col in usecols}
    read_options = {"n_rows": preview_rows} if preview_rows is not None else None
    df = pl.read_excel(
        io.BytesIO(file_bytes),
        engine="calamine",
        columns=usecols,
        schema_overrides=schema_overrides,
        infer_schema_length=0,
        read_options=read_options,
    )
    rename_map = {source: target for target, source in resolved.items()}
    return df.rename(rename_map)


def read_csv_selected(file_bytes, aliases_map, preview_rows=None):
    scan_rows = preview_rows if preview_rows is not None else 50
    preview_scan = pl.read_csv(
        io.BytesIO(file_bytes),
        infer_schema_length=0,
        n_rows=scan_rows,
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    resolved = resolve_columns(preview_scan.columns, aliases_map)
    usecols = list(dict.fromkeys(resolved.values()))
    df = pl.read_csv(
        io.BytesIO(file_bytes),
        infer_schema_length=0,
        columns=usecols,
        n_rows=preview_rows,
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    rename_map = {source: target for target, source in resolved.items()}
    return df.rename(rename_map)


def read_table_selected(file_name, file_bytes, aliases_map, preview_rows=None):
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return read_csv_selected(file_bytes, aliases_map, preview_rows=preview_rows)
    return read_excel_selected(file_name, file_bytes, aliases_map, preview_rows=preview_rows)


def read_uploaded_file(uploaded_file):
    return uploaded_file.name, uploaded_file.getvalue()


def load_activity_logs_fast(uploaded_files):
    payloads = [read_uploaded_file(file) for file in uploaded_files]
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(payloads)))) as pool:
        frames = list(pool.map(lambda item: read_table_selected(item[0], item[1], ACTIVITY_REQUIRED_ALIASES), payloads))
    if not frames:
        raise ValueError("Please upload at least one activity log file.")
    return pl.concat(frames, how="vertical_relaxed")


def load_reference_df(uploaded_file, aliases_map):
    file_name, file_bytes = read_uploaded_file(uploaded_file)
    return read_table_selected(file_name, file_bytes, aliases_map)


def duration_expr(column_name):
    text = pl.col(column_name).cast(pl.Utf8).fill_null("").str.strip_chars()
    colon_count = text.str.count_matches(":")
    parts = text.str.split_exact(":", 2)
    part0 = parts.struct.field("field_0").cast(pl.Float64, strict=False).fill_null(0.0)
    part1 = parts.struct.field("field_1").cast(pl.Float64, strict=False).fill_null(0.0)
    part2 = parts.struct.field("field_2").cast(pl.Float64, strict=False).fill_null(0.0)
    return (
        pl.when(text == "")
        .then(pl.lit(0.0))
        .when(colon_count >= 2)
        .then(part0 * 3600 + part1 * 60 + part2)
        .when(colon_count == 1)
        .then(part0 * 60 + part1)
        .otherwise(
            text.str.replace_all(",", "")
            .str.extract(r"(-?\d+(?:\.\d+)?)", 1)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
        )
    )


def numeric_expr(column_name):
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .fill_null("")
        .str.replace_all(",", "")
        .str.extract(r"(-?\d+(?:\.\d+)?)", 1)
        .cast(pl.Float64, strict=False)
        .fill_null(0.0)
    )


def contains_any_expr(column_name, keywords):
    expr = pl.lit(False)
    upper_col = pl.col(column_name).fill_null("").cast(pl.Utf8).str.to_uppercase()
    for keyword in keywords:
        expr = expr | upper_col.str.contains(keyword, literal=True)
    return expr


def contains_none_expr(column_name, keywords):
    expr = pl.lit(True)
    upper_col = pl.col(column_name).fill_null("").cast(pl.Utf8).str.to_uppercase()
    for keyword in keywords:
        expr = expr & (~upper_col.str.contains(keyword, literal=True))
    return expr


def prepare_activity_polars(activity):
    activity = activity.with_columns(
        [
            pl.col("Date").cast(pl.Utf8).str.strptime(pl.Date, strict=False).dt.strftime("%Y-%m-%d").alias("Date"),
            pl.col("Remark By").cast(pl.Utf8).fill_null("").str.strip_chars().alias("CMS User"),
            pl.col("Old IC").cast(pl.Utf8).fill_null("").str.strip_chars().alias("Old IC"),
            pl.col("Old IC").cast(pl.Utf8).fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]+", "").alias("Old IC Key"),
            pl.col("Status").cast(pl.Utf8).fill_null("").str.to_uppercase().alias("Status Text"),
            duration_expr("Talk Time Duration").alias("Talk Seconds"),
            duration_expr("Call Duration").alias("Call Seconds"),
            numeric_expr("PTP Amount").alias("PTP Amount Num"),
            numeric_expr("Claim Paid Amount").alias("Claim Paid Amount Num"),
        ]
    ).filter(
        pl.col("Date").is_not_null()
        & (pl.col("CMS User") != "")
        & (pl.col("Old IC") != "")
    )

    activity = activity.with_columns(
        [
            (pl.col("Talk Seconds") > 0).alias("Is Connected"),
            (pl.col("Call Seconds") > 0).alias("Is Active CMS"),
            (
                (pl.col("PTP Amount Num") > 0)
                & pl.col("Status Text").str.contains("PTP", literal=True)
                & contains_none_expr("Status Text", [item.upper() for item in PTP_EXCLUSIONS])
            ).alias("Is PTP"),
            contains_any_expr("Status Text", STATUS_RPC_KEYWORDS).alias("Is RPC Base"),
            (
                (pl.col("Claim Paid Amount Num") > 0)
                & contains_any_expr("Status Text", STATUS_KEPT_KEYWORDS)
                & (~pl.col("Status Text").str.contains("PTP", literal=True))
                & (~pl.col("Status Text").str.contains("RPC", literal=True))
            ).alias("Is Kept"),
        ]
    ).with_columns(
        [
            (pl.col("Is RPC Base") | pl.col("Is PTP") | pl.col("Is Kept")).alias("Is RPC"),
        ]
    )
    return activity


def prepare_reference_polars(cms_pd, target_pd, old_ic_pd):
    cms = (
        cms_pd
        .with_columns(
            [
                pl.col("CMS User").cast(pl.Utf8).fill_null("").str.strip_chars(),
                pl.col("Name").cast(pl.Utf8).fill_null("").str.strip_chars(),
                pl.col("Placement").cast(pl.Utf8).fill_null("").str.strip_chars(),
            ]
        )
        .filter(pl.col("CMS User") != "")
        .unique(subset=["CMS User"], keep="first")
        .with_row_index("ref_order")
    )

    target = (
        target_pd
        .with_columns(
            [
                pl.col("Placement").cast(pl.Utf8).fill_null("").str.strip_chars(),
                pl.col("Placement").cast(pl.Utf8).fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]+", "").alias("Placement Key"),
                pl.col("Connected Calls Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("RPC Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("RPC OB Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("PTP Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("PTP OB Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("Kept Target").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("KEPT OB Target").cast(pl.Float64, strict=False).fill_null(0.0),
            ]
        )
        .filter(pl.col("Placement") != "")
        .unique(subset=["Placement Key"], keep="first")
    )

    old_ic = (
        old_ic_pd
        .with_columns(
            [
                pl.col("Old IC").cast(pl.Utf8).fill_null("").str.strip_chars(),
                pl.col("Old IC").cast(pl.Utf8).fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]+", "").alias("Old IC Key"),
                pl.col("Placement").cast(pl.Utf8).fill_null("").str.strip_chars(),
                numeric_expr("Principal").alias("Principal"),
            ]
        )
        .filter(pl.col("Old IC") != "")
        .select(["Old IC Key", "Placement", "Principal"])
        .unique(subset=["Old IC Key"], keep="first")
    )

    return cms, target, old_ic


def metric_by_agent_date(activity, old_ic, flag_col, count_name, ob_name=None):
    metric = (
        activity.filter(pl.col(flag_col))
        .select(["CMS User", "Date", "Old IC Key"])
        .unique()
        .join(old_ic.select(["Old IC Key", "Principal"]), on="Old IC Key", how="left")
        .with_columns(pl.col("Principal").fill_null(0.0))
    )
    aggs = [pl.len().alias(count_name)]
    if ob_name:
        aggs.append(pl.col("Principal").sum().alias(ob_name))
    return metric.group_by(["CMS User", "Date"]).agg(aggs)


def build_summary_long(activity, cms_ref, target_ref, old_ic_ref):
    active_users = (
        activity.filter(pl.col("Is Active CMS"))
        .select("CMS User")
        .unique()
    )
    if active_users.height == 0:
        raise ValueError("No CMS User found from activity logs where Call Duration > 0.")

    dates = activity.select("Date").unique().sort("Date")
    if dates.height == 0:
        raise ValueError("No valid dates found in the activity logs.")

    connected = metric_by_agent_date(activity, old_ic_ref, "Is Connected", "Connected Calls")
    rpc = metric_by_agent_date(activity, old_ic_ref, "Is RPC", "RPC Count", "RPC OB")
    ptp = metric_by_agent_date(activity, old_ic_ref, "Is PTP", "PTP Count", "PTP OB")
    kept = metric_by_agent_date(activity, old_ic_ref, "Is Kept", "Kept Count", "KEPT OB")

    agent_order = (
        active_users
        .join(cms_ref, on="CMS User", how="inner")
        .sort(["ref_order", "CMS User"])
        .drop("ref_order")
    )

    if agent_order.height == 0:
        raise ValueError("No matching CMS User found between activity logs and the uploaded CMS reference.")

    base = agent_order.join(dates, how="cross")
    base = base.with_columns(
        pl.col("Placement").cast(pl.Utf8).fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]+", "").alias("Placement Key")
    )
    base = (
        base.join(connected, on=["CMS User", "Date"], how="left")
        .join(rpc, on=["CMS User", "Date"], how="left")
        .join(ptp, on=["CMS User", "Date"], how="left")
        .join(kept, on=["CMS User", "Date"], how="left")
        .join(target_ref, on="Placement Key", how="left")
        .with_columns(
            [
                pl.col("Connected Calls").fill_null(0).cast(pl.Int64),
                pl.col("RPC Count").fill_null(0).cast(pl.Int64),
                pl.col("PTP Count").fill_null(0).cast(pl.Int64),
                pl.col("Kept Count").fill_null(0).cast(pl.Int64),
                pl.col("RPC OB").fill_null(0.0),
                pl.col("PTP OB").fill_null(0.0),
                pl.col("KEPT OB").fill_null(0.0),
                pl.col("Connected Calls Target").fill_null(0.0),
                pl.col("RPC Target").fill_null(0.0),
                pl.col("RPC OB Target").fill_null(0.0),
                pl.col("PTP Target").fill_null(0.0),
                pl.col("PTP OB Target").fill_null(0.0),
                pl.col("Kept Target").fill_null(0.0),
                pl.col("KEPT OB Target").fill_null(0.0),
            ]
        )
    )

    for pct_col, (num_col, den_col) in PERCENTAGE_COLS.items():
        base = base.with_columns(
            pl.when(pl.col(den_col) > 0)
            .then(pl.col(num_col) / pl.col(den_col))
            .otherwise(None)
            .alias(pct_col)
        )

    return base.select(LEFT_COLS + ["Date"] + DAY_COLS)


def build_export_matrix(summary_long):
    rows = summary_long.to_dicts()
    dates = sorted({row["Date"] for row in rows})
    users = []
    seen = set()
    for row in rows:
        key = row["CMS User"]
        if key not in seen:
            seen.add(key)
            users.append(
                {
                    "CMS User": row["CMS User"],
                    "Name": row["Name"],
                    "Placement": row["Placement"],
                }
            )

    lookup = {(row["CMS User"], row["Date"]): row for row in rows}
    export_rows = []
    for user in users:
        out = {
            "CMS User": user["CMS User"],
            "Name": user["Name"],
            "Placement": user["Placement"],
        }
        for date_value in dates:
            source = lookup.get((user["CMS User"], date_value), {})
            for metric in DAY_COLS:
                out[(date_value, metric)] = source.get(metric)
        export_rows.append(out)
    return export_rows, dates


def build_month_label(dates):
    if not dates:
        return "No Dates"

    parsed = pd.to_datetime(sorted(dates))
    start_date = parsed.min()
    end_date = parsed.max()

    if start_date.year == end_date.year:
        if start_date.month == end_date.month:
            return start_date.strftime("%b %Y")
        return f"{start_date.strftime('%b')}-{end_date.strftime('%b %Y')}"

    return f"{start_date.strftime('%b %Y')}-{end_date.strftime('%b %Y')}"


def build_monthly_summaries(summary_long):
    monthly_base = summary_long.with_columns(
        [
            pl.col("Date").cast(pl.Utf8).str.slice(0, 7).alias("Month Key"),
            pl.col("Date").cast(pl.Utf8).str.strptime(pl.Date, strict=False).dt.strftime("%b %Y").alias("Month Label"),
        ]
    )

    monthly = (
        monthly_base.group_by(["Month Key", "Month Label"] + LEFT_COLS, maintain_order=True)
        .agg(
            [
                pl.col("Connected Calls").sum().alias("Total Connected Calls"),
                pl.col("RPC Count").sum().alias("RPC Count"),
                pl.col("RPC OB").sum().alias("RPC OB"),
                pl.col("PTP Count").sum().alias("PTP Count"),
                pl.col("PTP OB").sum().alias("PTP OB"),
                pl.col("Kept Count").sum().alias("KEPT Count"),
                pl.col("KEPT OB").sum().alias("KEPT OB"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("Total Connected Calls") > 0)
                .then(pl.col("RPC Count") / pl.col("Total Connected Calls"))
                .otherwise(None)
                .alias("RPC Rate"),
                pl.when(pl.col("RPC Count") > 0)
                .then(pl.col("PTP Count") / pl.col("RPC Count"))
                .otherwise(None)
                .alias("PTP Rate"),
                pl.when(pl.col("PTP Count") > 0)
                .then(pl.col("KEPT Count") / pl.col("PTP Count"))
                .otherwise(None)
                .alias("KEPT Rate"),
            ]
        )
        .sort(["Month Key"] + LEFT_COLS)
    )
    return monthly


def build_month_ai_notes(month_rows, top_count, middle_count, low_count):
    total_agents = max(len(month_rows), 1)
    avg_rpc_rate = sum(float(row.get("RPC Rate") or 0.0) for row in month_rows) / total_agents
    avg_ptp_rate = sum(float(row.get("PTP Rate") or 0.0) for row in month_rows) / total_agents
    avg_kept_rate = sum(float(row.get("KEPT Rate") or 0.0) for row in month_rows) / total_agents

    strongest_rpc = max(month_rows, key=lambda row: float(row.get("RPC Rate") or 0.0))
    strongest_ptp = max(month_rows, key=lambda row: float(row.get("PTP Rate") or 0.0))
    strongest_kept = max(month_rows, key=lambda row: float(row.get("KEPT Rate") or 0.0))

    summary_lines = [
        f"Team mix: {top_count} top performer(s), {middle_count} middle performer(s), and {low_count} low performer(s).",
        f"Average monthly conversion is RPC {avg_rpc_rate:.2%}, PTP {avg_ptp_rate:.2%}, and KEPT {avg_kept_rate:.2%}.",
        f"Best RPC conversion: {strongest_rpc['CMS User']} at {float(strongest_rpc.get('RPC Rate') or 0.0):.2%}.",
        f"Best PTP conversion: {strongest_ptp['CMS User']} at {float(strongest_ptp.get('PTP Rate') or 0.0):.2%}.",
        f"Best KEPT conversion: {strongest_kept['CMS User']} at {float(strongest_kept.get('KEPT Rate') or 0.0):.2%}.",
    ]

    action_rows = []
    if avg_rpc_rate < 0.25:
        action_rows.append(
            ("RPC Improvement Focus", f"RPC is below target at {avg_rpc_rate:.2%}. Coach opening script discipline, contact timing, and rebuttal handling to lift connection-to-RPC conversion.")
        )
    else:
        action_rows.append(
            ("RPC Maintenance Focus", f"RPC is holding at {avg_rpc_rate:.2%}. Keep call-opening quality checks and use top RPC agents as call examples for the rest of the team.")
        )

    if avg_ptp_rate < 0.55:
        action_rows.append(
            ("PTP Coaching Focus", f"PTP is below target at {avg_ptp_rate:.2%}. Reinforce probing for true payment capacity, salary-date alignment, and realistic commitment setting.")
        )
    else:
        action_rows.append(
            ("PTP Coaching Focus", f"PTP is healthy at {avg_ptp_rate:.2%}. Capture the negotiation patterns of {strongest_ptp['CMS User']} and cascade them to lower-performing agents.")
        )

    if avg_kept_rate < 0.60:
        action_rows.append(
            ("KEPT Recovery Focus", f"KEPT is below target at {avg_kept_rate:.2%}. Prioritize confirmation scripting, reminder cadence, and post-PTP follow-up for low-conversion agents.")
        )
    else:
        action_rows.append(
            ("KEPT Recovery Focus", f"KEPT is strong at {avg_kept_rate:.2%}. Standardize the close-and-confirm habits used by {strongest_kept['CMS User']} across the team.")
        )

    if low_count > 0:
        action_rows.append(
            ("Leadership Action", f"Run focused weekly reviews for the {low_count} low performer(s) and compare their calls against the strongest performers in this month.")
        )
    else:
        action_rows.append(
            ("Leadership Action", "No low performers identified this month. Maintain recognition, peer shadowing, and quality review cadence to sustain results.")
        )

    return summary_lines, action_rows


def build_workbook_bytes(export_rows, dates, summary_long):
    output = io.BytesIO()
    month_label = build_month_label(dates)
    daily_sheet_name = f"DAILY PRODUCTIVITY {month_label}"[:31]
    monthly_sheet_name = "MONTHLY PROD - Insights & Exec"
    monthly_summary = build_monthly_summaries(summary_long)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet(daily_sheet_name)
        writer.sheets[daily_sheet_name] = worksheet

        fmt_header = workbook.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#17324D",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )
        fmt_subheader = workbook.add_format(
            {
                "bold": True,
                "font_color": "#17324D",
                "bg_color": "#E8EEF5",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "text_wrap": True,
            }
        )
        fmt_left = workbook.add_format({"border": 1, "bg_color": "#F7FAFC", "valign": "vcenter"})
        fmt_count = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "#,##0"})
        fmt_amount = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "#,##0.00"})
        fmt_pct = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0.00%"})
        fmt_section = workbook.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#244A73",
                "align": "left",
                "valign": "vcenter",
                "border": 1,
            }
        )
        fmt_text = workbook.add_format({"text_wrap": True, "valign": "top", "border": 1})
        fmt_text_header = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#E8EEF5",
                "font_color": "#17324D",
                "border": 1,
            }
        )
        fmt_month_label = workbook.add_format(
            {
                "bold": True,
                "font_color": "#17324D",
                "bg_color": "#DCE6F2",
                "align": "left",
                "valign": "vcenter",
                "border": 1,
            }
        )

        worksheet.freeze_panes(2, 3)
        worksheet.hide_gridlines(2)
        worksheet.set_row(0, 24)
        worksheet.set_row(1, 42)

        for col_idx, label in enumerate(LEFT_COLS):
            worksheet.merge_range(0, col_idx, 1, col_idx, label, fmt_header)

        start_col = 3
        for date_value in dates:
            end_col = start_col + len(DAY_COLS) - 1
            worksheet.merge_range(0, start_col, 0, end_col, date_value, fmt_header)
            for offset, metric in enumerate(DAY_COLS):
                worksheet.write(1, start_col + offset, metric, fmt_subheader)
            start_col = end_col + 1

        worksheet.set_column(0, 0, 18)
        worksheet.set_column(1, 1, 28)
        worksheet.set_column(2, 2, 18)

        col_idx = 3
        for _date in dates:
            for metric in DAY_COLS:
                width = 12 if "%" in metric else 14 if "OB" in metric else 13
                worksheet.set_column(col_idx, col_idx, width)
                col_idx += 1

        for row_idx, row in enumerate(export_rows, start=2):
            worksheet.write(row_idx, 0, row["CMS User"], fmt_left)
            worksheet.write(row_idx, 1, row["Name"], fmt_left)
            worksheet.write(row_idx, 2, row["Placement"], fmt_left)

            col_idx = 3
            for date_value in dates:
                for metric in DAY_COLS:
                    value = row.get((date_value, metric))
                    if "%" in metric:
                        if value is None:
                            worksheet.write_blank(row_idx, col_idx, None, fmt_pct)
                        else:
                            worksheet.write_number(row_idx, col_idx, float(value), fmt_pct)
                    elif "OB" in metric or "Target" in metric:
                        worksheet.write_number(row_idx, col_idx, float(value) if value is not None else 0.0, fmt_amount)
                    else:
                        worksheet.write_number(row_idx, col_idx, int(value or 0), fmt_count)
                    col_idx += 1

        monthly_ws = workbook.add_worksheet(monthly_sheet_name)
        writer.sheets[monthly_sheet_name] = monthly_ws
        monthly_ws.hide_gridlines(2)
        monthly_ws.freeze_panes(1, 1)
        monthly_ws.set_column(0, 0, 24)

        row_labels = [
            ("CMS User", "CMS User", "text"),
            ("Name", "Name", "text"),
            ("Placement", "Placement", "text"),
            ("Total Connected Calls", "Total Connected Calls", "count"),
            ("RPC Count", "RPC Count", "count"),
            ("RPC OB", "RPC OB", "amount"),
            ("RPC Rate", "RPC Rate", "pct"),
            ("PTP Count", "PTP Count", "count"),
            ("PTP OB", "PTP OB", "amount"),
            ("PTP Rate", "PTP Rate", "pct"),
            ("KEPT Count", "KEPT Count", "count"),
            ("KEPT OB", "KEPT OB", "amount"),
            ("KEPT Rate", "KEPT Rate", "pct"),
        ]

        month_blocks = monthly_summary.partition_by(["Month Key", "Month Label"], maintain_order=True)
        start_row = 0
        for block_idx, month_block in enumerate(month_blocks):
            month_rows = month_block.to_dicts()
            block_month_label = month_rows[0]["Month Label"] if month_rows else "Unknown Month"
            agent_columns = [row["CMS User"] for row in month_rows]
            title_end_col = max(6, len(agent_columns))

            monthly_ws.merge_range(start_row, 0, start_row, title_end_col, f"MONTHLY PROD - Insights & Exec ({block_month_label})", fmt_header)
            monthly_ws.merge_range(start_row + 1, 0, start_row + 1, title_end_col, f"Month Label: {block_month_label}", fmt_month_label)

            monthly_ws.write(start_row + 2, 0, "Metric", fmt_subheader)
            for col_idx, cms_user in enumerate(agent_columns, start=1):
                monthly_ws.write(start_row + 2, col_idx, cms_user, fmt_subheader)
                monthly_ws.set_column(col_idx, col_idx, 16)

            for row_idx, (label, key, value_type) in enumerate(row_labels, start=start_row + 3):
                monthly_ws.write(row_idx, 0, label, fmt_left)
                for col_idx, row in enumerate(month_rows, start=1):
                    value = row.get(key)
                    if value_type == "text":
                        monthly_ws.write(row_idx, col_idx, value, fmt_left)
                    elif value_type == "pct":
                        if value is None:
                            monthly_ws.write_blank(row_idx, col_idx, None, fmt_pct)
                        else:
                            monthly_ws.write_number(row_idx, col_idx, float(value), fmt_pct)
                    elif value_type == "amount":
                        monthly_ws.write_number(row_idx, col_idx, float(value or 0.0), fmt_amount)
                    else:
                        monthly_ws.write_number(row_idx, col_idx, int(value or 0), fmt_count)

            top_mask = (pl.col("PTP Rate").fill_null(0.0) >= 0.50) & (pl.col("KEPT Rate").fill_null(0.0) >= 0.60)
            low_mask = pl.col("KEPT Rate").fill_null(0.0) < 0.50
            top_performers = month_block.filter(top_mask).sort(["KEPT Rate", "PTP Rate"], descending=[True, True]).to_dicts()
            low_performers = month_block.filter(low_mask).sort(["KEPT Rate", "PTP Rate"], descending=[False, False]).to_dicts()

            total_agents = max(len(month_rows), 1)
            top_count = len(top_performers)
            low_count = len(low_performers)
            middle_count = max(total_agents - top_count - low_count, 0)
            summary_lines, action_plan_rows = build_month_ai_notes(month_rows, top_count, middle_count, low_count)

            insights_row = start_row + len(row_labels) + 6

            monthly_ws.merge_range(insights_row, 0, insights_row, 2, "Top Performers (Strong PTP + Strong KEPT)", fmt_section)
            monthly_ws.write(insights_row + 1, 0, "CMS User", fmt_subheader)
            monthly_ws.write(insights_row + 1, 1, "PTP Rate", fmt_subheader)
            monthly_ws.write(insights_row + 1, 2, "KEPT Rate", fmt_subheader)
            current_row = insights_row + 2
            for row in top_performers:
                monthly_ws.write(current_row, 0, row["CMS User"], fmt_left)
                monthly_ws.write_number(current_row, 1, float(row.get("PTP Rate") or 0.0), fmt_pct)
                monthly_ws.write_number(current_row, 2, float(row.get("KEPT Rate") or 0.0), fmt_pct)
                current_row += 1
            if not top_performers:
                monthly_ws.merge_range(current_row, 0, current_row, 2, "No agents met the current top-performer thresholds.", fmt_text)
                current_row += 1

            low_start_row = current_row + 2
            monthly_ws.merge_range(low_start_row, 0, low_start_row, 2, "Low Performers (Low KEPT Conversion)", fmt_section)
            monthly_ws.write(low_start_row + 1, 0, "CMS User", fmt_subheader)
            monthly_ws.write(low_start_row + 1, 1, "PTP Rate", fmt_subheader)
            monthly_ws.write(low_start_row + 1, 2, "KEPT Rate", fmt_subheader)
            current_row = low_start_row + 2
            for row in low_performers:
                monthly_ws.write(current_row, 0, row["CMS User"], fmt_left)
                monthly_ws.write_number(current_row, 1, float(row.get("PTP Rate") or 0.0), fmt_pct)
                monthly_ws.write_number(current_row, 2, float(row.get("KEPT Rate") or 0.0), fmt_pct)
                current_row += 1
            if not low_performers:
                monthly_ws.merge_range(current_row, 0, current_row, 2, "No low performers under the current KEPT threshold.", fmt_text)
                current_row += 1

            notes_row = insights_row
            notes_col = 4
            notes_end_col = max(notes_col + 4, notes_col + len(agent_columns))
            monthly_ws.merge_range(notes_row, notes_col, notes_row, notes_end_col, "Executive Summary", fmt_section)
            monthly_ws.write(notes_row + 1, notes_col, "Metric", fmt_text_header)
            monthly_ws.merge_range(notes_row + 1, notes_col + 1, notes_row + 1, notes_end_col, "Value", fmt_text_header)
            monthly_ws.write(notes_row + 2, notes_col, "Coverage Period", fmt_text_header)
            monthly_ws.merge_range(notes_row + 2, notes_col + 1, notes_row + 2, notes_end_col, block_month_label, fmt_text)
            monthly_ws.write(notes_row + 3, notes_col, "Top Performers", fmt_text_header)
            monthly_ws.merge_range(notes_row + 3, notes_col + 1, notes_row + 3, notes_end_col, f"{top_count} agent(s) - {top_count / total_agents:.1%}", fmt_text)
            monthly_ws.write(notes_row + 4, notes_col, "Middle Performers", fmt_text_header)
            monthly_ws.merge_range(notes_row + 4, notes_col + 1, notes_row + 4, notes_end_col, f"{middle_count} agent(s) - {middle_count / total_agents:.1%}", fmt_text)
            monthly_ws.write(notes_row + 5, notes_col, "Low Performers", fmt_text_header)
            monthly_ws.merge_range(notes_row + 5, notes_col + 1, notes_row + 5, notes_end_col, f"{low_count} agent(s) - {low_count / total_agents:.1%}", fmt_text)
            monthly_ws.write(notes_row + 6, notes_col, "Summary", fmt_text_header)
            monthly_ws.merge_range(notes_row + 6, notes_col + 1, notes_row + 10, notes_end_col, "\n".join(summary_lines), fmt_text)

            action_row = notes_row + 12
            monthly_ws.merge_range(action_row, notes_col, action_row, notes_end_col, "Action Plan", fmt_section)
            for offset, (label, text) in enumerate(action_plan_rows, start=1):
                monthly_ws.write(action_row + offset, notes_col, label, fmt_text_header)
                monthly_ws.merge_range(action_row + offset, notes_col + 1, action_row + offset, notes_end_col, text, fmt_text)

            block_end_row = max(current_row, action_row + len(action_plan_rows) + 1)
            start_row = block_end_row + 4
            if block_idx < len(month_blocks) - 1:
                start_row += 1

    output.seek(0)
    return output


def preview_df(uploaded_file, aliases_map, rows=5):
    file_name, file_bytes = read_uploaded_file(uploaded_file)
    return read_table_selected(file_name, file_bytes, aliases_map, preview_rows=rows)


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        activity_files = st.file_uploader(
            "1. Upload DRR Via CSV",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            key="activity_files",
            help="Multiple files allowed. CSV is fastest. Only the 8 required activity columns are read for speed.",
        )
        cms_reference_file = st.file_uploader(
            "2. Upload Agent Reference to be Include in Report",
            type=["xlsx", "xls", "csv"],
            key="cms_reference_file",
            help="Columns: Name, CMS User, Placement",
        )
    with col2:
        target_reference_file = st.file_uploader(
            "3. Upload target reference",
            type=["xlsx", "xls", "csv"],
            key="target_reference_file",
            help="Columns: Placement, Connected Calls Target, RPC Target, RPC OB Target, PTP Target, PTP OB Target, Kept Target, KEPT OB Target",
        )
        old_ic_reference_file = st.file_uploader(
            "4. Upload Masterfile reference",
            type=["xlsx", "xls", "csv"],
            key="old_ic_reference_file",
            help="Columns: Old IC, Placement, Principal",
        )


if activity_files:
    with st.expander("Preview: Activity Logs", expanded=False):
        try:
            st.dataframe(preview_df(activity_files[0], ACTIVITY_REQUIRED_ALIASES), width="stretch")
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")

if cms_reference_file:
    with st.expander("Preview: CMS Reference", expanded=False):
        try:
            st.dataframe(preview_df(cms_reference_file, CMS_REQUIRED_ALIASES), width="stretch")
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")

if target_reference_file:
    with st.expander("Preview: Target Reference", expanded=False):
        try:
            st.dataframe(preview_df(target_reference_file, TARGET_REQUIRED_ALIASES), width="stretch")
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")

if old_ic_reference_file:
    with st.expander("Preview: Old IC Reference", expanded=False):
        try:
            st.dataframe(preview_df(old_ic_reference_file, OLD_IC_REQUIRED_ALIASES), width="stretch")
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")


st.divider()
st.info("Best-speed suggestion: use `.csv` for activity logs whenever possible. CSV is the fastest path; Excel parsing is still the slowest part.")

can_process = all([activity_files, cms_reference_file, target_reference_file, old_ic_reference_file])

if st.button("Process Report", type="primary", disabled=not can_process):
    logs = []

    def log(message):
        logs.append(message)

    try:
        total_start = time.perf_counter()
        with st.spinner("Processing large files with the optimized pipeline..."):
            t0 = time.perf_counter()
            activity_pd = load_activity_logs_fast(activity_files)
            log(f"Activity rows loaded: {activity_pd.height:,}")
            log(f"Load activity files: {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            cms_pd = load_reference_df(cms_reference_file, CMS_REQUIRED_ALIASES)
            target_pd = load_reference_df(target_reference_file, TARGET_REQUIRED_ALIASES)
            old_ic_pd = load_reference_df(old_ic_reference_file, OLD_IC_REQUIRED_ALIASES)
            log(f"Load reference files: {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            activity = prepare_activity_polars(activity_pd)
            cms_ref, target_ref, old_ic_ref = prepare_reference_polars(cms_pd, target_pd, old_ic_pd)
            log(f"Normalize and prepare data: {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            summary_long = build_summary_long(activity, cms_ref, target_ref, old_ic_ref)
            log(f"Aggregate summary: {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            export_rows, dates = build_export_matrix(summary_long)
            output = build_workbook_bytes(export_rows, dates, summary_long)
            log(f"Build XLSX output: {time.perf_counter() - t0:.2f}s")

            total_elapsed = time.perf_counter() - total_start
            log(f"Total elapsed: {total_elapsed:.2f}s")
            st.session_state.agent_prod_result = {
                "message": f"Done. Built {len(export_rows):,} agent row(s) across {len(dates):,} date block(s) in {total_elapsed:.2f}s.",
                "output": output.getvalue(),
                "preview": summary_long.to_pandas(),
                "logs": "\n".join(logs),
            }

    except Exception as exc:
        st.error(f"Error: {exc}")
        log(f"Error: {exc}")
        st.session_state.agent_prod_result = {
            "error": str(exc),
            "logs": "\n".join(logs),
        }

elif not can_process:
    st.info("Upload all 4 required files to enable processing.")

result = st.session_state.agent_prod_result
if result:
    if result.get("error"):
        st.error(f"Error: {result['error']}")
    else:
        st.success(result["message"])
        st.download_button(
            "Download Summary Report",
            data=result["output"],
            file_name="agent_production_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        with st.expander("Preview: Final Summary Data", expanded=True):
            st.dataframe(result["preview"], width="stretch")

    with st.expander("Processing Log", expanded=True):
        st.code(result.get("logs") or "No logs recorded.", language=None)
