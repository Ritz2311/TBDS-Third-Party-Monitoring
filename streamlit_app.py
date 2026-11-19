
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from datetime import date, timedelta
from collections import Counter

st.set_page_config(
    page_title="TBDS Account Monitoring",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ TBDS – Third-Party Account Behavior Monitoring")
st.caption(
    "Research prototype for detecting potential compromised-account "
    "behavior through temporal behavioral divergence."
)

st.info(
    "This prototype analyzes data that you are authorized to provide. "
    "It does not directly access Facebook or monitor accounts in the background."
)

# ---------------- SESSION STATE ----------------

if "events" not in st.session_state:
    st.session_state.events = pd.DataFrame(
        columns=["timestamp", "event_type", "source"]
    )

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ---------------- DATA PROCESSING ----------------

def find_timestamp(record):
    for key in ["timestamp", "creation_timestamp", "time"]:
        if key in record:
            try:
                return pd.to_datetime(
                    record[key], unit="s", utc=True
                ).tz_convert("Asia/Kolkata").tz_localize(None)
            except Exception:
                try:
                    return pd.to_datetime(record[key])
                except Exception:
                    pass
    return None


def extract_records(obj, event_type, output):
    """Recursively find timestamped records in exported JSON."""
    if isinstance(obj, list):
        for item in obj:
            extract_records(item, event_type, output)

    elif isinstance(obj, dict):
        timestamp = find_timestamp(obj)

        if timestamp is not None:
            output.append({
                "timestamp": timestamp,
                "event_type": event_type,
                "source": "Facebook JSON import"
            })

        for value in obj.values():
            if isinstance(value, (dict, list)):
                extract_records(value, event_type, output)


def parse_json_file(uploaded_file):
    data = json.load(uploaded_file)
    records = []

    # Recognize common Facebook Download Your Information filenames.
    filename = uploaded_file.name.lower()

    if "comment" in filename:
        event_type = "comment"
    elif "reaction" in filename or "like" in filename:
        event_type = "reaction"
    elif "friend" in filename:
        event_type = "friend"
    elif "login" in filename or "logout" in filename or "access" in filename:
        event_type = "login"
    elif "share" in filename:
        event_type = "share"
    else:
        event_type = "other"

    extract_records(data, event_type, records)
    return records


def parse_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Expected columns: timestamp, event_type
    if "timestamp" not in df.columns or "event_type" not in df.columns:
        raise ValueError(
            "CSV must contain columns named timestamp and event_type."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "event_type"])
    df["event_type"] = df["event_type"].astype(str).str.lower()
    df["source"] = "CSV import"

    return df[["timestamp", "event_type", "source"]].to_dict("records")



def calculate_features(df, start_date, end_date):
    selected = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date)
    ]

    total_days = (end_date - start_date).days + 1
    total_days = max(total_days, 1)

    counts = selected["event_type"].value_counts().to_dict()

    return {
        "comments": counts.get("comment", 0) / total_days,
        "reactions": counts.get("reaction", 0) / total_days,
        "shares": counts.get("share", 0) / total_days,
        "friend_activity": counts.get("friend", 0) / total_days,
        "login_activity": counts.get("login", 0) / total_days
    }



def calculate_tbds(past, recent, weights):
    rows = []
    score = 0.0

    for feature in past:
        past_value = float(past[feature])
        recent_value = float(recent[feature])

        divergence = abs(recent_value - past_value) / (past_value + 1e-6)
        contribution = weights[feature] * divergence
        score += contribution

        rows.append({
            "Feature": feature.replace("_", " ").title(),
            "Historical": past_value,
            "Recent": recent_value,
            "Divergence": divergence,
            "Weighted Contribution": contribution
        })

    return score, pd.DataFrame(rows)


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Import Activity Data",
        "TBDS Analysis",
        "Monitoring Records",
        "Research Information"
    ]
)

st.sidebar.divider()
st.sidebar.caption("TBDS Research Prototype")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Monitoring Dashboard")

    total = len(st.session_state.events)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Activity Events", total)

    if total:
        c2.metric(
            "Activity Types",
            st.session_state.events["event_type"].nunique()
        )
        c3.metric(
            "Data Sources",
            st.session_state.events["source"].nunique()
        )
    else:
        c2.metric("Activity Types", 0)
        c3.metric("Data Sources", 0)

    st.divider()

    st.subheader("System Overview")

    st.write(
        """
        The system compares two time periods of the same account:

        1. **Historical Baseline:** Previously observed account behavior.
        2. **Recent Behavior:** Activity in the selected recent period.
        3. **Feature Extraction:** Activity counts are extracted for supported event types.
        4. **Temporal Divergence:** Differences between historical and recent features are calculated.
        5. **TBDS Evaluation:** Weighted divergence is compared with a configurable threshold.
        """
    )

    if total:
        chart_data = (
            st.session_state.events["event_type"]
            .value_counts()
            .reset_index()
        )
        chart_data.columns = ["Activity Type", "Count"]

        fig = px.bar(
            chart_data,
            x="Activity Type",
            y="Count",
            title="Imported Activity Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No activity data imported yet. Open Import Activity Data.")


# ---------------- IMPORT DATA ----------------

elif page == "Import Activity Data":

    st.subheader("Import Authorized Account Activity")

    st.write(
        "Upload Facebook Download Your Information JSON files "
        "or a CSV file containing timestamped activity events."
    )

    st.caption(
        "Supported JSON examples: comments, reactions, friends, "
        "and login activity exports."
    )

    uploaded_files = st.file_uploader(
        "Choose JSON or CSV files",
        type=["json", "csv"],
        accept_multiple_files=True
    )

    if st.button("Import Selected Files", type="primary"):

        if not uploaded_files:
            st.error("Please select at least one file.")
        else:
            all_records = []
            errors = []

            for file in uploaded_files:
                try:
                    if file.name.lower().endswith(".json"):
                        records = parse_json_file(file)
                    else:
                        records = parse_csv_file(file)

                    all_records.extend(records)

                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")

            if all_records:
                new_df = pd.DataFrame(all_records)
                new_df["timestamp"] = pd.to_datetime(
                    new_df["timestamp"], errors="coerce"
                )
                new_df = new_df.dropna(subset=["timestamp"])

                st.session_state.events = pd.concat(
                    [st.session_state.events, new_df],
                    ignore_index=True
                )

                st.session_state.analysis = None

                st.success(
                    f"Successfully imported {len(new_df)} activity events."
                )

            if errors:
                for error in errors:
                    st.warning(error)

            if not all_records and not errors:
                st.warning(
                    "No timestamped activity records were found in the files."
                )

    st.divider()

    st.subheader("Manual Activity Entry")

    with st.form("manual_event_form"):
        event_date = st.date_input("Activity Date", value=date.today())

        event_type = st.selectbox(
            "Activity Type",
            [
                "comment",
                "reaction",
                "share",
                "friend",
                "login"
            ]
        )

        submitted = st.form_submit_button("Add Activity Event")

        if submitted:
            new_event = pd.DataFrame([{
                "timestamp": pd.Timestamp(event_date),
                "event_type": event_type,
                "source": "Manual entry"
            }])

            st.session_state.events = pd.concat(
                [st.session_state.events, new_event],
                ignore_index=True
            )

            st.session_state.analysis = None
            st.success("Activity event added.")


# ---------------- TBDS ANALYSIS ----------------

elif page == "TBDS Analysis":

    st.subheader("Temporal Behavioral Divergence Analysis")

    if st.session_state.events.empty:
        st.warning("Import activity data before performing analysis.")
    else:
        df = st.session_state.events.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        minimum_date = df["timestamp"].min().date()
        maximum_date = df["timestamp"].max().date()

        st.write("Select the historical and recent observation periods.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Historical Baseline Period**")
            past_start = st.date_input(
                "Baseline start",
                value=minimum_date,
                key="past_start"
            )
            past_end = st.date_input(
                "Baseline end",
                value=minimum_date + (
                    maximum_date - minimum_date
                ) // 2,
                key="past_end"
            )

        with col2:
            st.markdown("**Recent Behavior Period**")
            recent_start = st.date_input(
                "Recent start",
                value=minimum_date + (
                    maximum_date - minimum_date
                ) // 2 + timedelta(days=1),
                key="recent_start"
            )
            recent_end = st.date_input(
                "Recent end",
                value=maximum_date,
                key="recent_end"
            )

        st.divider()

        st.subheader("TBDS Configuration")

        threshold = st.slider(
            "Suspicion Threshold",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.05
        )

        st.write("Feature Weights")

        feature_names = [
            "comments",
            "reactions",
            "shares",
            "friend_activity",
            "login_activity"
        ]

        default_weights = {
            "comments": 0.20,
            "reactions": 0.20,
            "shares": 0.20,
            "friend_activity": 0.20,
            "login_activity": 0.20
        }

        weights = {}
        weight_cols = st.columns(5)

        for i, feature in enumerate(feature_names):
            with weight_cols[i]:
                weights[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weights[feature],
                    step=0.05,
                    key=f"weight_{feature}"
                )

        weight_total = sum(weights.values())

        if weight_total > 0:
            weights = {
                key: value / weight_total
                for key, value in weights.items()
            }

        st.caption(
            "Weights are normalized automatically so their total equals 1."
        )

        if st.button("Calculate TBDS", type="primary"):

            if past_start > past_end:
                st.error("Historical start date must not exceed end date.")

            elif recent_start > recent_end:
                st.error("Recent start date must not exceed end date.")

            elif past_end >= recent_start:
                st.error(
                    "Historical and recent periods must not overlap. "
                    "The recent period must begin after the baseline ends."
                )

            elif weight_total == 0:
                st.error("At least one feature weight must be greater than zero.")

            else:
                past = calculate_features(df, past_start, past_end)
                recent = calculate_features(df, recent_start, recent_end)

                if sum(past.values()) == 0:
                    st.error(
                        "No supported activity was found in the historical "
                        "period. A valid baseline is required."
                    )

                elif sum(recent.values()) == 0:
                    st.error(
                        "No supported activity was found in the recent "
                        "period. A valid recent observation period is required."
                    )

                else:
                    score, result_df = calculate_tbds(
                        past, recent, weights
                    )

                    decision = (
                        "Potentially Suspicious"
                        if score >= threshold
                        else "Normal"
                    )

                    st.session_state.analysis = {
                        "score": score,
                        "threshold": threshold,
                        "decision": decision,
                        "details": result_df,
                        "past": past,
                        "recent": recent
                    }

        result = st.session_state.analysis

        if result is not None:

            st.divider()
            st.subheader("TBDS Analysis Result")

            a, b, c = st.columns(3)

            a.metric("TBDS Score", f"{result['score']:.4f}")
            b.metric("Threshold", f"{result['threshold']:.2f}")

            if result["decision"] == "Normal":
                c.success("Normal")
            else:
                c.error("Potentially Suspicious")

            st.write(
                f"**System Decision:** {result['decision']}"
            )

            st.caption(
                "A suspicious result indicates behavioral deviation. "
                "It does not independently prove account compromise."
            )

            st.subheader("Feature-wise Divergence")

            display_df = result["details"].copy()

            for column in [
                "Historical",
                "Recent",
                "Divergence",
                "Weighted Contribution"
            ]:
                display_df[column] = display_df[column].round(4)

            st.dataframe(display_df, use_container_width=True)

            fig = px.bar(
                display_df,
                x="Feature",
                y="Divergence",
                title="Temporal Feature Divergence"
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = display_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download TBDS Analysis Report",
                data=csv,
                file_name="TBDS_analysis_report.csv",
                mime="text/csv"
            )


# ---------------- RECORDS ----------------

elif page == "Monitoring Records":

    st.subheader("Imported Monitoring Records")

    if st.session_state.events.empty:
        st.info("No activity records available.")
    else:
        st.dataframe(
            st.session_state.events.sort_values(
                "timestamp", ascending=False
            ),
            use_container_width=True
        )

        if st.button("Clear All Imported Records"):
            st.session_state.events = pd.DataFrame(
                columns=["timestamp", "event_type", "source"]
            )
            st.session_state.analysis = None
            st.success("Records cleared.")
            st.rerun()


# ---------------- RESEARCH INFORMATION ----------------

elif page == "Research Information":

    st.subheader("About the TBDS Research System")

    st.write(
        """
        **Research Objective**

        To identify potential compromised accounts by measuring temporal
        deviations from the account owner's historical behavioral patterns.

        **Methodology**

        - Authorized activity data collection
        - Historical and recent time-window separation
        - Behavioral feature extraction
        - Temporal feature divergence calculation
        - Weighted TBDS score computation
        - Threshold-based anomaly identification

        **TBDS Formula**

        TBDS = Sum of (Feature Weight × Feature Divergence)

        **Feature Divergence**

        Absolute difference between recent and historical activity,
        divided by historical activity plus a small epsilon.

        **Important Limitation**

        This application currently analyzes imported or manually entered
        events. It does not yet provide continuous third-party data
        collection, direct Facebook integration, or automatic account
        compromise confirmation.
        """
    )

    st.info(
        "Future integration with authorized third-party data sources "
        "can be developed separately, subject to platform permissions, "
        "privacy requirements, and API availability."
    )

st.divider()

st.caption(
    "TBDS Research Prototype | Behavioral Anomaly Monitoring | "
    "No machine-learning classifier is used."
)

