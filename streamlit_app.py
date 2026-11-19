
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from datetime import date, timedelta

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

EVENT_COLUMNS = [
    "timestamp", "event_type", "source", "account"
]

if "events" not in st.session_state:
    st.session_state.events = pd.DataFrame(columns=EVENT_COLUMNS)
else:
    # Add account field to records created by an older version.
    if "account" not in st.session_state.events.columns:
        st.session_state.events["account"] = "Unassigned"

    for column in EVENT_COLUMNS:
        if column not in st.session_state.events.columns:
            st.session_state.events[column] = None

    st.session_state.events = st.session_state.events[EVENT_COLUMNS]

if "analysis" not in st.session_state:
    st.session_state.analysis = {}

# Ensure analysis is stored separately for each account.
if not isinstance(st.session_state.analysis, dict):
    st.session_state.analysis = {}


def empty_events():
    return pd.DataFrame(columns=EVENT_COLUMNS)


def get_account_events(account):
    return st.session_state.events[
        st.session_state.events["account"] == account
    ].copy()


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

    filename = uploaded_file.name.lower()

    if "comment" in filename:
        event_type = "comment"
    elif "reaction" in filename or "like" in filename:
        event_type = "reaction"
    elif "friend" in filename:
        event_type = "friend"
    elif (
        "login" in filename
        or "logout" in filename
        or "access" in filename
    ):
        event_type = "login"
    elif "share" in filename:
        event_type = "share"
    else:
        event_type = "other"

    extract_records(data, event_type, records)
    return records


def parse_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)

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

        divergence = (
            abs(recent_value - past_value) /
            (past_value + 1e-6)
        )

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
st.sidebar.subheader("Account Management")

# Known accounts and any additional account names already imported.
known_accounts = ["User 1", "User 2"]

existing_accounts = (
    st.session_state.events["account"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

account_options = known_accounts + [
    name for name in existing_accounts
    if name not in known_accounts
]

selected_account = st.sidebar.selectbox(
    "Select Account",
    account_options,
    key="selected_account"
)

st.sidebar.caption(
    "All dashboard, import, analysis and record views "
    "are filtered by the selected account."
)

st.sidebar.divider()
st.sidebar.caption("TBDS Research Prototype")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Monitoring Dashboard")
    st.write(f"**Selected Account:** {selected_account}")

    account_df = get_account_events(selected_account)
    total = len(account_df)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Activity Events", total)

    if total:
        c2.metric(
            "Activity Types",
            account_df["event_type"].nunique()
        )
        c3.metric(
            "Data Sources",
            account_df["source"].nunique()
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
            account_df["event_type"]
            .value_counts()
            .reset_index()
        )
        chart_data.columns = ["Activity Type", "Count"]

        fig = px.bar(
            chart_data,
            x="Activity Type",
            y="Count",
            title=f"Activity Distribution – {selected_account}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"No activity data imported for {selected_account}. "
            "Open Import Activity Data."
        )


# ---------------- IMPORT DATA ----------------

elif page == "Import Activity Data":

    st.subheader("Import Authorized Account Activity")
    st.write(f"**Importing data for:** {selected_account}")

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
                new_df["account"] = selected_account

                new_df = new_df[EVENT_COLUMNS]

                st.session_state.events = pd.concat(
                    [st.session_state.events, new_df],
                    ignore_index=True
                )

                st.session_state.analysis.pop(selected_account, None)

                st.success(
                    f"Successfully imported {len(new_df)} activity events "
                    f"for {selected_account}."
                )

                st.dataframe(
                    new_df["event_type"].value_counts().rename_axis(
                        "Activity Type"
                    ).reset_index(name="Count"),
                    use_container_width=True
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
        event_date = st.date_input(
            "Activity Date",
            value=date.today()
        )

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
                "source": "Manual entry",
                "account": selected_account
            }])

            st.session_state.events = pd.concat(
                [st.session_state.events, new_event],
                ignore_index=True
            )

            st.session_state.analysis.pop(selected_account, None)

            st.success(
                f"Activity event added to {selected_account}."
            )


# ---------------- TBDS ANALYSIS ----------------

elif page == "TBDS Analysis":

    st.subheader("Temporal Behavioral Divergence Analysis")
    st.write(f"**Selected Account:** {selected_account}")

    df = get_account_events(selected_account)

    if df.empty:
        st.warning(
            f"Import activity data for {selected_account} "
            "before performing analysis."
        )

    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        minimum_date = df["timestamp"].min().date()
        maximum_date = df["timestamp"].max().date()

        midpoint = minimum_date + (
            maximum_date - minimum_date
        ) // 2

        st.write(
            "Select the historical and recent observation periods."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Historical Baseline Period**")

            past_start = st.date_input(
                "Baseline start",
                value=minimum_date,
                key=f"past_start_{selected_account}"
            )

            past_end = st.date_input(
                "Baseline end",
                value=midpoint,
                key=f"past_end_{selected_account}"
            )

        with col2:
            st.markdown("**Recent Behavior Period**")

            recent_start = st.date_input(
                "Recent start",
                value=min(midpoint + timedelta(days=1), maximum_date),
                key=f"recent_start_{selected_account}"
            )

            recent_end = st.date_input(
                "Recent end",
                value=maximum_date,
                key=f"recent_end_{selected_account}"
            )

        st.divider()
        st.subheader("TBDS Configuration")

        threshold = st.slider(
            "Suspicion Threshold",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.05,
            key=f"threshold_{selected_account}"
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
                    key=f"weight_{feature}_{selected_account}"
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

        if st.button(
            "Calculate TBDS",
            type="primary",
            key=f"calculate_{selected_account}"
        ):

            if past_start > past_end:
                st.error(
                    "Historical start date must not exceed end date."
                )

            elif recent_start > recent_end:
                st.error(
                    "Recent start date must not exceed end date."
                )

            elif past_end >= recent_start:
                st.error(
                    "Historical and recent periods must not overlap. "
                    "The recent period must begin after the baseline ends."
                )

            elif weight_total == 0:
                st.error(
                    "At least one feature weight must be greater than zero."
                )

            else:
                past = calculate_features(
                    df, past_start, past_end
                )

                recent = calculate_features(
                    df, recent_start, recent_end
                )

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

                    st.session_state.analysis[selected_account] = {
                        "score": score,
                        "threshold": threshold,
                        "decision": decision,
                        "details": result_df,
                        "past": past,
                        "recent": recent,
                        "historical_start": str(past_start),
                        "historical_end": str(past_end),
                        "recent_start": str(recent_start),
                        "recent_end": str(recent_end)
                    }

        result = st.session_state.analysis.get(selected_account)

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

            st.subheader("Analysis Periods")

            st.write(
                f"Historical: {result['historical_start']} "
                f"to {result['historical_end']}"
            )

            st.write(
                f"Recent: {result['recent_start']} "
                f"to {result['recent_end']}"
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

            st.dataframe(
                display_df,
                use_container_width=True
            )

            fig = px.bar(
                display_df,
                x="Feature",
                y="Divergence",
                title=f"Temporal Feature Divergence – {selected_account}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            csv = display_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download TBDS Analysis Report",
                data=csv,
                file_name=f"{selected_account.replace(' ', '_')}_TBDS_report.csv",
                mime="text/csv"
            )


# ---------------- RECORDS ----------------

elif page == "Monitoring Records":

    st.subheader("Imported Monitoring Records")
    st.write(f"**Selected Account:** {selected_account}")

    account_df = get_account_events(selected_account)

    if account_df.empty:
        st.info(
            f"No activity records available for {selected_account}."
        )

    else:
        st.metric(
            "Account Activity Events",
            len(account_df)
        )

        st.dataframe(
            account_df.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True
        )

        if st.button(
            f"Clear {selected_account} Records",
            type="secondary"
        ):
            st.session_state.events = st.session_state.events[
                st.session_state.events["account"] != selected_account
            ].reset_index(drop=True)

            st.session_state.analysis.pop(selected_account, None)

            st.success(
                f"Records for {selected_account} cleared."
            )
            st.rerun()

    st.divider()

    if st.button("Clear All Accounts' Records"):
        st.session_state.events = empty_events()
        st.session_state.analysis = {}

        st.success("All account records cleared.")
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

        This application analyzes imported or manually entered events.
        It does not yet provide continuous third-party data collection,
        direct Facebook integration, or automatic account compromise confirmation.
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
