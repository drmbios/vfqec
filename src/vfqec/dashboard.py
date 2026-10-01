"""Streamlit UI: queued experiments, live optimization, six-method plots and exports."""

import os

import httpx
import pandas as pd
import streamlit as st

from vfqec.config import preset

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="VFQEC lab", layout="wide")
st.title("Virtual-field QEC lab")
st.caption("Syndrome-only learning • Reproducible experiments • Explicit execution provenance")


def request(method: str, path: str, **kwargs):
    response = httpx.request(method, API + path, timeout=30, **kwargs)
    response.raise_for_status()
    return response


with st.sidebar:
    st.header("New experiment")
    experiment = st.selectbox("Experiment", ["E1", "E2", "E3", "E4"])
    backend = st.selectbox("Backend", ["local", "aer", "bluequbit"])
    optimizer = st.selectbox("Optimizer", ["spsa", "bayes"])
    basis = st.selectbox("Logical memory basis", ["Z", "X"])
    quick = st.checkbox("Small smoke run", value=True)
    st.caption("Dashboard launches one configuration. CLI expands E3/E4 into full sweeps.")
    if st.button("Start run", type="primary"):
        try:
            overrides = dict(backend=backend, optimizer=optimizer, basis=basis)
            if quick:
                overrides.update(shots=128, evaluation_shots=128, rounds=6, iterations=4, cadence=3)
            if experiment != "E2" and basis == "X":
                overrides.update(eps_x=[0], eps_z=[0.12, -0.08, 0.15])
            config = preset(experiment, **overrides)
            row = request("POST", "/runs", json=config.model_dump()).json()
            st.session_state["run_id"] = row["run_id"]
            st.success("Queued " + row["run_id"])
        except (httpx.HTTPError, ValueError) as exc:
            st.error(str(exc))


@st.fragment(run_every="3s")
def monitor():
    try:
        runs = request("GET", "/runs").json()
        if not runs:
            st.info("Start an experiment to populate the ledger.")
            return
        st.subheader("Run ledger")
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
        ids = [row["id"] for row in runs]
        default = (
            ids.index(st.session_state["run_id"]) if st.session_state.get("run_id") in ids else 0
        )
        run_id = st.selectbox("Inspect run", ids, index=default)
        row = request("GET", f"/runs/{run_id}").json()
        st.write("Status:", row["status"])
        events = request("GET", f"/runs/{run_id}/progress").json()
        optimizer_rows = [
            e["payload"]
            for e in events
            if e["payload"]["type"] == "optimizer"
            and e["payload"]["phase"] in ("initial", "candidate")
        ]
        if optimizer_rows:
            frame = pd.DataFrame(optimizer_rows)
            frame["label"] = "round " + frame["calibration_round"].astype(str)
            st.subheader("Live measured syndrome cost")
            st.line_chart(frame, x="shots_used", y="cost", color="label")
        if row["status"] == "complete":
            result = row["result"]
            st.info(result["backend"])
            for image in ("logical-error.png", "fields.png"):
                st.image(request("GET", f"/runs/{run_id}/artifacts/{image}").content)
            st.dataframe(pd.DataFrame(result["analysis"]["fits"]).T, use_container_width=True)
            for note in result["analysis"]["anomalies"]:
                st.warning(note)
            left, right = st.columns(2)
            for column, kind, mime in (
                (left, "pdf", "application/pdf"),
                (right, "html", "text/html"),
            ):
                content = request("GET", f"/runs/{run_id}/artifacts/report.{kind}").content
                column.download_button(
                    f"Download {kind.upper()} report", content, f"vfqec-{run_id}.{kind}", mime=mime
                )
            st.caption("HTML and PDF exports include all figures.")
            with st.expander("Configuration and diagnostics"):
                st.json(result)
        elif row["status"] == "failed":
            st.error(row["result"])
    except httpx.HTTPError as exc:
        st.error(f"API unavailable: {exc}")


monitor()
