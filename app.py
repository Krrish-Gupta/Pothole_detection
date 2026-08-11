import streamlit as st
from PIL import Image
from pathlib import Path
from io import BytesIO

from pipeline import run_pipeline
from database import get_all_results, get_severity_summary, get_daily_counts, clear_results
from severity import get_severity_color_hex
from charts   import severity_donut, daily_trend, repair_priority_bar, confidence_histogram
from config   import APP_TITLE, MAX_UPLOAD_MB

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🛣️")

# ── Custom CSS: card-style KPIs, spacing, subtle polish ────────────────────────
st.markdown("""
<style>
    /* Tighter top padding, page feels less boxed-in */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(145deg, #161B24, #1B212C);
        border: 1px solid #2A2F3A;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }
    .kpi-label {
        font-size: 13px;
        color: #9AA3AF;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #E6E8EB;
        line-height: 1.1;
    }
    .kpi-value.accent   { color: #5B8DEF; }
    .kpi-value.low      { color: #3ECF8E; }
    .kpi-value.medium   { color: #F5A623; }
    .kpi-value.high     { color: #F04747; }

    /* Section header spacing */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #E6E8EB;
        margin: 6px 0 14px 0;
    }
</style>
""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, css_class: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🛣️  Navigation")
page = st.sidebar.radio("", ["📤 Detection", "📊 Dashboard", "🗂️ History"])
page = page.split(" ", 1)[1]   # strip emoji for logic below
st.sidebar.markdown("---")
st.sidebar.caption("AI-Driven Urban Road Defect Detection · MIET 2026")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Detection":
    st.title("Road defect detection")
    st.caption("Upload a road image to detect and classify potholes.")

    upload = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help=f"Max file size: {MAX_UPLOAD_MB} MB"
    )

    if upload:
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large. Max size is {MAX_UPLOAD_MB} MB.")
            st.stop()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="section-title">Input image</div>', unsafe_allow_html=True)
            st.image(upload, width="stretch")

        if st.button("Run detection", type="primary"):
            with st.spinner("Running inference..."):
                upload.seek(0)
                try:
                    result = run_pipeline(upload, filename=upload.name)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    st.stop()

            with col2:
                st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)
                ann_img = Image.open(result["annotated_path"])
                st.image(ann_img, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── KPI row (custom cards) ──────────────────────────────────────────
            sev = result["severity"]
            sev_class = {"Low": "low", "Medium": "medium", "High": "high"}.get(sev, "")

            k1, k2, k3, k4 = st.columns(4)
            with k1: kpi_card("Defects found", str(result["defect_count"]))
            with k2: kpi_card("Max confidence", f"{result['confidence']:.0%}", "accent")
            with k3: kpi_card("Inference time", f"{result['inference_ms']} ms")
            with k4: kpi_card("Severity", sev, sev_class)

            st.markdown("<br>", unsafe_allow_html=True)

            if result["defect_count"] == 0:
                st.success("No potholes detected in this image.")
            elif sev == "High":
                st.error("High-severity defect detected — flagged for priority repair.")
            elif sev == "Medium":
                st.warning("Medium-severity defect — schedule inspection.")
            elif sev == "Low":
                st.info("Low-severity defect — monitor over time.")

            st.caption(f"Result saved to database (ID #{result['db_id']})")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("Analytics dashboard")

    df      = get_all_results()
    summary = get_severity_summary()
    daily   = get_daily_counts()

    if df.empty:
        st.info("No detections yet. Run detection on some images first.")
        st.stop()

    total_images  = len(df)
    total_defects = int(df["defect_count"].sum())
    avg_defects   = round(df["defect_count"].mean(), 1)
    high_pct      = round(summary.get("High", 0) / max(total_images, 1) * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total images processed", str(total_images))
    with k2: kpi_card("Total defects detected", str(total_defects), "accent")
    with k3: kpi_card("Avg defects / image", str(avg_defects))
    with k4: kpi_card("High-severity images", f"{high_pct}%", "high" if high_pct > 0 else "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart row 1 ───────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title">Severity distribution</div>', unsafe_allow_html=True)
            st.image(severity_donut(summary), width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">Repair priority breakdown</div>', unsafe_allow_html=True)
            st.image(repair_priority_bar(summary), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart row 2 ───────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown('<div class="section-title">Detection trend</div>', unsafe_allow_html=True)
            st.image(daily_trend(daily), width="stretch")
    with c4:
        with st.container(border=True):
            st.markdown('<div class="section-title">Confidence distribution</div>', unsafe_allow_html=True)
            st.image(confidence_histogram(df), width="stretch")

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Export all results as CSV", csv,
                       "detections.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    st.title("Detection history")

    df = get_all_results()

    if df.empty:
        st.info("No detections logged yet.")
        st.stop()

    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            sev_filter = st.multiselect(
                "Filter by severity",
                options=["High", "Medium", "Low", "None", "Ignore"],
                default=["High", "Medium", "Low", "None", "Ignore"]
            )
        with f2:
            if not df["timestamp"].isna().all():
                date_range = st.date_input(
                    "Date range",
                    value=(df["timestamp"].min().date(), df["timestamp"].max().date())
                )
            else:
                date_range = None

    filtered = df[df["severity"].isin(sev_filter)]
    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start) &
            (filtered["timestamp"].dt.date <= end)
        ]

    st.markdown(f"Showing **{len(filtered)}** of {len(df)} records")

    display_cols = ["id", "image_name", "defect_count",
                    "severity", "confidence", "timestamp"]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️  Clear all records", type="secondary"):
            clear_results()
            st.rerun()