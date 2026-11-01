"""
AgriSense Agent - Autonomous Agricultural Intelligence & Crop Yield Prediction System.
Combines Supervised Machine Learning Crop-Yield Predictor with Local RAG Knowledge Base
and Live Agricultural Search Fallback.
"""

import json
from datetime import datetime
import pandas as pd
import streamlit as st

# Local module imports
from src.config import APIConfig, ARTIFACTS_DIR, DATA_DIR, KNOWLEDGE_DIR
from src.yield_predictor import YieldPredictor, CropYieldInput
from src.weather_fetcher import AnnualWeather, WeatherFetchError, WeatherFetcher
from src.agent_router import AgriSenseAgentRouter
from src.data_audit import run_data_audit
from src.train_model import train_yield_model

# Page Configuration
st.set_page_config(
    page_title="AgriSense Agent | Crop Yield AI & Agronomic Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Mode Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #062316 0%, #0d3824 50%, #124e33 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(6, 35, 22, 0.4);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #cbd5e1 !important;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399 !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 4px;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .citation-card {
        background: #13231c;
        border: 1px solid #1e3a2f;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        color: #e2e8f0 !important;
    }
    .citation-card * {
        color: #e2e8f0 !important;
    }
    .citation-card a {
        color: #34d399 !important;
        font-weight: 600;
        text-decoration: none;
    }
    .citation-card a:hover {
        text-decoration: underline;
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.4);
        transform: translateY(-1px);
    }
    div[data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def get_annual_weather(area: str, year: int, location: str) -> AnnualWeather:
    """Cache annual climate estimates while the user edits prediction inputs."""
    return WeatherFetcher().fetch_annual_weather(area, year, location or None)




# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "agent_router" not in st.session_state:
    st.session_state.agent_router = AgriSenseAgentRouter()
if "yield_predictor" not in st.session_state:
    st.session_state.yield_predictor = YieldPredictor()


# Top Navigation / Hero Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; color: #ffffff;">🌾 AgriSense Agent</h1>
            <p style="margin: 6px 0 12px 0; font-size: 1.05rem; opacity: 0.9;">
                Low-Tension Autonomous Agricultural Intelligence & Crop Yield Predictor
            </p>
            <div>
                <span class="badge-pill">🌲 Scikit-Learn RandomForest</span>
                <span class="badge-pill">📚 Crop Yield Local RAG</span>
                <span class="badge-pill">🌐 Live Web Search Fallback</span>
                <span class="badge-pill">🛡️ Deterministic Guardrails</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Citations & Telemetry
with st.sidebar:
    st.header("📚 Active Sources & Citations")
    st.caption("Verified agronomic literature and live web evidence.")

    if st.session_state.last_citations:
        for i, c in enumerate(st.session_state.last_citations, 1):
            with st.container():
                st.markdown(f"""
                <div class="citation-card">
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">#{i}. {c.get('title', 'Agronomic Citation')}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0;">
                        🏛️ <b>{c.get('publisher', 'Agri Authority')}</b> | 📅 {c.get('date', 'Recent')}
                    </div>
                    <div style="font-size: 0.82rem; margin-top: 6px;">
                        <a href="{c.get('url', '#')}" target="_blank" style="color: #34d399; text-decoration: none; font-weight: 500;">
                            🔗 View Active Reference &rarr;
                        </a>
                    </div>
                </div>

                """, unsafe_allow_html=True)
    else:
        st.info("No active citations yet. Run a yield prediction or ask an agronomic question to see citations.")

    st.markdown("---")
    st.subheader("⚙️ System Status")
    predictor = st.session_state.yield_predictor
    st.markdown(f"**ML Model Status:** {'🟢 Ready (Fitted Pipeline)' if predictor.is_ready() else '🟡 Standby (Baseline Active)'}")
    
    docs_count = len(list(KNOWLEDGE_DIR.glob("*.txt")))
    st.markdown(f"**Local Yield Knowledge Base:** 🟢 {docs_count} Curated Docs")
    
    tavily_ready = bool(APIConfig.get_tavily_key())
    st.markdown(f"**Live Search Mode:** {'🟢 Tavily API' if tavily_ready else '🟢 DuckDuckGo (Free / 0 Key)'}")


# Primary Application Tabs
tabs = st.tabs([
    "🌾 Crop Yield Predictor",
    "💬 Agricultural Assistant (RAG & Web Search)",
    "🔑 API Keys & Free Services",
    "📊 Model & Data Diagnostics"
])


# ====================================================================
# TAB 1: CROP YIELD PREDICTION FORM
# ====================================================================
with tabs[0]:
    st.markdown("### 🌾 Crop Yield Prediction Form (Supervised Machine Learning)")
    st.markdown(
        "Forecast crop yield in **tonnes/hectare** and **hg/hectare** using "
        "annual climate estimates for the selected location and harvest year."
    )

    predictor = st.session_state.yield_predictor
    supported_areas = predictor.get_supported_areas()
    supported_crops = predictor.get_supported_crops()

    location_col, area_col, year_col = st.columns([1.3, 1.3, 1])
    with area_col:
        selected_area = st.selectbox(
            "🌍 Country / Region",
            options=supported_areas,
            index=supported_areas.index("India") if "India" in supported_areas else 0,
            key="yield_area",
            help="Select the country used by the crop-yield model.",
        )
    with location_col:
        selected_location = st.text_input(
            "📍 Specific location (optional)",
            key="yield_location",
            placeholder="e.g. Coimbatore",
            help="Enter a city or town for more local climate estimates. Leave blank to use the country-level location.",
        ).strip()
    with year_col:
        selected_year = st.slider(
            "📅 Harvest Year",
            min_value=1950,
            max_value=2049,
            value=min(max(datetime.now().year, 1950), 2049),
            step=1,
            key="yield_year",
            help="Open-Meteo climate-model data is available through 2049.",
        )

    weather_selection = (selected_area, selected_location, selected_year)
    if st.session_state.get("weather_selection") != weather_selection:
        st.session_state.weather_selection = weather_selection
        st.session_state.weather_data = None
        st.session_state.weather_error = None
        st.session_state.weather_override = False
        try:
            st.session_state.weather_data = get_annual_weather(
                selected_area, selected_year, selected_location
            )
        except WeatherFetchError as exc:
            st.session_state.weather_error = str(exc)

    if st.session_state.weather_data is None and st.button(
        "Retry automatic weather lookup", key="retry_weather_lookup"
    ):
        try:
            st.session_state.weather_data = get_annual_weather(
                selected_area, selected_year, selected_location
            )
            st.session_state.weather_error = None
        except WeatherFetchError as exc:
            st.session_state.weather_error = str(exc)

    weather_data = st.session_state.weather_data
    if weather_data is None:
        st.error(f"Automatic rainfall and temperature lookup failed: {st.session_state.weather_error}")
        st.info("Choose a valid location or try again when Open-Meteo is reachable. No weather defaults are substituted.")
    else:
        st.markdown("#### 🌦️ Automatic annual climate inputs")
        weather_col1, weather_col2 = st.columns(2)
        weather_col1.metric("Estimated annual rainfall", f"{weather_data.rainfall_mm:,.1f} mm")
        weather_col2.metric("Estimated mean temperature", f"{weather_data.temperature_celsius:,.1f} °C")
        st.caption(
            f"Location: {weather_data.resolved_location} ({weather_data.latitude:.3f}, "
            f"{weather_data.longitude:.3f}) · Year: {selected_year}. "
            "These are climate-model estimates, not day-specific weather forecasts."
        )

        manual_weather_override = st.checkbox(
            "Manually override automatic rainfall and temperature",
            key="weather_override",
        )
        if manual_weather_override:
            manual_key = f"{selected_area}_{selected_location}_{selected_year}"
            manual_col1, manual_col2 = st.columns(2)
            with manual_col1:
                rainfall_input = st.number_input(
                    "🌧️ Average Annual Rainfall (mm/year)",
                    min_value=0.0,
                    max_value=8000.0,
                    value=float(weather_data.rainfall_mm),
                    step=25.0,
                    key=f"manual_rainfall_{manual_key}",
                )
            with manual_col2:
                temperature_input = st.number_input(
                    "🌡️ Average Temperature (°C)",
                    min_value=-10.0,
                    max_value=55.0,
                    value=float(weather_data.temperature_celsius),
                    step=0.2,
                    key=f"manual_temperature_{manual_key}",
                )
        else:
            rainfall_input = weather_data.rainfall_mm
            temperature_input = weather_data.temperature_celsius

        with st.form("yield_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                selected_crop = st.selectbox(
                    "🌱 Crop Species (Item)",
                    options=supported_crops,
                    index=supported_crops.index("Maize") if "Maize" in supported_crops else 0,
                    help="Target crop commodity",
                )
                pesticides_input = st.number_input(
                    "🛡️ Pesticides Used (metric tonnes)",
                    min_value=0.0,
                    max_value=500000.0,
                    value=45000.0,
                    step=500.0,
                    help="Total national/regional pesticide application",
                )
            with col2:
                st.write("Weather values are automatic unless the override is enabled above.")

            submit_pred = st.form_submit_button("🚀 Forecast Crop Yield", use_container_width=True)

        if submit_pred:
            with st.spinner("Executing RandomForestRegressor ML Pipeline..."):
                pred_input = CropYieldInput(
                    area=selected_area,
                    crop=selected_crop,
                    year=selected_year,
                    rainfall_mm=rainfall_input,
                    pesticides_tonnes=pesticides_input,
                    temperature_celsius=temperature_input
                )
                result = predictor.predict(pred_input)

                st.markdown("#### 🎯 Prediction Results")
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric(
                        label="Yield (Tonnes per Hectare)",
                        value=f"{result.yield_tonnes_per_ha:,.2f} t/ha",
                        help="1 metric tonne = 10,000 hectograms"
                    )
                with mcol2:
                    st.metric(
                        label="Yield (Hectograms per Hectare)",
                        value=f"{result.yield_hg_per_ha:,.1f} hg/ha",
                        help="FAO standard statistical unit"
                    )
                with mcol3:
                    st.metric(
                        label="Confidence Tier",
                        value=result.confidence_tier.split("(")[0].strip()
                    )

                st.caption(
                    "Weather inputs: manually overridden"
                    if manual_weather_override
                    else "Weather inputs: automatic climate-model estimate"
                )
                st.markdown("##### 📋 Agronomic Notes & Limiting Factors:")
                if result.agronomic_notes:
                    for note in result.agronomic_notes:
                        st.warning(note)
                else:
                    st.success("Environmental and input conditions fall comfortably within optimal biological growth bands.")

                st.session_state.last_citations = [
                    {
                        "title": f"Crop Yield Regressor - {result.crop} ({result.area})",
                        "publisher": "AgriSense Supervised ML Pipeline",
                        "url": "https://www.fao.org/faostat/en/#data/QCL",
                        "date": str(selected_year),
                        "type": "Trained ML Model"
                    },
                    {
                        "title": f"Climate estimates for {weather_data.resolved_location} ({selected_year})",
                        "publisher": "Open-Meteo Climate API",
                        "url": "https://open-meteo.com/en/docs/climate-api",
                        "date": str(selected_year),
                        "type": "Climate-model estimate"
                    },
                    {
                        "title": "AgriSense Local Crop-Yield Training Dataset",
                        "publisher": "World Bank rows and embedded benchmark estimates",
                        "url": "https://api.worldbank.org/v2/indicator/AG.YLD.CREL.KG",
                        "date": "Current dataset",
                        "type": "Model training data"
                    }
                ]


# ====================================================================
# TAB 2: AGRICULTURAL ASSISTANT (RAG + LIVE SEARCH)
# ====================================================================
with tabs[1]:
    st.markdown("### 💬 AgriSense Agricultural Assistant")
    st.caption(
        "Ask about **crop yield prediction** (answered from the curated local knowledge base) "
        "or **other agricultural aspects** like pests, diseases, fertilizers, and market insights "
        "(dynamically researched from live online agricultural sources)."
    )

    # Quick Prompts
    st.markdown("**Quick Prompts:**")
    qcols = st.columns(4)
    q1 = qcols[0].button("🚀 Fetch & Train Model", key="qp_train")
    q2 = qcols[1].button("Predict Rice in India", key="qp_pred")
    q3 = qcols[2].button("Maize Yield Benchmarks", key="qp1")
    q4 = qcols[3].button("Fall Armyworm Pest Control", key="qp3")

    prompt_to_send = None
    if q1:
        prompt_to_send = "Fetch latest datasets from trusted sources and train the model"
    elif q2:
        prompt_to_send = "Predict crop yield for rice in India with 1100 mm rainfall and 25C temperature"
    elif q3:
        prompt_to_send = "What are the global yield benchmarks for Maize and what causes yield loss?"
    elif q4:
        prompt_to_send = "What are the best integrated pest management practices for Fall Armyworm in maize?"


    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("mode"):
                st.caption(f"Source: {msg['mode']}")

    # Chat Input Box
    user_input = st.chat_input("Ask an agricultural or crop yield question...")
    if prompt_to_send:
        user_input = prompt_to_send

    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing through AgriSense Agent Router..."):
                response_data = st.session_state.agent_router.process_query(user_input)
                st.markdown(response_data["response"])
                st.caption(f"Mode: {response_data['source_mode']} | Intent: `{response_data['intent']}`")

                # Store citations
                st.session_state.last_citations = response_data.get("citations", [])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "mode": response_data["source_mode"]
                })


# ====================================================================
# TAB 3: DEDICATED API KEYS & FREE SERVICES MANAGEMENT
# ====================================================================
with tabs[2]:
    st.markdown("### 🔑 Dedicated API Keys & Free Services Configuration")
    st.info(
        "💡 **All services used in this project have 100% free tiers.** You can enter your free API keys below "
        "or in the `.env` file whenever you are ready. The system already works offline and has free fallbacks "
        "even before you enter any keys!"
    )

    status = APIConfig.get_status()

    st.markdown("#### 1. Tavily Web Search API (Free Tier: 1,000 queries/month)")
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        tavily_input = st.text_input(
            "TAVILY_API_KEY",
            value=APIConfig.get_tavily_key(),
            type="password",
            help="Free key from tavily.com for deep agricultural web search"
        )
    with tcol2:
        st.write("")
        st.write("")
        if st.button("Save Tavily Key", key="btn_save_tavily"):
            APIConfig.update_key("TAVILY_API_KEY", tavily_input)
            st.success("Tavily key saved successfully!")

    st.caption("Get free Tavily key (No credit card needed): [https://tavily.com](https://tavily.com)")

    st.markdown("---")
    st.markdown("#### 2. Google Gemini API (Free Tier via Google AI Studio)")
    gcol1, gcol2 = st.columns([3, 1])
    with gcol1:
        gemini_input = st.text_input(
            "GEMINI_API_KEY",
            value=APIConfig.get_gemini_key(),
            type="password",
            help="Free key from aistudio.google.com for cloud LLM reasoning"
        )
    with gcol2:
        st.write("")
        st.write("")
        if st.button("Save Gemini Key", key="btn_save_gemini"):
            APIConfig.update_key("GEMINI_API_KEY", gemini_input)
            st.success("Gemini key saved successfully!")

    st.caption("Get free Google Gemini key: [https://aistudio.google.com](https://aistudio.google.com)")

    st.markdown("---")
    st.markdown("#### 3. Groq Cloud API (Free Ultra-Fast Inference Tier)")
    grcol1, grcol2 = st.columns([3, 1])
    with grcol1:
        groq_input = st.text_input(
            "GROQ_API_KEY",
            value=APIConfig.get_groq_key(),
            type="password",
            help="Free fast inference key from console.groq.com"
        )
    with grcol2:
        st.write("")
        st.write("")
        if st.button("Save Groq Key", key="btn_save_groq"):
            APIConfig.update_key("GROQ_API_KEY", groq_input)
            st.success("Groq key saved successfully!")

    st.caption("Get free Groq key: [https://console.groq.com](https://console.groq.com)")

    st.markdown("---")
    st.markdown("#### 4. DuckDuckGo Free Search (Always Ready)")
    st.success("✅ **DuckDuckGo Free Web Search** is automatically configured as an active fallback. Requires **0 keys and 0 cost**.")


# ====================================================================
# TAB 4: MODEL & DATA DIAGNOSTICS (AUTOMATED TRAINING & DATASET FETCHING)
# ====================================================================
with tabs[3]:
    st.markdown("### 📊 Automated Dataset Acquisition & Model Retraining")
    st.caption("Acquire World Bank yield data, merge it with embedded crop benchmark estimates, and retrain the model. Inspect the exact local CSV used for training below.")

    tcol_a, tcol_b = st.columns(2)
    
    with tcol_a:
        st.markdown("#### 🌐 Automated Live Ingestion & Training")
        st.write("Fetches World Bank cereal-yield rows when available, combines them with embedded crop benchmark estimates, and trains an optimized RandomForest model.")
        
        years_opt = st.slider("📅 Historical Years of Data to Source", min_value=3, max_value=10, value=6, step=1)
        optimize_opt = st.checkbox("⚡ Enable Hyperparameter Search (Grid tuning for maximum R² accuracy)", value=True)

        if st.button("🌐 Fetch From Trusted Sources & Retrain Model Now", key="btn_fetch_train"):
            with st.spinner("Fetching World Bank data and preparing local benchmark records..."):
                from src.data_fetcher import AutomatedDatasetFetcher
                fetcher = AutomatedDatasetFetcher()
                fetch_log = fetcher.fetch_and_update_dataset(years_back=years_opt)
                
            st.success(f"✅ Dataset updated! {fetch_log.get('new_records_added', 0)} previously unseen crop records added.")
            st.write(f"**Sources Queried:** {', '.join(fetch_log.get('sources_queried', [])) or 'None'}")
            st.write(f"**Data Included:** {', '.join(fetch_log.get('sources_used', [])) or 'No new source records'}")
            st.write(f"**Years Covered:** {min(fetch_log.get('years_covered', [2018]))} &rarr; {max(fetch_log.get('years_covered', [2024]))}")

            with st.spinner("Training & optimizing RandomForestRegressor pipeline..."):
                pipe, metrics = train_yield_model(auto_fetch_sources=False, optimize_hyperparams=optimize_opt)
                st.session_state.yield_predictor.load_model()
                st.success(f"🎉 Model Retrained! Test R² Accuracy Score: {metrics['r2_score_test']}")

    with tcol_b:
        st.markdown("#### 🔍 Dataset Audit & Inspection")
        st.write("Inspects column distributions, missing value counts, and physical yield limits on the same local dataset shown below.")
        if st.button("Run Data Audit on yield_df.csv", key="btn_audit"):
            with st.spinner("Auditing dataset schema and statistics..."):
                audit_res = run_data_audit()
                st.success("Audit complete!")
                st.json(audit_res)

    st.markdown("---")
    st.markdown("#### 🧾 Training Dataset Rows")
    dataset_path = DATA_DIR / "yield_df.csv"
    if dataset_path.exists():
        training_df = pd.read_csv(dataset_path)
        st.caption(
            f"Showing the persisted dataset the training pipeline reads: `{dataset_path}`. "
            "The model uses Area, Item, Year, rainfall, pesticides, and temperature as inputs; "
            "`hg/ha_yield` is the target it learns to predict."
        )
        st.markdown(
            """
            <div style="padding: 14px 18px; margin: 12px 0; border-radius: 12px;
                        background: linear-gradient(120deg, #0b2f24, #123f30);
                        border: 1px solid #287454; color: #e8fff3;">
              <span style="font-size: 0.78rem; font-weight: 700; letter-spacing: 1px;
                           color: #9de8bd;">TRUSTED DATA SOURCE</span>
              <span style="margin-left: 12px; padding: 5px 12px; border-radius: 20px;
                           background: #1d5b40; color: #ffffff; font-weight: 600;">
                World Bank Open Data
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        filter_area, filter_crop, filter_year = st.columns(3)
        areas = ["All"] + sorted(training_df["Area"].dropna().astype(str).unique().tolist())
        crops = ["All"] + sorted(training_df["Item"].dropna().astype(str).unique().tolist())
        years = ["All"] + sorted(training_df["Year"].dropna().unique().tolist(), reverse=True)
        selected_filter_area = filter_area.selectbox("Filter by area", areas, key="dataset_area_filter")
        selected_filter_crop = filter_crop.selectbox("Filter by crop", crops, key="dataset_crop_filter")
        selected_filter_year = filter_year.selectbox("Filter by year", years, key="dataset_year_filter")

        filtered_df = training_df
        if selected_filter_area != "All":
            filtered_df = filtered_df[filtered_df["Area"] == selected_filter_area]
        if selected_filter_crop != "All":
            filtered_df = filtered_df[filtered_df["Item"] == selected_filter_crop]
        if selected_filter_year != "All":
            filtered_df = filtered_df[filtered_df["Year"] == selected_filter_year]

        st.caption(f"{len(filtered_df):,} of {len(training_df):,} rows · {len(training_df.columns)} columns")
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download training dataset (CSV)",
            data=training_df.to_csv(index=False).encode("utf-8"),
            file_name="yield_df.csv",
            mime="text/csv",
            key="download_training_dataset",
        )
    else:
        st.warning(f"No training CSV exists at `{dataset_path}` yet. Fetch data above or add the dataset before training.")

    st.markdown("---")
    # Display current metrics if available
    metrics_file = ARTIFACTS_DIR / "model_metrics.json"
    if metrics_file.exists():
        st.markdown("#### 📈 Active Model Performance & Capabilities")
        with open(metrics_file, "r", encoding="utf-8") as f:
            curr_metrics = json.load(f)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test R² Accuracy", curr_metrics.get("r2_score_test", "N/A"))
        m2.metric("Mean Error (MAE)", f"{curr_metrics.get('mae_tonnes_per_ha', 'N/A')} t/ha")
        m3.metric("RMSE Error", f"{curr_metrics.get('rmse_tonnes_per_ha', 'N/A')} t/ha")
        m4.metric("Total Samples", curr_metrics.get("train_samples", 0) + curr_metrics.get("test_samples", 0))

        with st.expander("🔍 View Trained Model Details & Feature Ranges"):
            st.json(curr_metrics)
