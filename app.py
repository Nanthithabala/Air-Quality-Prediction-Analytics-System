import streamlit as st
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai

import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import timedelta

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD DATA ---------------- #

@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_air_quality_hist.csv")

    df["date"] = pd.to_datetime(df["date"])

    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df["Month_Name"] = df["date"].dt.month_name()
    df["Quarter"] = df["date"].dt.quarter

    def season(month):

        if month in [12,1,2]:
            return "Winter"

        elif month in [3,4,5]:
            return "Summer"

        elif month in [6,7,8,9]:
            return "Monsoon"

        else:
            return "Post-Monsoon"

    df["Season"] = df["Month"].apply(season)

    df["Pollutant_Count"] = df["prominent_pollutants"].fillna("").apply(
        lambda x: len(str(x).split(","))
    )

    return df

df = load_data()

# ---------------- LOAD MODELS ---------------- #

regression_model = joblib.load("aqi_regression_model.pkl")

try:
    prophet_model = joblib.load("prophet_aqi_model.pkl")
except Exception as e:
    st.error(f"Prophet loading error: {e}")
    prophet_model = None

# ---------------- GEMINI AI ---------------- #

import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

gemini_model = genai.GenerativeModel("gemini-1.5-flash")
# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🌍 Air Quality")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 AQI Prediction",
        "📊 AQI Forecast",
        "📉 Dashboard",
        "🤖 GenAI Assistant",
        "ℹ️ About"
    ]
)

# ---------------- HOME ---------------- #

if page=="🏠 Home":

    st.title("🌍 Air Quality Prediction System")

    st.markdown(
        """
Predict Air Quality using *Machine Learning* and
Forecast Future AQI using *Facebook Prophet*.
"""
    )

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "States",
        df["state"].nunique()
    )

    c2.metric(
        "Areas",
        df["area"].nunique()
    )

    c3.metric(
        "Monitoring Stations",
        int(df["number_of_monitoring_stations"].sum())
    )

    c4.metric(
        "Pollutants",
        df["prominent_pollutants"].nunique()
    )

    st.divider()

    col1,col2=st.columns(2)

    with col1:

        st.subheader("Project Features")

        st.success("✔️ AQI Prediction")
        st.success("✔️ Future AQI Forecast")
        st.success("✔️ Dashboard Analytics")
        st.success("✔️ Health Recommendation")

    with col2:

        st.subheader("Technologies")

        st.info("Python")
        st.info("Streamlit")
        st.info("Random Forest")
        st.info("Facebook Prophet")
        st.info("Plotly")

# ---------------- AQI PREDICTION ---------------- #

elif page == "📈 AQI Prediction":

    st.title("📈 AQI Value Prediction")

    col1, col2 = st.columns(2)

    with col1:

        state = st.selectbox(
            "State",
            sorted(df["state"].unique())
        )

        state_df = df[df["state"] == state]

        area = st.selectbox(
            "Area",
            sorted(state_df["area"].unique())
        )

        area_df = state_df[state_df["area"] == area]

        stations = st.number_input(
            "Number of Monitoring Stations",
            min_value=1,
            value=int(area_df["number_of_monitoring_stations"].iloc[0])
        )

    with col2:

        pollutant = st.selectbox(
            "Prominent Pollutant",
            sorted(area_df["prominent_pollutants"].unique())
        )

        year = st.selectbox(
            "Year",
            sorted(df["Year"].unique())
        )

        month = st.selectbox(
            "Month",
            sorted(df["Month"].unique())
        )

        month_df = df[df["Month"] == month].iloc[0]

        month_name = month_df["Month_Name"]
        quarter = month_df["Quarter"]
        season = month_df["Season"]

        pollutant_count = len(pollutant.split(","))

    st.divider()

    if st.button("Predict AQI", use_container_width=True):

        input_df = pd.DataFrame({
            
            "state":[state],
            "area":[area],
            "number_of_monitoring_stations":[stations],
            "prominent_pollutants":[pollutant],
            "unit":[df["unit"].iloc[0]],
            "Year":[year],
            "Month":[month],
            "Month_Name":[month_name],
            "Quarter":[quarter],
            "Season":[season],
            "Pollutant_Count":[pollutant_count]
            })

        prediction = regression_model.predict(input_df)

        aqi = round(float(prediction[0]),2)
        st.session_state["aqi"] = aqi
        st.session_state["state"] = state
        st.session_state["area"] = area
        st.session_state["pollutant"] = pollutant

        st.metric(
            "Predicted AQI",
            aqi
        )

        if aqi <= 50:

            status = "🟢 Good"
            advice = "Air quality is excellent."

            st.success(status)

        elif aqi <= 100:

            status = "🟡 Satisfactory"
            advice = "Acceptable for most people."

            st.info(status)

        elif aqi <= 200:

            status = "🟠 Moderate"
            advice = "Sensitive groups should reduce prolonged outdoor activity."

            st.warning(status)

        elif aqi <= 300:

            status = "🔴 Poor"
            advice = "Reduce outdoor activities."

            st.error(status)

        elif aqi <= 400:

            status = "🟣 Very Poor"
            advice = "Stay indoors whenever possible."

            st.error(status)

        else:

            status = "⚫ Severe"
            advice = "Avoid going outside."

            st.error(status)

        st.session_state["status"] = status    

        st.subheader("Health Recommendation")
        st.write(advice)

        st.subheader("Prediction Summary")

        summary = pd.DataFrame({
            "Field":[
                "State",
                "Area",
                "Pollutant",
                "Predicted AQI",
                "Air Quality"
            ],
            "Value":[
                state,
                area,
                pollutant,
                aqi,
                status
            ]
        })

        st.dataframe(summary, use_container_width=True) 
        
        if "aqi" in st.session_state:
            st.divider()
            st.subheader("🤖 AI Explanation")
            if st.button("✨ Explain Prediction with AI"):
                with st.spinner("Analyzing prediction..."):
                    prompt = f"""
                    You are an Air Quality Expert.

                    This Air Quality Prediction System predicted:

                    State: {st.session_state["state"]}
                    Area: {st.session_state["area"]}
                    Pollutant: {st.session_state["pollutant"]}

                    Predicted AQI: {st.session_state["aqi"]}

                    Air Quality Status: {st.session_state["status"]}

                    Provide:

                    1. Meaning of this AQI.
                    2. Possible reasons for this AQI.
                    3. Health effects.
                    4. Safety precautions.
                    5. Suggestions to improve air quality.

                    Explain in simple English.
                    """

                    response = gemini_model.generate_content(prompt)

                    st.write(response.text)

        
# ---------------- AQI FORECAST ---------------- #

elif page == "📊 AQI Forecast":

    st.title("📊 AQI Forecast")

    if prophet_model is None:

        st.error("Prophet model not found.")

    else:

        days = st.selectbox(
            "Forecast Period",
            [30,90,180,365]
        )

        future = prophet_model.make_future_dataframe(
            periods=days
        )

        forecast = prophet_model.predict(future)

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=forecast["ds"],
                y=forecast["yhat"],

                mode="lines",

                name="Forecast"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=forecast["ds"],
                y=forecast["yhat_upper"],

                mode="lines",

                line=dict(width=0),

                showlegend=False

            )

        )

        fig.add_trace(

            go.Scatter(

                x=forecast["ds"],
                y=forecast["yhat_lower"],

                mode="lines",

                fill="tonexty",

                line=dict(width=0),

                name="Confidence Interval"

            )

        )

        fig.update_layout(

            title="Future AQI Forecast",

            xaxis_title="Date",

            yaxis_title="AQI",

            height=600

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Forecast Table")

        result = forecast[

            [

                "ds",

                "yhat",

                "yhat_lower",

                "yhat_upper"

            ]

        ].tail(days)

        result.columns = [

            "Date",

            "Predicted AQI",

            "Lower Bound",

            "Upper Bound"

        ]

        st.dataframe(
            result,
            use_container_width=True
        )

        st.divider()
        st.subheader("🤖 AI Forecast Analysis")
        if st.button("✨ Explain Forecast with AI"):
            with st.spinner("Analyzing forecast..."):
                avg_aqi = round(result["Predicted AQI"].mean(), 2)
                max_aqi = round(result["Predicted AQI"].max(), 2)
                min_aqi = round(result["Predicted AQI"].min(), 2)
                prompt = f"""
                
                You are an Air Quality Forecast Expert.
                
                The forecast period is {days} days.
                
                Forecast Summary:
                
                - Average AQI: {avg_aqi}
                - Highest AQI: {max_aqi}
                - Lowest AQI: {min_aqi}

                Explain:

                1. Overall AQI trend.
                2. Whether air quality is improving or worsening.
                3. Possible environmental reasons.
                4. Health precautions.
                5. Recommendations for the public.

                Explain in simple English.
                """
                
                response = gemini_model.generate_content(prompt)

                st.write(response.text)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(

            "Download Forecast CSV",

            csv,

            "AQI_Forecast.csv",

            "text/csv"

        )

# ---------------- DASHBOARD ---------------- #

elif page == "📉 Dashboard":

    st.title("📉 Air Quality Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        state_avg = (
            df.groupby("state")["aqi_value"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig1 = px.bar(
            state_avg,
            x="state",
            y="aqi_value",
            title="Average AQI by State",
            color="aqi_value"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        month_avg = (
            df.groupby("Month_Name")["aqi_value"]
            .mean()
            .reindex([
                "January","February","March","April",
                "May","June","July","August",
                "September","October","November","December"
            ])
            .reset_index()
        )

        fig2 = px.line(
            month_avg,
            x="Month_Name",
            y="aqi_value",
            markers=True,
            title="Average AQI by Month"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        season_avg = (
            df.groupby("Season")["aqi_value"]
            .mean()
            .reset_index()
        )

        fig3 = px.pie(
            season_avg,
            names="Season",
            values="aqi_value",
            title="Season-wise AQI Distribution"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    with col4:

        pollutant = (
            df["prominent_pollutants"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        pollutant.columns = [
            "Pollutant",
            "Count"
        ]

        fig4 = px.bar(
            pollutant,
            x="Pollutant",
            y="Count",
            title="Top 10 Pollutants",
            color="Count"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.subheader("Top 10 Most Polluted Areas")

    top_area = (
        df.groupby("area")["aqi_value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_area,
        use_container_width=True
    )

    st.divider()
    st.subheader("🤖 AI Dashboard Insights")
    if st.button("✨ Generate AI Insights"):
        with st.spinner("Analyzing dashboard..."):
            highest_state = (
                df.groupby("state")["aqi_value"]
                .mean()
                .sort_values(ascending=False)
                .index[0]
            )

            lowest_state = (
                df.groupby("state")["aqi_value"]
                .mean()
                .sort_values()
                .index[0]
            )

            highest_aqi = round(df["aqi_value"].max(), 2)
            average_aqi = round(df["aqi_value"].mean(), 2)

            top_pollutant = (
                df["prominent_pollutants"]
                .value_counts()
                .index[0]
            )
            
            prompt = f"""
            You are an Environmental Data Analyst.

            The dashboard contains the following information:

            Average AQI: {average_aqi}

            Highest AQI Recorded: {highest_aqi}

            Most Polluted State: {highest_state}

            Cleanest State: {lowest_state}

            Most Common Pollutant: {top_pollutant}

            Generate a professional dashboard summary including:

            1. Overall air quality condition.
            2. Key insights.
            3. Pollution trend.
            4. Health recommendations.
            5. Environmental recommendations.

            Keep the explanation simple and professional.
            """
            response = gemini_model.generate_content(prompt)

            st.success("AI Dashboard Summary")

            st.write(response.text)

elif page == "🤖 GenAI Assistant":

    st.title("🤖 GenAI Environmental Assistant")

    st.markdown("### Ask anything about Air Quality, AQI and Pollution")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What does AQI 200 mean?"
    )

    if st.button("Ask Gemini"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            with st.spinner("Thinking..."):

                prompt = f"""
You are the AI Assistant for an Air Quality Prediction and Forecasting System.

This project was developed using:

* Python
* Streamlit
* Random Forest Regression
* Facebook Prophet Forecasting
* Plotly
* Pandas
* Joblib
* Machine Learning
* Business Intelligence Dashboard
* Google Gemini (Generative AI)

The project includes:

1. AQI Prediction using Random Forest Regression.
2. Future AQI Forecasting using Facebook Prophet.
3. Interactive Analytics Dashboard.
4. Health Recommendations based on AQI.
5. AI-powered explanations using Gemini.

If the user asks about this project, explain it professionally.

If the user asks about:
- AQI
- Air Pollution
- PM2.5
- PM10
- Climate
- Environment
- Health Effects
- Pollution Prevention

answer accurately in simple English.

If the question is unrelated to air quality or this project, politely reply:

"I'm designed to answer questions related to this Air Quality Prediction System and environmental topics."

User Question:
{question}
"""

                response = gemini_model.generate_content(prompt)

                st.success("Answer")

                st.write(response.text)
# ---------------- ABOUT ---------------- #

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🌍 Air Quality Prediction System

    This project predicts Air Quality Index (AQI) using Machine Learning
    and forecasts future AQI using Prophet.

    ---
    """)

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📌 Project Information")

        st.write("""
        *Project:* Air Quality Prediction System

        *Course:* M.Sc Data Analytics

        *Language:* Python

        *Framework:* Streamlit
        """)

    with c2:

        st.subheader("🛠️ Technologies")

        st.write("""
        ✅ Random Forest Regression

        ✅ Prophet

        ✅ Pandas

        ✅ Plotly

        ✅ Joblib

        ✅ Streamlit
        """)

    st.divider()

    st.subheader("📂 Dataset Features")

    st.write("""
    - State
    - Area
    - AQI Value
    - Monitoring Stations
    - Prominent Pollutants
    - Month & Season
    """)

    st.divider()

    st.subheader("🎯 Project Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.success("✔️ AQI Prediction")
        st.success("✔️ Future AQI Forecast")
        st.success("✔️ Dashboard Analytics")

    with feature2:

        st.success("✔️ Health Recommendation")
        st.success("✔️ Interactive Charts")
        st.success("✔️ Professional Streamlit UI")

    st.divider()
                        
