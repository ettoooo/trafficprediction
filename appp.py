from fastapi import FastAPI
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

# API Keys
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

# Fetch traffic data from OpenStreetMap Overpass API
def fetch_traffic_data(bbox):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["highway"="traffic_signals"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out count;
    """
    response = requests.get(overpass_url, params={"data": query})
    data = response.json()

    traffic_count = data.get("elements", [{}])[0].get("count", 0)
    return {"timestamp": pd.Timestamp.now().isoformat(), "traffic_count": traffic_count}

# Fetch weather data from OpenWeatherMap
def fetch_weather_data(city="Delhi"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather_condition": data["weather"][0]["description"],
    }

# Save data to Excel
def save_data():
    traffic_data = fetch_traffic_data([28.60, 77.10, 28.90, 77.30])
    weather_data = fetch_weather_data("Delhi")

    df = pd.DataFrame([{**traffic_data, **weather_data}])
    df.to_excel("traffic_weather_data.xlsx", index=False)

    return df

# Train & Predict using Linear Regression
def predict_traffic():
    try:
        df = pd.read_excel("traffic_weather_data.xlsx")

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**9
        X = df[["timestamp", "temperature", "humidity"]]
        y = df["traffic_count"]

        model = LinearRegression()
        model.fit(X, y)

        future_time = np.array([[pd.Timestamp.now().timestamp(), df["temperature"].mean(), df["humidity"].mean()]])
        predicted_traffic = model.predict(future_time)[0]

        return predicted_traffic
    except Exception as e:
        return str(e)

@app.get("/data")
def get_data():
    save_data()
    predicted_traffic = predict_traffic()
    
    return JSONResponse({"traffic": fetch_traffic_data([28.60, 77.10, 28.90, 77.30]),
                         "weather": fetch_weather_data("Delhi"),
                         "predicted_traffic": predicted_traffic})

