from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io

from backend.ai_engine import (
    clean_data, generate_synthetic_data, create_sample_data,
    train_and_forecast, calculate_rul, calculate_health_score,
    generate_ai_insights
)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Industrial AI API", version="1.0.0")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Industrial AI API is running"}

@app.post("/api/forecast")
async def run_forecast(
    file: UploadFile = File(None),
    use_demo: bool = Form(True),
    forecast_years: int = Form(5),
    epochs: int = Form(30),
    degradation: float = Form(0.0003),
    eq_type: str = Form("Motor"),
    vyahh: float = Form(7.0),
    vxahh: float = Form(6.0)
):
    try:
        df = None
        if file is not None and file.filename:
            contents = await file.read()
            if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(contents))
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload Excel or CSV.")
        
        if df is None and use_demo:
            df = create_sample_data()
            
        if df is None:
            raise HTTPException(status_code=400, detail="No data provided")

        # Clean data
        df = clean_data(df)
        
        # Identify feature columns (Improved for real-world robustness)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_cols = []
        # Mapping for common industrial sensor names
        mapping = {
            'VYI': ['VYI', 'VIB_Y', 'VIBRATION_Y', 'VY'],
            'VXI': ['VXI', 'VIB_X', 'VIBRATION_X', 'VX'],
            'T': ['T', 'TEMP', 'TEMPERATURE', 'HARORAT'],
            'Speed': ['SPEED', 'RPM', 'TEZLIK', 'FREQUENCY'],
            'Current': ['CURRENT', 'AMPS', 'TOK', 'I']
        }
        
        for key, aliases in mapping.items():
            for col in numeric_cols:
                col_upper = col.upper()
                if col_upper in aliases or any(a in col_upper for a in aliases):
                    feature_cols.append(col)
                    break 
        
        # Fallback if no standard names found
        if not feature_cols:
            feature_cols = numeric_cols[:5]
        
        # Ensure unique columns
        feature_cols = list(dict.fromkeys(feature_cols))

        # Generate forecast
        # FIX: Use the actual forecast_years parameter for synthetic data extension
        total_target_days = len(df) + (forecast_years * 365)
        extended_df = generate_synthetic_data(df, target_days=total_target_days, degradation_factor=degradation)
        
        forecast_df, _, _, is_fallback = train_and_forecast(
            extended_df, feature_cols, forecast_years=forecast_years,
            seq_length=min(30, len(extended_df) // 3), epochs=epochs
        )

        # Health metrics
        latest = df[feature_cols].iloc[-1]
        thresholds = [vyahh if 'VY' in c else (vxahh if 'VX' in c else 100) for c in feature_cols]
        health_score = calculate_health_score(latest.values, thresholds)

        rul_vy_days, rul_vx_days = None, None
        if forecast_df is not None:
            if 'VYI' in forecast_df.columns:
                rul_vy_days, _ = calculate_rul(forecast_df, 'VYI', vyahh)
            if 'VXI' in forecast_df.columns:
                rul_vx_days, _ = calculate_rul(forecast_df, 'VXI', vxahh)

        insights = generate_ai_insights(health_score, rul_vy_days, rul_vx_days, forecast_df, feature_cols)

        # Formatting response
        if 'Time' in df.columns:
            df['Time'] = df['Time'].astype(str)
        historical_data = df.fillna("").to_dict(orient='records')
        
        forecast_data = []
        if forecast_df is not None:
            forecast_df_reset = forecast_df.reset_index()
            if 'index' in forecast_df_reset.columns:
                forecast_df_reset.rename(columns={'index': 'Time'}, inplace=True)
            forecast_df_reset['Time'] = forecast_df_reset['Time'].astype(str)
            forecast_data = forecast_df_reset.fillna("").to_dict(orient='records')

        return {
            "status": "success",
            "health_score": float(health_score),
            "rul_vy_days": int(rul_vy_days) if rul_vy_days is not None else None,
            "rul_vx_days": int(rul_vx_days) if rul_vx_days is not None else None,
            "insights": insights,
            "historical_data": historical_data,
            "forecast_data": forecast_data,
            "feature_cols": feature_cols,
            "forecast_horizon": forecast_years,
            "is_fallback": is_fallback
        }

    except Exception as e:
        logger.error(f"Error in /api/forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")
