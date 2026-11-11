"""Regulatory Forecaster — Time series risk forecasting and anomaly detection."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class Anomaly(BaseModel):
    """Detected anomaly in regulatory publishing or risk scores."""
    timestamp: str
    value: float
    expected_value: float
    deviation: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str


class Forecast(BaseModel):
    """Projected future values for regulatory velocity or risk scores."""
    metric_name: str
    horizon_days: int
    predicted_values: list[float]
    confidence_intervals_lower: list[float]
    confidence_intervals_upper: list[float]
    timestamps: list[str]
    trend: str  # "INCREASING", "DECREASING", "STABLE"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegulatoryForecaster:
    """Time series analysis for regulatory risk scores and velocity."""

    def __init__(self, anomaly_z_threshold: float = 2.0) -> None:
        self.anomaly_z_threshold = anomaly_z_threshold

    def forecast_risk(
        self,
        historical_scores: list[dict[str, Any]],
        horizon: int = 30,
        metric_key: str = "score",
    ) -> Forecast:
        """Forecast future values using Holt-Winters exponential smoothing baseline.
        
        Designed to allow hot-swapping with TimesFM/Chronos foundation models.
        """
        if not historical_scores:
            return Forecast(
                metric_name=metric_key,
                horizon_days=horizon,
                predicted_values=[0.0] * horizon,
                confidence_intervals_lower=[0.0] * horizon,
                confidence_intervals_upper=[0.0] * horizon,
                timestamps=[],
                trend="STABLE",
            )

        values = [float(item.get(metric_key, 0.0)) for item in historical_scores]
        n = len(values)

        if n < 2:
            base = values[0] if values else 50.0
            preds = [base] * horizon
            return Forecast(
                metric_name=metric_key,
                horizon_days=horizon,
                predicted_values=preds,
                confidence_intervals_lower=[max(0.0, base - 5.0)] * horizon,
                confidence_intervals_upper=[min(100.0, base + 5.0)] * horizon,
                timestamps=[],
                trend="STABLE",
            )

        # Compute level and trend via double exponential smoothing
        alpha = 0.3
        beta = 0.1
        level = values[0]
        trend = values[1] - values[0]

        for val in values[1:]:
            prev_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        # Residual variance for confidence intervals
        residuals = []
        running_level = values[0]
        running_trend = 0.0
        for val in values:
            pred = running_level + running_trend
            residuals.append(val - pred)
            prev_l = running_level
            running_level = alpha * val + (1 - alpha) * (running_level + running_trend)
            running_trend = beta * (running_level - prev_l) + (1 - beta) * running_trend

        variance = sum(r**2 for r in residuals) / max(1, len(residuals))
        std_err = math.sqrt(variance) if variance > 0 else 2.5

        predicted_values: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for h in range(1, horizon + 1):
            pred = round(max(0.0, min(100.0, level + h * trend)), 2)
            margin = 1.96 * std_err * math.sqrt(h)
            low = round(max(0.0, pred - margin), 2)
            high = round(min(100.0, pred + margin), 2)

            predicted_values.append(pred)
            lower_bounds.append(low)
            upper_bounds.append(high)

        overall_trend = "STABLE"
        if trend > 0.15:
            overall_trend = "INCREASING"
        elif trend < -0.15:
            overall_trend = "DECREASING"

        return Forecast(
            metric_name=metric_key,
            horizon_days=horizon,
            predicted_values=predicted_values,
            confidence_intervals_lower=lower_bounds,
            confidence_intervals_upper=upper_bounds,
            timestamps=[],
            trend=overall_trend,
            metadata={"base_level": level, "trend_slope": trend, "samples_used": n},
        )

    def detect_anomalies(
        self,
        time_series: list[dict[str, Any]],
        value_key: str = "value",
        time_key: str = "timestamp",
    ) -> list[Anomaly]:
        """Detect statistical anomalies using dynamic z-score thresholding."""
        if len(time_series) < 3:
            return []

        values = [float(item.get(value_key, 0.0)) for item in time_series]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        anomalies: list[Anomaly] = []
        for item in time_series:
            val = float(item.get(value_key, 0.0))
            ts = str(item.get(time_key, datetime.now(timezone.utc).isoformat()))
            z_score = abs(val - mean) / std_dev

            if z_score >= self.anomaly_z_threshold:
                deviation = round(val - mean, 2)
                severity = "LOW"
                if z_score >= 3.5:
                    severity = "CRITICAL"
                elif z_score >= 2.8:
                    severity = "HIGH"
                elif z_score >= 2.0:
                    severity = "MEDIUM"

                anomalies.append(
                    Anomaly(
                        timestamp=ts,
                        value=round(val, 2),
                        expected_value=round(mean, 2),
                        deviation=deviation,
                        severity=severity,
                        description=(
                            f"Z-score {z_score:.2f} exceeds threshold {self.anomaly_z_threshold}. "
                            f"Observed: {val:.2f}, Expected: {mean:.2f}"
                        ),
                    )
                )

        logger.info(
            "anomaly_detection_completed",
            total_points=len(time_series),
            anomalies_detected=len(anomalies),
        )
        return anomalies
