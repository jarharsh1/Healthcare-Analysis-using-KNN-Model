"""Request/response models for the serving API.

The feature model mirrors the six biomechanics measurements (no target).
"""

from pydantic import BaseModel, Field


class PatientMeasurements(BaseModel):
    P_incidence: float = Field(..., examples=[63.03])
    P_tilt: float = Field(..., examples=[22.55])
    L_angle: float = Field(..., examples=[39.61])
    S_slope: float = Field(..., examples=[40.48])
    P_radius: float = Field(..., examples=[98.67])
    S_Degree: float = Field(..., examples=[-0.25])


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict
