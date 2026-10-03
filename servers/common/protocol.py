"""Pydantic request/response models for all three services."""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")


class DetectResponse(BaseModel):
    boxes: List[List[float]] = Field(default_factory=list)
    width: int = 0
    height: int = 0


class PoseRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    boxes: List[List[float]] = Field(default_factory=list)


class PersonResult(BaseModel):
    box: List[float] = Field(default_factory=list)
    keypoints: Dict[str, List[float]] = Field(default_factory=dict)


class PoseResponse(BaseModel):
    persons: List[PersonResult] = Field(default_factory=list)


class DepthRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    points: List[List[float]] = Field(default_factory=list)


class DepthResponse(BaseModel):
    depths: List[float] = Field(default_factory=list)


class DetectObjectsRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    classes: List[int] = Field(default_factory=list, description="COCO class IDs; empty = all")


class DetectedObjectItem(BaseModel):
    box: List[float] = Field(default_factory=list)
    class_id: int = 0
    class_name: str = ""


class DetectObjectsResponse(BaseModel):
    objects: List[DetectedObjectItem] = Field(default_factory=list)
    width: int = 0
    height: int = 0


class FaceLandmarkRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    boxes: List[List[float]] = Field(default_factory=list)


class FaceResult(BaseModel):
    detected: bool = False
    landmarks: List[List[float]] = Field(default_factory=list)


class FaceLandmarkResponse(BaseModel):
    persons: List[FaceResult] = Field(default_factory=list)


class VLMDetectRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image")
    mode: str = Field("no_reason", description="no_reason: code only; reason: code + explanation")
    max_tokens: int = Field(1024, description="max generation tokens")


class VLMDetectResponse(BaseModel):
    primary_code: str = Field("", description="2-letter behavior code, e.g. 'ph'")
    explanation: str = Field("", description="model explanation, empty in no_reason mode")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="code -> probability for all 15 classes")
    raw_output: str = Field("", description="raw model output text")
    latency_ms: float = Field(0.0, description="inference latency in milliseconds")
    error: str = Field("", description="non-empty when inference failed")
