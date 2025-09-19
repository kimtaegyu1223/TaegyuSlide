"""AI 기반 분석 모듈 - 서버 API 방식"""

from .api_client import ObjectDetectionAPIClient, DetectionResult, APIConfig
from .server_detection_worker import ServerBasedDetectionWorker, BatchDetectionWorker, ProcessingStats
from .slide_processor import SlideProcessor, SlideAnalysis, PatchInfo
from .tissue_detector import TissueDetector, TissueRegion

__all__ = [
    'ObjectDetectionAPIClient', 'DetectionResult', 'APIConfig',
    'ServerBasedDetectionWorker', 'BatchDetectionWorker', 'ProcessingStats',
    'SlideProcessor', 'SlideAnalysis', 'PatchInfo',
    'TissueDetector', 'TissueRegion'
]