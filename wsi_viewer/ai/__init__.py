"""AI 기반 분석 모듈 - 서버 API 방식"""

from .api_client import MitosisAPIClient, DetectionResult, APIConfig
from .detection_worker import MitosisDetectionWorker
from .server_detection_worker import ServerBasedDetectionWorker, BatchDetectionWorker, ProcessingStats
from .slide_processor import SlideProcessor, SlideAnalysis, PatchInfo
from .tissue_detector import TissueDetector, TissueRegion

__all__ = [
    'MitosisAPIClient', 'DetectionResult', 'APIConfig', 'MitosisDetectionWorker',
    'ServerBasedDetectionWorker', 'BatchDetectionWorker', 'ProcessingStats',
    'SlideProcessor', 'SlideAnalysis', 'PatchInfo',
    'TissueDetector', 'TissueRegion'
]