#!/usr/bin/env python3
"""성능 테스트 및 검증 스크립트"""

import sys
import time
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_config_import():
    """설정 import 테스트"""
    print("=== 설정 Import 테스트 ===")
    try:
        from wsi_viewer.config import CONFIG
        print(f"✓ Config 로드 성공")
        print(f"  - 기본 패치 크기: {CONFIG.ai.default_patch_size}")
        print(f"  - 패치 크기 옵션: {CONFIG.ai.patch_sizes}")
        print(f"  - 모델 경로: {CONFIG.ai.model_path}")
        print(f"  - 멀티프로세싱: {CONFIG.ai.enable_multiprocessing}")
        print(f"  - 실시간 표시: {CONFIG.ai.enable_real_time_display}")
        print(f"  - 워커 수: {CONFIG.ai.default_worker_count}")
        return True
    except Exception as e:
        print(f"✗ Config 로드 실패: {e}")
        return False

def test_detector_import():
    """Detector import 테스트"""
    print("\n=== Detector Import 테스트 ===")
    try:
        from wsi_viewer.ai.mitosis_detector import MitosisDetector
        print("✓ MitosisDetector import 성공")

        from wsi_viewer.ai.multiprocess_patch_extractor import MultiprocessPatchProcessor
        print("✓ MultiprocessPatchProcessor import 성공")

        from wsi_viewer.ai.real_time_detection_worker import RealTimeDetectionWorker
        print("✓ RealTimeDetectionWorker import 성공")

        return True
    except Exception as e:
        print(f"✗ Detector import 실패: {e}")
        return False

def test_detector_initialization():
    """Detector 초기화 테스트"""
    print("\n=== Detector 초기화 테스트 ===")
    try:
        from wsi_viewer.ai.mitosis_detector import MitosisDetector
        from wsi_viewer.config import CONFIG

        # Config 설정으로 초기화 시도
        detector = MitosisDetector()
        print(f"✓ MitosisDetector 초기화 성공")
        print(f"  - 모델 경로: {detector.model_path}")
        print(f"  - 설정 경로: {detector.config_path}")

        # 모델 파일 존재 확인
        if detector.model_path.exists():
            print(f"✓ 모델 파일 존재: {detector.model_path}")
        else:
            print(f"⚠ 모델 파일 없음: {detector.model_path}")

        return True
    except Exception as e:
        print(f"✗ Detector 초기화 실패: {e}")
        return False

def test_performance_settings():
    """성능 설정 검증"""
    print("\n=== 성능 설정 검증 ===")
    try:
        from wsi_viewer.config import CONFIG

        print(f"패치 크기 옵션: {list(CONFIG.ai.patch_sizes)}")
        print(f"배치 크기 범위: 1 ~ {CONFIG.ai.max_batch_size}")
        print(f"워커 수 범위: 1 ~ {CONFIG.ai.max_worker_count}")
        print(f"큐 크기 - 패치: {CONFIG.ai.patch_queue_size}, 결과: {CONFIG.ai.result_queue_size}")

        # 메모리 예상 사용량 계산
        max_patch_size = max(CONFIG.ai.patch_sizes)
        patch_memory_mb = (max_patch_size * max_patch_size * 3 * 4) / (1024 * 1024)  # RGB float32
        queue_memory_mb = CONFIG.ai.patch_queue_size * patch_memory_mb

        print(f"\n예상 메모리 사용량:")
        print(f"  - 패치당: {patch_memory_mb:.1f} MB")
        print(f"  - 큐 최대: {queue_memory_mb:.1f} MB")
        print(f"  - 메모리 제한: {CONFIG.ai.memory_limit_mb} MB")

        if queue_memory_mb > CONFIG.ai.memory_limit_mb:
            print("⚠ 큐 크기가 메모리 제한을 초과할 수 있습니다")
        else:
            print("✓ 메모리 사용량이 적절합니다")

        return True
    except Exception as e:
        print(f"✗ 성능 설정 검증 실패: {e}")
        return False

def estimate_processing_time():
    """처리 시간 예상 계산"""
    print("\n=== 처리 시간 예상 ===")
    try:
        from wsi_viewer.config import CONFIG

        # 가상 슬라이드 크기 (40x 배율)
        slide_width_mm = 15  # 15mm
        slide_height_mm = 15  # 15mm
        mpp = 0.25  # 40x 배율의 mpp

        # 픽셀 크기 계산
        pixels_per_mm = 1000 / mpp  # 1000 microns per mm
        slide_width_px = int(slide_width_mm * pixels_per_mm)
        slide_height_px = int(slide_height_mm * pixels_per_mm)

        print(f"가상 슬라이드 크기: {slide_width_px} x {slide_height_px} 픽셀")

        for patch_size in CONFIG.ai.patch_sizes:
            overlap = int(patch_size * CONFIG.ai.overlap_ratio)
            step_size = patch_size - overlap

            patches_x = (slide_width_px + step_size - 1) // step_size
            patches_y = (slide_height_px + step_size - 1) // step_size
            total_patches = patches_x * patches_y

            # 티슈 영역 가정 (70%)
            tissue_patches = int(total_patches * 0.7)

            # 추정 처리 속도 (패치/초)
            # 멀티프로세싱: 더 빠름, 싱글프로세싱: 더 느림
            if CONFIG.ai.enable_multiprocessing:
                speed = CONFIG.ai.default_worker_count * 2.0  # 워커당 2패치/초
            else:
                speed = 1.0  # 1패치/초

            processing_time_sec = tissue_patches / speed
            processing_time_min = processing_time_sec / 60

            print(f"\n패치 크기 {patch_size}:")
            print(f"  - 총 패치: {total_patches:,}")
            print(f"  - 티슈 패치: {tissue_patches:,}")
            print(f"  - 예상 처리 시간: {processing_time_min:.1f}분")

        return True
    except Exception as e:
        print(f"✗ 처리 시간 예상 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("TaegyuSlide 성능 테스트 및 검증")
    print("=" * 50)

    tests = [
        test_config_import,
        test_detector_import,
        test_detector_initialization,
        test_performance_settings,
        estimate_processing_time
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 테스트 실행 중 오류: {e}")
            failed += 1

    print(f"\n=== 테스트 결과 ===")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"총 테스트: {passed + failed}")

    if failed == 0:
        print("✓ 모든 테스트 통과!")
        return 0
    else:
        print("⚠ 일부 테스트 실패")
        return 1

if __name__ == "__main__":
    sys.exit(main())