#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label Studio Helper 빌드 스크립트
Embedded Python + PyInstaller onedir 방식

특징:
- Embedded Python 포함 (~50MB)
- PyTorch는 런타임에 자동 설치 (외부)
- 일반 사용자 권한 강제
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime

# ==================== 설정 ====================
PROJECT_ROOT = Path(__file__).parent
RELEASE_DIR = PROJECT_ROOT / "release"
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"

SPEC_FILE = PROJECT_ROOT / "label_studio_helper.spec"
APP_NAME = "label_studio_helper"
APP_FOLDER = DIST_DIR / APP_NAME

# ================================================


def print_section(title):
    """섹션 제목 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def clean_build_artifacts():
    """빌드 산출물 폴더 삭제"""
    print_section("이전 빌드 산출물 정리")

    for folder in [BUILD_DIR, DIST_DIR]:
        if folder.exists():
            try:
                shutil.rmtree(folder)
                print(f"[OK] 삭제: {folder.name}/")
            except Exception as e:
                print(f"[경고] 삭제 실패 ({folder.name}): {e}")
        else:
            print(f"  (없음: {folder.name}/)")


def build_with_pyinstaller():
    """PyInstaller로 빌드"""
    print_section("PyInstaller 빌드 시작")

    if not SPEC_FILE.exists():
        print(f"[오류] .spec 파일 없음: {SPEC_FILE}")
        return False

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(SPEC_FILE),
        "--noconfirm",
        "--clean",
    ]

    print(f"빌드 명령: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print("\n[OK] PyInstaller 빌드 성공!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[오류] 빌드 실패: {e}")
        return False


def create_zip_distribution():
    """빌드 결과를 ZIP으로 압축"""
    print_section("ZIP 배포 파일 생성")

    if not APP_FOLDER.exists():
        print(f"[오류] 배포 폴더 없음: {APP_FOLDER}")
        return False

    RELEASE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"LabelStudioHelper_{timestamp}.zip"
    zip_path = RELEASE_DIR / zip_filename

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(APP_FOLDER):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(DIST_DIR)
                    zipf.write(file_path, arcname)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[OK] ZIP 생성 완료: {zip_filename}")
        print(f"  파일 크기: {size_mb:.2f} MB")
        print(f"  저장 위치: {zip_path}")
        return True
    except Exception as e:
        print(f"[오류] ZIP 생성 실패: {e}")
        return False


def print_summary():
    """빌드 결과 요약"""
    print_section("빌드 완료 - 결과 요약")

    if not RELEASE_DIR.exists():
        print("  (release 폴더 없음)")
        return

    print("\n배포 파일 목록:")
    print("-" * 70)

    for zip_file in RELEASE_DIR.glob("*.zip"):
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"  {zip_file.name} ({size_mb:.2f} MB)")

    print("-" * 70)
    print(f"\n배포 경로: {RELEASE_DIR.absolute()}")
    print("\n사용 방법:")
    print("  1. ZIP 압축 해제")
    print("  2. label_studio_helper.exe 실행")
    print("  3. 첫 실행 시 PyTorch 자동 설치")


def main():
    """메인 함수"""
    print_section("Label Studio Helper 빌드 스크립트")
    print(f"프로젝트 경로: {PROJECT_ROOT}")
    print(f"빌드 모드: onedir (Embedded Python)")

    # 1. .spec 파일 확인
    if not SPEC_FILE.exists():
        print(f"\n[오류] {SPEC_FILE.name} 파일을 찾을 수 없습니다.")
        return 1

    # 2. 이전 빌드 정리
    clean_build_artifacts()

    # 3. PyInstaller 빌드
    if not build_with_pyinstaller():
        print("\n[실패] 빌드 과정에서 오류가 발생했습니다.")
        return 1

    # 4. ZIP 생성
    if not create_zip_distribution():
        print("\n[경고] ZIP 생성 실패.")
        return 1

    # 5. 결과 요약
    print_summary()

    # 6. release 폴더 열기
    try:
        subprocess.Popen(['explorer', str(RELEASE_DIR.absolute())])
        print(f"[OK] release 폴더를 열었습니다.")
    except Exception as e:
        print(f"[경고] release 폴더 열기 실패: {e}")

    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
