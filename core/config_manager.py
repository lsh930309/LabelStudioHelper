#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
설정 관리 모듈
사용자 설정을 APPDATA에 JSON 파일로 저장/로드
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import os


class ConfigManager:
    """설정 관리 클래스 (Singleton)"""

    _instance = None

    def __init__(self):
        """설정 관리자 초기화"""
        # APPDATA 경로 설정
        if os.name == 'nt':  # Windows
            appdata = os.getenv('APPDATA')
            self.config_dir = Path(appdata) / 'LabelStudioHelper'
        else:
            # Linux/Mac
            home = Path.home()
            self.config_dir = home / '.labelstudiohelper'

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / 'settings.json'

        # 기본 설정
        self.default_config = {
            'output_directory': '',  # 출력 디렉토리 (빈 문자열 = 입력 파일과 같은 위치)
            'last_input_directory': '',  # 마지막 입력 파일 디렉토리
            'last_output_directory': '',  # 마지막 출력 파일 디렉토리
            'segmentation': {
                'static_threshold': 0.97,
                'min_static_duration_frames': 6,  # 프레임 단위 (기존 0.1초 = 6프레임 @ 60fps)
                'target_duration': 30.0,
                'use_gpu': True,
                'save_discarded': False,
                'feature_sample_rate': 1,  # N프레임마다 유사도 검사 (1=모든 프레임, 2=한 프레임 건너뛰기)
            }
        }

        # 설정 로드
        self.config = self.load_config()

    @classmethod
    def get_instance(cls):
        """Singleton 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def load_config(self) -> Dict[str, Any]:
        """
        설정 파일 로드

        Returns:
            설정 딕셔너리
        """
        if not self.config_file.exists():
            return self.default_config.copy()

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # 기본 설정과 병합 (누락된 키 보충)
            merged_config = self._merge_configs(self.default_config, loaded_config)
            return merged_config

        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패: {e}")
            return self.default_config.copy()

    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        기본 설정과 로드된 설정 병합

        Args:
            default: 기본 설정
            loaded: 로드된 설정

        Returns:
            병합된 설정
        """
        merged = default.copy()

        for key, value in loaded.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value

        return merged

    def save_config(self) -> bool:
        """
        설정을 파일에 저장

        Returns:
            성공 여부
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True

        except Exception as e:
            print(f"⚠️ 설정 파일 저장 실패: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 가져오기

        Args:
            key: 설정 키 (점으로 구분된 경로 지원, 예: 'segmentation.use_gpu')
            default: 기본값

        Returns:
            설정 값
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, save: bool = True):
        """
        설정 값 저장

        Args:
            key: 설정 키 (점으로 구분된 경로 지원)
            value: 설정 값
            save: 즉시 파일에 저장할지 여부
        """
        keys = key.split('.')
        config = self.config

        # 마지막 키 이전까지 탐색
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 마지막 키에 값 저장
        config[keys[-1]] = value

        if save:
            self.save_config()

    def get_output_directory(self, input_video_path: Optional[Path] = None) -> Path:
        """
        출력 디렉토리 가져오기

        Args:
            input_video_path: 입력 비디오 경로 (설정이 비어있으면 이 경로 기준으로)

        Returns:
            출력 디렉토리 Path
        """
        output_dir = self.get('output_directory', '')

        if output_dir:
            return Path(output_dir)
        elif input_video_path:
            # 설정이 없으면 입력 비디오와 같은 위치에 result_seg 폴더
            return Path(input_video_path).parent / 'result_seg'
        else:
            # 둘 다 없으면 현재 디렉토리
            return Path.cwd() / 'result_seg'

    def set_output_directory(self, directory: str):
        """출력 디렉토리 설정"""
        self.set('output_directory', directory)

    def get_last_input_directory(self) -> str:
        """마지막 입력 디렉토리 가져오기"""
        return self.get('last_input_directory', '')

    def set_last_input_directory(self, directory: str):
        """마지막 입력 디렉토리 설정"""
        self.set('last_input_directory', directory)

    def get_last_output_directory(self) -> str:
        """마지막 출력 디렉토리 가져오기"""
        return self.get('last_output_directory', '')

    def set_last_output_directory(self, directory: str):
        """마지막 출력 디렉토리 설정"""
        self.set('last_output_directory', directory)
