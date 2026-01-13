#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Studio Helper - gradio UI
비디오 세그멘테이션 도구 (독립 실행형)

일반 사용자 권한으로 실행되며, PyTorch를 자동 설치합니다.
"""

import gradio as gr
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config Manager import
from core.config_manager import ConfigManager


def check_admin_rights() -> bool:
    """
    관리자 권한으로 실행 중인지 확인

    Returns:
        True: 관리자 권한으로 실행 중
        False: 일반 사용자 권한
    """
    if sys.platform == 'win32':
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    return False


def force_non_admin():
    """
    관리자 권한으로 실행 시 경고하고 재시작 요청
    일반 사용자 권한으로만 실행되도록 강제
    """
    if check_admin_rights():
        logger.error("❌ 이 앱은 관리자 권한으로 실행할 수 없습니다!")
        logger.error("   일반 사용자 권한으로 실행해주세요.")
        logger.error("   (우클릭 → '관리자 권한으로 실행' 사용 금지)")

        # gradio UI로 경고 표시
        if '--no-gui-warning' not in sys.argv:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "권한 오류",
                "Label Studio Helper는 관리자 권한으로 실행할 수 없습니다.\n\n"
                "일반 사용자 권한으로 실행해주세요.\n"
                "(우클릭 → '관리자 권한으로 실행' 사용 금지)\n\n"
                "이는 PyTorch 등 Add-on 설치 시 권한 충돌을 방지하기 위함입니다."
            )
            root.destroy()

        sys.exit(1)


def check_pytorch_installation() -> Tuple[bool, str]:
    """
    PyTorch 설치 여부 확인

    Returns:
        (설치됨, 버전 정보)
    """
    try:
        from core.pytorch_installer import PyTorchInstaller
        installer = PyTorchInstaller.get_instance()

        if installer.is_pytorch_installed():
            version_info = installer.get_installed_version()
            if version_info:
                pytorch_ver = version_info.get('pytorch', 'unknown')
                cuda_ver = version_info.get('cuda', 'unknown')
                return True, f"PyTorch {pytorch_ver} (CUDA {cuda_ver})"

        return False, "미설치"
    except Exception as e:
        logger.error(f"PyTorch 확인 중 오류: {e}")
        return False, "확인 실패"


def install_pytorch_ui(progress=gr.Progress()):
    """
    PyTorch 자동 설치 (gradio UI)

    Args:
        progress: gradio Progress 객체
    """
    try:
        from core.pytorch_installer import PyTorchInstaller

        progress(0, desc="CUDA 버전 감지 중...")
        installer = PyTorchInstaller.get_instance()
        cuda_version = installer.detect_cuda_version()

        if not cuda_version:
            return "❌ NVIDIA GPU를 찾을 수 없습니다. CPU 버전을 설치하시겠습니까?"

        progress(0.2, desc=f"CUDA {cuda_version} 감지됨")

        # 설치 진행
        def progress_callback(message: str):
            logger.info(message)

        progress(0.3, desc="PyTorch 다운로드 중...")
        success = installer.install_pytorch(cuda_version, progress_callback)

        if success:
            progress(1.0, desc="설치 완료!")
            installed, version_info = check_pytorch_installation()
            return f"✅ PyTorch 설치 완료!\n{version_info}"
        else:
            return "❌ PyTorch 설치 실패. 로그를 확인해주세요."

    except Exception as e:
        logger.error(f"PyTorch 설치 중 오류: {e}")
        return f"❌ 오류 발생: {e}"


def segment_video_ui(
    video_file,
    static_threshold: float,
    min_static_duration_frames: int,
    target_duration: float,
    feature_sample_rate: int,
    use_gpu: bool,
    save_discarded: bool,
    output_directory: str,
    progress=gr.Progress()
):
    """
    비디오 세그멘테이션 실행 (gradio UI)
    """
    if video_file is None:
        return "❌ 비디오 파일을 선택해주세요.", None

    try:
        from core.video_segmenter import VideoSegmenter, SegmentConfig
        from pathlib import Path
        import cv2

        # Config Manager 인스턴스
        config_manager = ConfigManager.get_instance()

        # 마지막 입력 디렉토리 저장
        video_path = Path(video_file) if isinstance(video_file, str) else Path(video_file.name)
        config_manager.set_last_input_directory(str(video_path.parent))

        # PyTorch 설치 확인 (GPU 사용 시)
        if use_gpu:
            installed, _ = check_pytorch_installation()
            if not installed:
                return "❌ GPU 가속을 사용하려면 먼저 PyTorch를 설치해주세요.", None

        # FPS 정보 가져오기 (프레임 → 초 변환용)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps == 0:
            fps = 30.0  # 기본값

        # 프레임 단위 → 초 단위 변환
        min_static_duration = min_static_duration_frames / fps

        # 설정
        config = SegmentConfig(
            mode="custom",
            static_threshold=static_threshold,
            min_static_duration=min_static_duration,
            target_segment_duration=target_duration,
            feature_sample_rate=feature_sample_rate,
            use_gpu=use_gpu,
            enable_visualization=True,
            save_discarded=save_discarded
        )

        # 출력 디렉토리 결정
        if output_directory and output_directory.strip():
            output_dir = Path(output_directory)
        else:
            # 설정에서 가져오기
            output_dir = config_manager.get_output_directory(video_path)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 마지막 출력 디렉토리 저장
        config_manager.set_last_output_directory(str(output_dir.parent))

        progress(0, desc="세그멘테이션 초기화 중...")
        segmenter = VideoSegmenter(config)

        # 진행률 콜백
        def progress_callback(current, total):
            progress(current / total, desc=f"프레임 분석 중... ({current}/{total})")

        progress(0.1, desc="세그먼트 탐지 중...")
        segments = segmenter.detect_segments(video_path, progress_callback)

        if not segments:
            return "❌ 유효한 세그먼트를 찾을 수 없습니다.", None

        progress(0.7, desc="세그먼트 비디오 생성 중...")
        saved_paths = segmenter.export_segments(video_path, segments, output_dir)

        progress(0.9, desc="메타데이터 저장 중...")
        segmenter.save_metadata(output_dir, video_path, segments)

        progress(1.0, desc="완료!")

        # 결과 메시지
        total_duration = sum(seg.duration for seg in segments)
        result_msg = f"""
✅ 세그멘테이션 완료!

📊 통계:
- 세그먼트 수: {len(saved_paths)}개
- 총 길이: {total_duration / 60:.1f}분
- 출력 폴더: {output_dir}

📁 생성된 파일:
{chr(10).join(f'  • {p.name}' for p in saved_paths[:5])}
{'  ...' if len(saved_paths) > 5 else ''}
"""

        # 시각화 그래프 반환
        graph_path = output_dir / 'similarity_graph.png'
        if graph_path.exists():
            return result_msg, str(graph_path)
        else:
            return result_msg, None

    except Exception as e:
        logger.error(f"세그멘테이션 중 오류: {e}", exc_info=True)
        return f"❌ 오류 발생: {e}", None


def open_explorer():
    """Windows 탐색기를 열어 사용자가 경로를 선택하도록 안내"""
    try:
        import subprocess
        from pathlib import Path

        config_manager = ConfigManager.get_instance()
        last_dir = config_manager.get_last_output_directory()

        # 마지막 디렉토리가 없으면 홈 디렉토리
        if not last_dir or not Path(last_dir).exists():
            last_dir = str(Path.home())

        # Windows 탐색기 열기
        subprocess.Popen(['explorer', last_dir])

        return f"📂 탐색기가 열렸습니다.\n경로를 복사하여 위 텍스트 박스에 붙여넣어주세요.\n\n현재 설정: {config_manager.get('output_directory', '(비어있음 - 입력 파일 위치 사용)')}"

    except Exception as e:
        logger.error(f"탐색기 열기 오류: {e}")
        return f"❌ 탐색기를 열 수 없습니다: {e}\n\n직접 경로를 입력해주세요."


def save_output_directory(directory: str):
    """출력 디렉토리 저장"""
    try:
        config_manager = ConfigManager.get_instance()

        if directory and directory.strip():
            directory = directory.strip()
            # 경로 유효성 검사
            path = Path(directory)
            if not path.exists():
                return f"⚠️ 경로가 존재하지 않습니다: {directory}\n\n계속 사용하시려면 디렉토리를 생성해주세요."

            config_manager.set_output_directory(directory)
            config_manager.set_last_output_directory(str(path.parent))
            return f"✅ 출력 디렉토리 저장 완료!\n\n{directory}"
        else:
            # 비어있으면 기본값 사용
            config_manager.set_output_directory('')
            return "ℹ️ 출력 디렉토리가 비워졌습니다.\n입력 파일과 같은 위치의 result_seg 폴더를 사용합니다."

    except Exception as e:
        logger.error(f"디렉토리 저장 중 오류: {e}")
        return f"❌ 오류 발생: {e}"


def create_ui():
    """gradio UI 생성"""

    # Config Manager 인스턴스
    config_manager = ConfigManager.get_instance()

    # 테마
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    )

    with gr.Blocks(theme=theme, title="Label Studio Helper") as app:
        gr.Markdown("""
        # 🎬 Label Studio Helper
        **비디오 세그멘테이션 도구** - AI 기반 자동 클립 분할

        > 일반 사용자 권한으로 실행됩니다.
        """)

        # 탭 구성
        with gr.Tabs():
            # 탭 1: 비디오 세그멘테이션
            with gr.Tab("🎥 비디오 세그멘테이션"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 입력")
                        video_input = gr.File(
                            label="비디오 파일",
                            file_types=["video"],
                            type="filepath"
                        )

                        gr.Markdown("### 📁 출력")
                        output_directory = gr.Textbox(
                            label="출력 디렉토리",
                            value=config_manager.get('output_directory', ''),
                            placeholder="비어있으면 입력 파일과 같은 위치의 result_seg 폴더",
                            interactive=True
                        )
                        with gr.Row():
                            open_explorer_btn = gr.Button("📂 탐색기 열기", size="sm")
                            save_output_btn = gr.Button("💾 저장", size="sm", variant="primary")

                        output_status = gr.Textbox(
                            label="출력 디렉토리 상태",
                            lines=3,
                            interactive=False,
                            visible=False
                        )

                        gr.Markdown("### ⚙️ 설정")
                        static_threshold = gr.Slider(
                            minimum=0.8,
                            maximum=0.99,
                            value=config_manager.get('segmentation.static_threshold', 0.95),
                            step=0.01,
                            label="정적 임계값",
                            info="높을수록 더 많이 제거됨"
                        )
                        min_static_duration_frames = gr.Slider(
                            minimum=1,
                            maximum=300,
                            value=config_manager.get('segmentation.min_static_duration_frames', 6),
                            step=1,
                            label="최소 정적 길이 (프레임)",
                            info="이보다 짧은 정적 구간은 무시 (예: 60fps 기준 6프레임 = 0.1초)"
                        )
                        target_duration = gr.Slider(
                            minimum=10,
                            maximum=120,
                            value=config_manager.get('segmentation.target_duration', 30),
                            step=5,
                            label="목표 세그먼트 길이 (초)"
                        )

                        with gr.Accordion("⚡ 고급 설정", open=False):
                            feature_sample_rate = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=config_manager.get('segmentation.feature_sample_rate', 1),
                                step=1,
                                label="프레임 샘플링 레이트",
                                info="N프레임마다 유사도 검사 (1=모든 프레임, 2=한 프레임 건너뛰기, 높을수록 빠르지만 정확도 감소)"
                            )

                        use_gpu = gr.Checkbox(
                            label="GPU 가속 사용",
                            value=config_manager.get('segmentation.use_gpu', True),
                            info="PyTorch 설치 필요"
                        )
                        save_discarded = gr.Checkbox(
                            label="채택되지 않은 구간도 저장 (else 폴더)",
                            value=config_manager.get('segmentation.save_discarded', False),
                            info="정적 구간 등 제외된 부분을 별도 저장"
                        )

                        segment_btn = gr.Button("🚀 세그멘테이션 시작", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 결과")
                        result_output = gr.Textbox(
                            label="실행 결과",
                            lines=15,
                            max_lines=20
                        )
                        graph_output = gr.Image(
                            label="유사도 그래프",
                            type="filepath"
                        )

                # 이벤트 연결
                open_explorer_btn.click(
                    fn=open_explorer,
                    outputs=output_status
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=output_status
                )

                save_output_btn.click(
                    fn=save_output_directory,
                    inputs=output_directory,
                    outputs=output_status
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=output_status
                )

                segment_btn.click(
                    fn=segment_video_ui,
                    inputs=[
                        video_input,
                        static_threshold,
                        min_static_duration_frames,
                        target_duration,
                        feature_sample_rate,
                        use_gpu,
                        save_discarded,
                        output_directory
                    ],
                    outputs=[result_output, graph_output]
                )

            # 탭 2: PyTorch 설정
            with gr.Tab("🔧 PyTorch 설정"):
                gr.Markdown("""
                ### PyTorch 관리
                GPU 가속을 사용하려면 PyTorch를 설치해야 합니다.
                """)

                with gr.Row():
                    pytorch_status = gr.Textbox(
                        label="설치 상태",
                        value=check_pytorch_installation()[1],
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 새로고침")

                install_btn = gr.Button("⬇️ PyTorch 설치 (최신 CUDA)", variant="primary")
                install_output = gr.Textbox(label="설치 로그", lines=10)

                # 이벤트
                refresh_btn.click(
                    fn=lambda: check_pytorch_installation()[1],
                    outputs=pytorch_status
                )
                install_btn.click(
                    fn=install_pytorch_ui,
                    outputs=install_output
                )

            # 탭 3: 정보
            with gr.Tab("ℹ️ 정보"):
                gr.Markdown("""
                ## Label Studio Helper v1.0

                ### 기능
                - 🎬 AI 기반 비디오 세그멘테이션
                - 🚀 GPU 가속 지원 (PyTorch)
                - 📊 시각화 그래프 생성

                ### 시스템 요구사항
                - Windows 10/11
                - NVIDIA GPU (GPU 가속 사용 시)
                - 10GB 이상의 디스크 공간

                ### 권한 안내
                - ✅ 일반 사용자 권한으로 실행됩니다
                - ❌ 관리자 권한으로 실행 금지
                - 📁 모든 데이터는 `%APPDATA%/LabelStudioHelper`에 저장됩니다

                ### License
                MIT License
                """)

    return app


def main():
    """메인 함수"""
    # 1. 관리자 권한 체크 (강제)
    force_non_admin()

    # 2. 로그
    logger.info("=" * 60)
    logger.info("Label Studio Helper 시작")
    logger.info(f"Python: {sys.version}")
    logger.info(f"일반 사용자 권한으로 실행 중 ✓")
    logger.info("=" * 60)

    # 3. PyTorch 상태 확인
    installed, version_info = check_pytorch_installation()
    if installed:
        logger.info(f"PyTorch: {version_info}")
    else:
        logger.warning("PyTorch: 미설치 (GPU 가속 사용 불가)")

    # 4. gradio UI 실행
    app = create_ui()

    app.launch(
        server_name="127.0.0.1",  # 로컬만 접근
        server_port=7860,
        share=False,  # 외부 공유 비활성화
        inbrowser=True,  # 자동으로 브라우저 열기
        quiet=False
    )


if __name__ == "__main__":
    main()
