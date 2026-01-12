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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    min_static_duration: float,
    target_duration: float,
    use_gpu: bool,
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

        # PyTorch 설치 확인 (GPU 사용 시)
        if use_gpu:
            installed, _ = check_pytorch_installation()
            if not installed:
                return "❌ GPU 가속을 사용하려면 먼저 PyTorch를 설치해주세요.", None

        # 설정
        config = SegmentConfig(
            mode="custom",
            static_threshold=static_threshold,
            min_static_duration=min_static_duration,
            target_segment_duration=target_duration,
            feature_sample_rate=1,
            use_gpu=use_gpu,
            enable_visualization=True
        )

        # 출력 디렉토리 - gradio type="filepath"는 문자열 경로를 직접 반환
        video_path = Path(video_file) if isinstance(video_file, str) else Path(video_file.name)
        output_dir = video_path.parent / "result_seg"
        output_dir.mkdir(exist_ok=True)

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


def create_ui():
    """gradio UI 생성"""

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

                        gr.Markdown("### ⚙️ 설정")
                        static_threshold = gr.Slider(
                            minimum=0.8,
                            maximum=0.99,
                            value=0.97,
                            step=0.01,
                            label="정적 임계값",
                            info="높을수록 더 많이 제거됨"
                        )
                        min_static_duration = gr.Slider(
                            minimum=0.1,
                            maximum=5.0,
                            value=0.1,
                            step=0.1,
                            label="최소 정적 길이 (초)",
                            info="이보다 짧은 정적 구간은 무시"
                        )
                        target_duration = gr.Slider(
                            minimum=10,
                            maximum=120,
                            value=30,
                            step=5,
                            label="목표 세그먼트 길이 (초)"
                        )
                        use_gpu = gr.Checkbox(
                            label="GPU 가속 사용",
                            value=True,
                            info="PyTorch 설치 필요"
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
                segment_btn.click(
                    fn=segment_video_ui,
                    inputs=[
                        video_input,
                        static_threshold,
                        min_static_duration,
                        target_duration,
                        use_gpu
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
