"""Top-Level Checkpointed Orchestrator for the EasyColab Cinema-Grade Short-Drama System."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .cloud_client import CLI_SCRIPT, EasyColabCloudClient
from .exporters import (
    export_html_storyboard_board,
    export_jianying_draft_bundle,
    render_final_cinema_mp4,
)
from .models import EpisodeBible, ProductionConfig
from .rules import lint_episode_bible
from .stages_prep import (
    run_stage1_showrunner,
    run_stage2_asset_factory,
    run_stage3_voice_and_dubbing,
)
from .stages_prod import run_stage4_keyframes, run_stage5_video_and_review


class CinemaDramaPipeline:
    """End-to-end 6-stage automated short-drama production state machine."""

    def __init__(self, config: ProductionConfig, cloud: Optional[EasyColabCloudClient] = None) -> None:
        self.cfg = config
        self.cfg.project_dir = Path(self.cfg.project_dir).expanduser().resolve()
        self.cfg.project_dir.mkdir(parents=True, exist_ok=True)
        self.cloud = cloud or EasyColabCloudClient(idle_minutes=30)

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            self.cloud.ensure_awake()

            # Stage 1: Story -> Production Bible + Cinema Shot Breakdown + Gate 0 Lint
            bible: EpisodeBible = run_stage1_showrunner(self.cfg, self.cloud)

            # Stage 2: Character Design Cards (User Face or Auto-Generated Face) + 3x3 Scene Grids + Prop Cards + Gate 1 VLM
            bible = run_stage2_asset_factory(self.cfg, bible, self.cloud)

            # Stage 3: Character Voice Seed Casting (Gate 2) + Clause-Level Drama 3 Dubbing (Gate 3)
            bible = run_stage3_voice_and_dubbing(self.cfg, bible, self.cloud)

            # Stage 4: Mechanical Reference Compilation + Multi-Scale Qwen-Edit-2511 Keyframes + Gate 4 VLM
            bible = run_stage4_keyframes(self.cfg, bible, self.cloud)

            # Stage 5: MiniMax-H3 10Eros-Max Short-Shot Rendering + Gate 5A Blind ASR + FFmpeg Signals + Gate 5B ContactSheet VLM
            bible = run_stage5_video_and_review(self.cfg, bible, self.cloud)

            # Free GPU VRAM cleanly before local Stage 6 post-production
            self.cloud.unload_llm()
            self.cloud.free_comfy_vram()

            # Stage 6: Per-Cut Cinema Assembly + Subtitles + HTML Workbench Board + Jianying Draft
            print("[Stage 6 · 后期精剪与多形态交付] 正在执行对白 J-Cut / 动作硬切合成、R128 响度母带与导出...", flush=True)
            final_info = render_final_cinema_mp4(self.cfg, bible)

            deliverables: Dict[str, Any] = {
                "ok": True,
                "project_dir": str(self.cfg.project_dir),
                "episode_bible": str(self.cfg.project_dir / "episode_bible.json"),
                "gate0_metrics": lint_episode_bible(bible, self.cfg.target_duration_s).metrics,
                "final_video": final_info,
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            if self.cfg.export_html_board:
                board_html = export_html_storyboard_board(self.cfg, bible, final_info)
                deliverables["html_board"] = str(board_html)
            if self.cfg.export_jianying:
                jy = export_jianying_draft_bundle(self.cfg, bible)
                deliverables["jianying_draft"] = jy

            report_path = self.cfg.project_dir / "production_manifest.json"
            report_path.write_text(json.dumps(deliverables, ensure_ascii=False, indent=2), encoding="utf-8")
            deliverables["production_manifest"] = str(report_path)
            print(f"[DramaStudio] 全流程交付完成！成片: {final_info['final_mp4']} ({final_info['shot_count']} 镜, {final_info['duration_s']}s)", flush=True)
            return deliverables
        finally:
            if not self.cfg.keep_alive:
                print("[DramaStudio] 触发完成自动休眠 (stop-all) 防烧 CCU...", flush=True)
                subprocess.run([sys.executable, str(CLI_SCRIPT), "stop-all"], check=False)


def run_drama_production(config: ProductionConfig) -> Dict[str, Any]:
    return CinemaDramaPipeline(config).run()
