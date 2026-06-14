#!/usr/bin/env python3
import sys
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

# 프로젝트 src 경로 추가
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from avfusion.audio.gst_file_source import GstFileSource
from avfusion.dsp.mel import MelPatchExtractor
from avfusion.config import load_config

def visualize(media_path, output_path="mel_visual.png"):
    print(f"Analyzing: {media_path}")
    
    # 1. 아키텍처 설정 로드
    cfg = load_config()
    a = cfg.audio
    
    # 2. 프로젝트의 GstFileSource를 사용하여 MP4에서 오디오 추출
    # (torchaudio.load 대신 이 방식을 쓰는 것이 아키텍처와 일치함)
    src = GstFileSource(
        media_path, a.sample_rate, a.channels, 
        a.hop_samples, a.ring_capacity_hops,
        use_nvdec=False # 시각화용이므로 하드웨어 가속기 점유 최소화
    )
    
    # 3. MelPatchExtractor 설정
    extractor = MelPatchExtractor(
        capture_rate=a.sample_rate, 
        classifier_rate=cfg.classifier.classifier_sample_rate
    )
    
    print("Extracting audio via GStreamer...")
    src.start()
    
    try:
        # 패치가 준비될 때까지 혹은 파일이 끝날 때까지 오디오 읽기
        start_t = time.time()
        while not extractor.ready and not src.finished:
            chunk = src.read(timeout=0.1)
            if chunk:
                extractor.push_hop(chunk.samples)
            
            # 너무 오래 걸리면 중단 (안전 가드)
            if time.time() - start_t > 10:
                break
    finally:
        src.stop()

    # 4. 멜 패치 생성 및 시각화
    if extractor.ready:
        patch = extractor.patch() # [1, 1, 64, 96]
        patch_np = patch.squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(patch_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Log-Mel Energy')
        plt.title(f"Mel-Spectrogram (64x96): {os.path.basename(media_path)}")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Mel Bin (Frequency)")
        plt.savefig(output_path)
        print(f"Saved visualization to: {os.path.abspath(output_path)}")
    else:
        print("Error: Could not extract enough audio to create a full 0.96s patch.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("media", nargs="?", help="Path to .mp4 or .wav")
    args = ap.parse_args()
    
    media = args.media
    if not media:
        mat_dir = REPO_ROOT / "Edge-materials"
        if mat_dir.exists():
            clips = sorted(list(mat_dir.glob("*.mp4")))
            if clips:
                media = str(clips[0])
    
    if media and os.path.exists(media):
        visualize(media)
    else:
        print(f"No media file found at: {media}")
