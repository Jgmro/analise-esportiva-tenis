"""
shot_analysis.py
────────────────
Script independente que:
1. Lê o vídeo
2. Detecta jogadores e bola
3. Conecta cada tacada ao jogador mais próximo da bola
4. Gera um gráfico PNG com a % de tacadas de cada jogador
"""

import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import measure_distance, get_center_of_bbox


# ─────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────
INPUT_VIDEO     = "input_videos/input_video.mp4"
OUTPUT_CHART    = "output_videos/shot_analysis.png"
PLAYER_MODEL    = "yolov8x"
BALL_MODEL      = "models/yolo5_last.pt"
PLAYER_STUB     = "tracker_stubs/player_detections.pkl"
BALL_STUB       = "tracker_stubs/ball_detections.pkl"
READ_FROM_STUB  = True   # True = usa cache se existir, False = roda detecção do zero


# ─────────────────────────────────────────────
# 1. Ler vídeo
# ─────────────────────────────────────────────
def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─────────────────────────────────────────────
# 2. Detecção de jogadores
# ─────────────────────────────────────────────
def detect_players(frames, read_from_stub=True, stub_path=None):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        print("  Carregando detecções de jogadores do cache...")
        with open(stub_path, 'rb') as f:
            return pickle.load(f)

    print("  Rodando detecção de jogadores (pode demorar)...")
    model = YOLO(PLAYER_MODEL)
    detections = []
    for frame in frames:
        results = model.track(frame, persist=True)[0]
        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            cls_name = results.names[int(box.cls.tolist()[0])]
            if cls_name == "person":
                player_dict[track_id] = box.xyxy.tolist()[0]
        detections.append(player_dict)

    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(detections, f)

    return detections


# ─────────────────────────────────────────────
# 3. Detecção de bola
# ─────────────────────────────────────────────
def detect_ball(frames, read_from_stub=True, stub_path=None):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        print("  Carregando detecções de bola do cache...")
        with open(stub_path, 'rb') as f:
            return pickle.load(f)

    print("  Rodando detecção de bola (pode demorar)...")
    model = YOLO(BALL_MODEL)
    detections = []
    for frame in frames:
        results = model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            ball_dict[1] = box.xyxy.tolist()[0]
        detections.append(ball_dict)

    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(detections, f)

    return detections


# ─────────────────────────────────────────────
# 4. Interpolar posições da bola
# ─────────────────────────────────────────────
def interpolate_ball(ball_detections):
    positions = [x.get(1, []) for x in ball_detections]
    df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate().bfill()
    return [{1: x} for x in df.to_numpy().tolist()]


# ─────────────────────────────────────────────
# 5. Detectar frames de tacada
# ─────────────────────────────────────────────
def get_ball_shot_frames(ball_detections):
    positions = [x.get(1, []) for x in ball_detections]
    df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
    df['ball_hit'] = 0
    df['mid_y'] = (df['y1'] + df['y2']) / 2
    df['mid_y_rolling'] = df['mid_y'].rolling(window=5, min_periods=1).mean()
    df['delta_y'] = df['mid_y_rolling'].diff()

    min_frames = 25
    for i in range(1, len(df) - int(min_frames * 1.2)):
        neg = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
        pos = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0
        if neg or pos:
            count = 0
            for cf in range(i+1, i + int(min_frames * 1.2) + 1):
                if neg and df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[cf] < 0:
                    count += 1
                elif pos and df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[cf] > 0:
                    count += 1
            if count > min_frames - 1:
                df.iloc[i, df.columns.get_loc('ball_hit')] = 1

    return df[df['ball_hit'] == 1].index.tolist()


# ─────────────────────────────────────────────
# 6. Escolher os 2 jogadores principais
# ─────────────────────────────────────────────
def choose_main_players(player_detections):
    shot_counts = {}
    for frame_dict in player_detections:
        for pid in frame_dict:
            shot_counts[pid] = shot_counts.get(pid, 0) + 1
    sorted_players = sorted(shot_counts.items(), key=lambda x: -x[1])
    return [p[0] for p in sorted_players[:2]]


# ─────────────────────────────────────────────
# 7. Conectar tacada → jogador mais próximo
# ─────────────────────────────────────────────
def assign_shots_to_players(ball_shot_frames, ball_detections, player_detections, main_players):
    shot_counts = {pid: 0 for pid in main_players}
    shot_log = []  # (frame, player_id, ball_pos, player_pos)

    for frame_idx in ball_shot_frames:
        if frame_idx >= len(ball_detections) or frame_idx >= len(player_detections):
            continue

        ball = ball_detections[frame_idx].get(1)
        if not ball:
            continue
        ball_center = ((ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2)

        players = player_detections[frame_idx]
        best_pid, best_dist = None, float('inf')

        for pid in main_players:
            if pid not in players:
                continue
            p_center = get_center_of_bbox(players[pid])
            dist = measure_distance(p_center, ball_center)
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_pid is not None:
            shot_counts[best_pid] += 1
            shot_log.append({
                'frame': frame_idx,
                'player': best_pid,
                'ball_x': ball_center[0],
                'ball_y': ball_center[1],
                'player_x': get_center_of_bbox(players[best_pid])[0],
                'player_y': get_center_of_bbox(players[best_pid])[1],
            })

    return shot_counts, shot_log


# ─────────────────────────────────────────────
# 8. Gerar gráfico
# ─────────────────────────────────────────────
def generate_chart(shot_counts, shot_log, main_players, total_frames):
    total_shots = sum(shot_counts.values())
    if total_shots == 0:
        print("Nenhuma tacada detectada. Verifique o vídeo.")
        return

    player_labels = {main_players[0]: "Player 1", main_players[1]: "Player 2"}
    colors = {main_players[0]: "#3498db", main_players[1]: "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Gráfico 1: Pizza com % de tacadas ──
    ax1 = axes[0]
    sizes  = [shot_counts[p] for p in main_players]
    clrs   = [colors[p] for p in main_players]
    labels = [f"{player_labels[p]}\n{shot_counts[p]} tacadas" for p in main_players]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=clrs,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 11},
        wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight('bold')
    ax1.set_title("% de Tacadas", color='white', fontsize=14, pad=15)



    # ── Gráfico 2: Barras de tacadas por jogador ──
    ax2 = axes[1]
    pids  = list(main_players)
    vals  = [shot_counts[p] for p in pids]
    clrs2 = [colors[p] for p in pids]
    bars  = ax2.bar([player_labels[p] for p in pids], vals, color=clrs2,
                    edgecolor='#1a1a2e', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha='center', va='bottom', color='white', fontsize=13, fontweight='bold')
    ax2.set_title("Total de Tacadas", color='white', fontsize=14)
    ax2.set_ylabel("Número de tacadas", color='white')
    ax2.yaxis.label.set_color('white')
    ax2.set_ylim(0, max(vals) * 1.25)

    plt.suptitle("Análise de Tacadas — Tennis Analysis",
                 color='white', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_CHART), exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nGráfico salvo em: {OUTPUT_CHART}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("1. Lendo vídeo...")
    frames = read_video(INPUT_VIDEO)
    print(f"   {len(frames)} frames carregados.")

    print("2. Detectando jogadores...")
    player_detections = detect_players(frames, READ_FROM_STUB, PLAYER_STUB)

    print("3. Detectando bola...")
    ball_detections = detect_ball(frames, READ_FROM_STUB, BALL_STUB)
    ball_detections = interpolate_ball(ball_detections)

    print("4. Identificando tacadas...")
    ball_shot_frames = get_ball_shot_frames(ball_detections)
    print(f"   {len(ball_shot_frames)} tacadas detectadas.")

    print("5. Escolhendo jogadores principais...")
    main_players = choose_main_players(player_detections)
    print(f"   Jogadores: {main_players}")

    print("6. Conectando tacadas aos jogadores...")
    shot_counts, shot_log = assign_shots_to_players(
        ball_shot_frames, ball_detections, player_detections, main_players)

    total = sum(shot_counts.values())
    print(f"\n── Resultado ──────────────────────")
    for pid, count in shot_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"   Player {main_players.index(pid)+1} (ID {pid}): {count} tacadas ({pct:.1f}%)")
    print(f"   Total: {total} tacadas")
    print(f"───────────────────────────────────")

    print("\n7. Gerando gráfico...")
    generate_chart(shot_counts, shot_log, main_players, len(frames))
    print("\nPronto!")


if __name__ == "__main__":
    main()
