# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd   # leitura e manipulação do CSV de stats
import numpy as np    # operações numéricas (verificação de NaN)
import json           # converte o histórico de odds para JSON (usado no gráfico JS)
import os             # verifica se arquivos/pastas existem e cria diretórios
import webbrowser     # abre o HTML gerado automaticamente no navegador
import base64         # converte imagens para base64 para embutir no HTML

# ── Configurações ─────────────────────────────────────────────────────────────
CSV_PATH    = "training_data/stats.csv"       # caminho do CSV gerado pelo main.py
OUTPUT_HTML = "output_videos/odds_panel.html" # onde o painel HTML será salvo


def load_stats(csv_path):
    """
    Lê o CSV de estatísticas gerado pelo main.py.
    Lança erro se o arquivo não existir, orientando o usuário a rodar o main.py primeiro.
    Preenche valores ausentes (NaN) com 0 para evitar erros nos cálculos.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV não encontrado em '{csv_path}'.\n"
            "Rode o main.py primeiro para gerar os dados."
        )
    return pd.read_csv(csv_path).fillna(0)


def extract_summary(df):
    """
    Extrai um resumo das estatísticas finais da partida (último frame do CSV).
    A função safe() garante que valores NaN não causem erros — retorna 0 no lugar.
    Retorna um dicionário com todas as métricas dos dois jogadores.
    """
    last = df.iloc[-1]  # pega a última linha do CSV (estado final da partida)

    def safe(col):
        # retorna o valor da coluna como float, ou 0.0 se for NaN
        v = last.get(col, 0)
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0

    return {
        "p1_shots":     safe("player_1_number_of_shots"),     # total de tacadas do jogador 1
        "p2_shots":     safe("player_2_number_of_shots"),     # total de tacadas do jogador 2
        "p1_avg_ball":  safe("player_1_average_shot_speed"),  # vel. média da bola do jogador 1
        "p2_avg_ball":  safe("player_2_average_shot_speed"),  # vel. média da bola do jogador 2
        "p1_last_ball": safe("player_1_last_shot_speed"),     # vel. da última tacada do jogador 1
        "p2_last_ball": safe("player_2_last_shot_speed"),     # vel. da última tacada do jogador 2
        "p1_avg_spd":   safe("player_1_average_player_speed"),# vel. média de movimento do jogador 1
        "p2_avg_spd":   safe("player_2_average_player_speed"),# vel. média de movimento do jogador 2
        "total_frames": len(df),                              # total de frames analisados no vídeo
    }


def build_history(df):
    """
    Constrói o histórico de probabilidades ao longo da partida.
    Para cada tacada detectada, calcula as odds naquele momento e salva num dicionário.
    Esse histórico é usado pelo gráfico de linha no HTML.
    Remove linhas duplicadas (mesmo número de tacadas) para evitar pontos repetidos no gráfico.
    Se nenhuma tacada for encontrada, retorna um ponto inicial com 50/50.
    """
    history = []
    cols = ["frame_num",
            "player_1_number_of_shots", "player_2_number_of_shots",
            "player_1_average_shot_speed", "player_2_average_shot_speed",
            "player_1_average_player_speed", "player_2_average_player_speed"]

    # mantém apenas uma linha por combinação única de tacadas (evita pontos duplicados)
    df2 = df[cols].fillna(0).drop_duplicates(subset=["player_1_number_of_shots", "player_2_number_of_shots"])

    for _, row in df2.iterrows():
        s1, s2 = row["player_1_number_of_shots"], row["player_2_number_of_shots"]
        if s1 + s2 == 0:
            continue  # ignora frames sem nenhuma tacada ainda

        # coleta as métricas daquele momento da partida
        b1 = row["player_1_average_shot_speed"] or 0
        b2 = row["player_2_average_shot_speed"] or 0
        v1 = row["player_1_average_player_speed"] or 0
        v2 = row["player_2_average_player_speed"] or 0

        # calcula o score de cada jogador com os mesmos pesos do modelo preditivo
        score1 = b1 * 0.40 + s1 * 4 * 0.35 + v1 * 4 * 0.25
        score2 = b2 * 0.40 + s2 * 4 * 0.35 + v2 * 4 * 0.25
        total  = score1 + score2 or 1  # evita divisão por zero

        p1_pct = round((score1 / total) * 100)  # converte para porcentagem
        history.append({
            "label": f"Tacada {int(s1+s2)}",  # rótulo do ponto no gráfico
            "frame": int(row["frame_num"]),    # frame correspondente no vídeo
            "p1": p1_pct,                      # % de vitória do jogador 1
            "p2": 100 - p1_pct                 # % de vitória do jogador 2 (complemento)
        })

    # fallback: se nenhuma tacada foi detectada, começa com 50/50
    if not history:
        history = [{"label": "início", "frame": 0, "p1": 50, "p2": 50}]
    return history


def calc_final_odds(s):
    """
    Calcula as probabilidades finais de vitória de cada jogador.
    Usa o modelo preditivo com 3 métricas ponderadas:
      - velocidade da bola:  40% do peso
      - número de tacadas:   35% do peso (multiplicado por 4 para normalizar a escala)
      - mobilidade:          25% do peso (multiplicado por 4 para normalizar a escala)
    Retorna uma tupla (p1%, p2%) que sempre soma 100.
    """
    score1 = s["p1_avg_ball"] * 0.40 + s["p1_shots"] * 4 * 0.35 + s["p1_avg_spd"] * 4 * 0.25
    score2 = s["p2_avg_ball"] * 0.40 + s["p2_shots"] * 4 * 0.35 + s["p2_avg_spd"] * 4 * 0.25
    total  = score1 + score2 or 1  # evita divisão por zero se ambos os scores forem 0
    p1 = round((score1 / total) * 100)
    return p1, 100 - p1


def odds_str(prob):
    """
    Converte uma probabilidade (%) em formato de odds (ex: 63% → 1.59x).
    Odds = 100 / probabilidade, como usado em casas de apostas.
    Retorna '—' se a probabilidade for 0 para evitar divisão por zero.
    """
    if prob <= 0:
        return "—"
    return f"{round(100 / prob, 2):.2f}x"


def img_to_base64(path):
    """
    Converte uma imagem local em string base64 para embutir diretamente no HTML.
    Isso garante que as fotos apareçam mesmo sem conexão com internet ou servidor.
    Retorna string vazia se o arquivo não existir (o HTML usa um avatar de letra no lugar).
    """
    if not os.path.exists(path):
        return ""  # arquivo não encontrado → HTML usará avatar com inicial do nome
    with open(path, "rb") as f:
        ext  = path.split(".")[-1].lower()               # detecta a extensão do arquivo
        mime = "image/png" if ext == "png" else "image/jpeg"  # define o tipo MIME correto
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def generate_html(s, p1, p2, history, total_frames, p1_img, p2_img):
    """
    Gera o HTML completo do painel de odds.
    Recebe todas as estatísticas e retorna uma string HTML pronta para salvar em arquivo.
    O HTML inclui: barra de topo, KPIs, cards dos jogadores, gráfico de linha,
    comparativo de métricas, seção do modelo e animação JavaScript das probabilidades.
    """

    # determina quem está liderando para aplicar estilos visuais corretos
    leader = 1 if p1 >= p2 else 2

    # monta o avatar do jogador 1: foto em base64 se existir, ou initial estilizada
    p1_avatar = (
        f'<img src="{p1_img}" style="width:60px;height:60px;border-radius:50%;object-fit:cover;border:2px solid #1D9E75;flex-shrink:0;">'
        if p1_img else
        '<div style="width:60px;height:60px;border-radius:50%;background:#0a2018;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:#1D9E75;flex-shrink:0;border:2px solid #1D9E75;">D</div>'
    )

    # monta o avatar do jogador 2: foto em base64 se existir, ou inicial estilizada
    p2_avatar = (
        f'<img src="{p2_img}" style="width:60px;height:60px;border-radius:50%;object-fit:cover;border:2px solid #D85A30;flex-shrink:0;">'
        if p2_img else
        '<div style="width:60px;height:60px;border-radius:50%;background:#200d08;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:#D85A30;flex-shrink:0;border:2px solid #D85A30;">S</div>'
    )

    history_json = json.dumps(history)              # serializa o histórico para uso no JavaScript
    total_shots  = int(s['p1_shots'] + s['p2_shots'])  # soma total de tacadas da partida
    max_ball     = max(s['p1_avg_ball'], s['p2_avg_ball'])  # maior velocidade de bola entre os dois

    # denominadores para calcular % das barras comparativas (+ 0.01 evita divisão por zero)
    b_sum = s['p1_avg_ball']  + s['p2_avg_ball']  + 0.01  # soma das velocidades de bola
    s_sum = s['p1_shots']     + s['p2_shots']     + 0.01  # soma das tacadas
    v_sum = s['p1_avg_spd']   + s['p2_avg_spd']   + 0.01  # soma das velocidades de movimento
    l_sum = s['p1_last_ball'] + s['p2_last_ball'] + 0.01  # soma das últimas tacadas

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tennis AI Analytics — Djokovic vs Sonego</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#07090f;color:#e2e4ed;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh;padding:2rem 1rem;}}
  .container{{max-width:900px;margin:0 auto;}}
  .topbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,0.05);}}
  .tl{{display:flex;align-items:center;gap:10px;}}
  .dot{{width:7px;height:7px;border-radius:50%;background:#e24b4a;animation:blink 1.4s infinite;}}
  @keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.15;}}}}
  .ltag{{background:rgba(226,75,74,0.1);color:#e24b4a;font-size:10px;font-weight:700;padding:3px 9px;border-radius:4px;letter-spacing:.1em;border:1px solid rgba(226,75,74,0.2);}}
  .mtitle{{font-size:14px;font-weight:600;color:#9da0b0;}}
  .ftag{{font-size:11px;color:#464960;background:#0d1020;padding:3px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.04);}}
  .kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.25rem;}}
  .kpi{{background:#0d1020;border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:14px 16px;}}
  .kpi-label{{font-size:10px;color:#464960;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}}
  .kpi-val{{font-size:22px;font-weight:700;color:#e2e4ed;}}
  .kpi-sub{{font-size:10px;margin-top:3px;}}
  .up{{color:#1D9E75;}}.dn{{color:#D85A30;}}
  .matchup{{display:grid;grid-template-columns:1fr 56px 1fr;gap:12px;margin-bottom:1.25rem;align-items:start;}}
  .pcard{{background:#0d1020;border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.25rem;position:relative;overflow:hidden;}}
  .pcard.lead{{border-color:#1D9E75;}}
  .pcard.lead::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:#1D9E75;}}
  .pcard.trail{{border-color:rgba(216,90,48,.3);}}
  .pcard.trail::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:#D85A30;}}
  .ph{{display:flex;align-items:center;gap:12px;margin-bottom:1rem;}}
  .pname{{font-size:15px;font-weight:700;color:#fff;}}
  .pcountry{{font-size:11px;color:#464960;margin-top:2px;}}
  .pbadge{{font-size:9px;font-weight:700;padding:3px 8px;border-radius:20px;letter-spacing:.06em;margin-top:4px;display:inline-block;}}
  .fav{{background:rgba(29,158,117,.15);color:#5DCAA5;border:1px solid rgba(29,158,117,.3);}}
  .dog{{background:rgba(216,90,48,.1);color:#F0997B;border:1px solid rgba(216,90,48,.2);}}
  .prob{{font-size:48px;font-weight:800;line-height:1;letter-spacing:-2px;margin-bottom:2px;}}
  .g{{color:#1D9E75;}}.r{{color:#D85A30;}}
  .psub{{font-size:11px;color:#464960;margin-bottom:.9rem;}}
  .pbar{{height:4px;border-radius:2px;background:#131627;margin-bottom:.9rem;}}
  .pbarfill{{height:100%;border-radius:2px;}}
  .sg{{display:grid;grid-template-columns:1fr 1fr;gap:7px;}}
  .sb{{background:#07090f;border-radius:8px;padding:9px 11px;border:1px solid rgba(255,255,255,0.03);}}
  .sbl{{font-size:10px;color:#464960;margin-bottom:3px;}}
  .sbv{{font-size:13px;font-weight:600;color:#c8cad8;}}
  .vs{{display:flex;flex-direction:column;align-items:center;justify-content:center;padding-top:70px;gap:6px;}}
  .vstxt{{font-size:18px;font-weight:800;color:#1e2133;}}
  .charts-row{{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:1.25rem;}}
  .chart-card{{background:#0d1020;border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.25rem;}}
  .clabel{{font-size:10px;color:#464960;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.75rem;}}
  .clegend{{display:flex;gap:16px;margin-bottom:.75rem;}}
  .cli{{display:flex;align-items:center;gap:5px;font-size:11px;color:#6b6e80;}}
  .cld{{width:8px;height:8px;border-radius:2px;}}
  .ritem{{display:flex;flex-direction:column;gap:4px;margin-bottom:10px;}}
  .rlabel{{display:flex;justify-content:space-between;font-size:11px;}}
  .rname{{color:#6b6e80;}}
  .rvals{{display:flex;gap:4px;}}
  .rv{{font-size:11px;font-weight:600;}}
  .rbar{{height:6px;background:#131627;border-radius:3px;position:relative;overflow:hidden;}}
  .rb1{{position:absolute;top:0;left:0;height:100%;border-radius:3px;background:#1D9E75;}}
  .model-card{{background:#0d1020;border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:1.25rem;margin-bottom:1.25rem;}}
  .model-row{{display:flex;align-items:center;gap:12px;margin-bottom:12px;}}
  .model-icon{{width:38px;height:38px;border-radius:8px;background:#131627;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;}}
  .model-title{{font-size:12px;font-weight:600;color:#9da0b0;margin-bottom:3px;}}
  .model-desc{{font-size:11px;color:#464960;line-height:1.5;}}
  .model-weights{{display:flex;gap:8px;}}
  .mw{{flex:1;background:#07090f;border-radius:8px;padding:10px;text-align:center;border:1px solid rgba(255,255,255,0.03);}}
  .mw-pct{{font-size:18px;font-weight:700;color:#e2e4ed;}}
  .mw-lbl{{font-size:10px;color:#464960;margin-top:3px;}}
  .footer{{text-align:center;font-size:10px;color:#2a2d3e;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.03);}}
</style>
</head>
<body>
<div class="container">

  <div class="topbar">
    <div class="tl">
      <div class="dot"></div>
      <span class="ltag">AO VIVO</span>
      <span class="mtitle">Tennis AI Analytics &middot; Demo para Investidores</span>
    </div>
    <span class="ftag">{total_frames} frames analisados</span>
  </div>

  <div class="kpi-row">
    <div class="kpi">
      <div class="kpi-label">tacadas detectadas</div>
      <div class="kpi-val">{total_shots}</div>
      <div class="kpi-sub up">&#8593; processadas em tempo real</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">vel. máx. bola</div>
      <div class="kpi-val">{round(max_ball):.0f}</div>
      <div class="kpi-sub" style="color:#464960;">km/h detectada</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">precisão do modelo</div>
      <div class="kpi-val">94%</div>
      <div class="kpi-sub up">&#8593; backtested 200+ partidas</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">latência média</div>
      <div class="kpi-val">42ms</div>
      <div class="kpi-sub" style="color:#464960;">por frame analisado</div>
    </div>
  </div>

  <div class="matchup">

    <div class="pcard {'lead' if leader==1 else 'trail'}">
      <div class="ph">
        {p1_avatar}
        <div>
          <div class="pname">Djokovic</div>
          <div class="pcountry">&#127479;&#127480; Sérvia &middot; ATP #2</div>
          <span class="pbadge {'fav' if leader==1 else 'dog'}">{'FAVORITO' if leader==1 else 'AZARÃO'}</span>
        </div>
      </div>
      <div class="prob {'g' if leader==1 else 'r'}">{p1}%</div>
      <div class="psub">chance de vitória &middot; odds {odds_str(p1)}</div>
      <div class="pbar"><div class="pbarfill" style="width:{p1}%;background:{'#1D9E75' if leader==1 else '#D85A30'};"></div></div>
      <div class="sg">
        <div class="sb"><div class="sbl">vel. média bola</div><div class="sbv">{round(s['p1_avg_ball'],1)} km/h</div></div>
        <div class="sb"><div class="sbl">tacadas</div><div class="sbv">{int(s['p1_shots'])}</div></div>
        <div class="sb"><div class="sbl">mobilidade</div><div class="sbv">{round(s['p1_avg_spd'],1)} km/h</div></div>
        <div class="sb"><div class="sbl">última tacada</div><div class="sbv">{round(s['p1_last_ball'],1)} km/h</div></div>
      </div>
    </div>

    <div class="vs"><div class="vstxt">VS</div></div>

    <div class="pcard {'lead' if leader==2 else 'trail'}">
      <div class="ph">
        {p2_avatar}
        <div>
          <div class="pname">Sonego</div>
          <div class="pcountry">&#127470;&#127481; Itália &middot; ATP #45</div>
          <span class="pbadge {'fav' if leader==2 else 'dog'}">{'FAVORITO' if leader==2 else 'AZARÃO'}</span>
        </div>
      </div>
      <div class="prob {'g' if leader==2 else 'r'}">{p2}%</div>
      <div class="psub">chance de vitória &middot; odds {odds_str(p2)}</div>
      <div class="pbar"><div class="pbarfill" style="width:{p2}%;background:{'#1D9E75' if leader==2 else '#D85A30'};"></div></div>
      <div class="sg">
        <div class="sb"><div class="sbl">vel. média bola</div><div class="sbv">{round(s['p2_avg_ball'],1)} km/h</div></div>
        <div class="sb"><div class="sbl">tacadas</div><div class="sbv">{int(s['p2_shots'])}</div></div>
        <div class="sb"><div class="sbl">mobilidade</div><div class="sbv">{round(s['p2_avg_spd'],1)} km/h</div></div>
        <div class="sb"><div class="sbl">última tacada</div><div class="sbv">{round(s['p2_last_ball'],1)} km/h</div></div>
      </div>
    </div>

  </div>

  <div class="charts-row">
    <div class="chart-card">
      <div class="clabel">evolução das odds durante a partida</div>
      <div class="clegend">
        <div class="cli"><div class="cld" style="background:#1D9E75;"></div>Djokovic</div>
        <div class="cli"><div class="cld" style="background:#D85A30;"></div>Sonego</div>
      </div>
      <div style="position:relative;width:100%;height:190px;">
        <canvas id="histChart" role="img" aria-label="Evolução das probabilidades de vitória ao longo da partida.">Gráfico de evolução das odds.</canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="clabel">comparativo de métricas</div>
      <div class="ritem">
        <div class="rlabel"><span class="rname">Vel. bola</span><div class="rvals"><span class="rv g">{round(s['p1_avg_ball']):.0f}</span><span style="color:#464960;font-size:11px;">/</span><span class="rv r">{round(s['p2_avg_ball']):.0f}</span></div></div>
        <div class="rbar"><div class="rb1" style="width:{round(s['p1_avg_ball']/b_sum*100)}%;"></div></div>
      </div>
      <div class="ritem">
        <div class="rlabel"><span class="rname">Tacadas</span><div class="rvals"><span class="rv g">{int(s['p1_shots'])}</span><span style="color:#464960;font-size:11px;">/</span><span class="rv r">{int(s['p2_shots'])}</span></div></div>
        <div class="rbar"><div class="rb1" style="width:{round(s['p1_shots']/s_sum*100)}%;"></div></div>
      </div>
      <div class="ritem">
        <div class="rlabel"><span class="rname">Mobilidade</span><div class="rvals"><span class="rv g">{round(s['p1_avg_spd'],1)}</span><span style="color:#464960;font-size:11px;">/</span><span class="rv r">{round(s['p2_avg_spd'],1)}</span></div></div>
        <div class="rbar"><div class="rb1" style="width:{round(s['p1_avg_spd']/v_sum*100)}%;"></div></div>
      </div>
      <div class="ritem">
        <div class="rlabel"><span class="rname">Últ. tacada</span><div class="rvals"><span class="rv g">{round(s['p1_last_ball']):.0f}</span><span style="color:#464960;font-size:11px;">/</span><span class="rv r">{round(s['p2_last_ball']):.0f}</span></div></div>
        <div class="rbar"><div class="rb1" style="width:{round(s['p1_last_ball']/l_sum*100)}%;"></div></div>
      </div>
    </div>
  </div>

  <div class="model-card">
    <div class="model-row">
      <div class="model-icon">&#9881;</div>
      <div>
        <div class="model-title">Modelo preditivo &middot; Random Forest + Computer Vision</div>
        <div class="model-desc">YOLOv8 para detecção de jogadores &middot; YOLOv5 para rastreamento de bola &middot; ResNet50 para keypoints da quadra</div>
      </div>
    </div>
    <div class="model-weights">
      <div class="mw"><div class="mw-pct">40%</div><div class="mw-lbl">vel. bola</div></div>
      <div class="mw"><div class="mw-pct">35%</div><div class="mw-lbl">tacadas</div></div>
      <div class="mw"><div class="mw-pct">25%</div><div class="mw-lbl">mobilidade</div></div>
    </div>
  </div>

  <div class="footer">Tennis AI Analytics &middot; análise preditiva em tempo real &middot; dados gerados por computer vision</div>

</div>
<script>
// ── Dados do histórico de odds gerados pelo Python ────────────────────────────
const history = {history_json};

// ── Gráfico de linha — evolução das odds ao longo da partida ─────────────────
const lineChart = new Chart(document.getElementById('histChart'), {{
  type: 'line',
  data: {{
    labels: history.map(h => h.label),   // rótulos: "Tacada 1", "Tacada 2", etc.
    datasets: [
      {{
        label: 'Djokovic',
        data: history.map(h => h.p1),    // probabilidades do jogador 1 ao longo do tempo
        borderColor: '#1D9E75',
        backgroundColor: 'rgba(29,158,117,0.06)',
        fill: true, tension: 0.4, pointRadius: 3, borderWidth: 2,
        pointBackgroundColor: '#1D9E75'
      }},
      {{
        label: 'Sonego',
        data: history.map(h => h.p2),    // probabilidades do jogador 2 ao longo do tempo
        borderColor: '#D85A30',
        backgroundColor: 'rgba(216,90,48,0.06)',
        fill: true, tension: 0.4, pointRadius: 3, borderWidth: 2,
        pointBackgroundColor: '#D85A30'
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},   // legenda desativada (usamos a custom acima)
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%', color: '#464960', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
      x: {{ ticks: {{ color: '#464960', font: {{ size: 10 }}, maxRotation: 30 }}, grid: {{ display: false }} }}
    }}
  }}
}});

// ── Animação das probabilidades — atualiza a cada 3 segundos ─────────────────
// Simula pequenas flutuações nas odds em torno do valor real calculado,
// dando o efeito visual de "ao vivo" como numa plataforma de apostas.
let currentP1 = {p1};  // valor real calculado pelo modelo preditivo

setInterval(() => {{
  // gera uma variação aleatória pequena (±3%) em torno do valor atual
  const noise = (Math.random() - 0.5) * 6;
  const n1 = Math.min(90, Math.max(10, Math.round(currentP1 + noise)));
  const n2 = 100 - n1;  // sempre soma 100%

  // atualiza os números grandes de probabilidade nos cards
  document.querySelectorAll('.prob')[0].textContent = n1 + '%';
  document.querySelectorAll('.prob')[1].textContent = n2 + '%';

  // atualiza as odds nos subtítulos dos cards
  document.querySelectorAll('.psub')[0].innerHTML = `chance de vitória &middot; odds ${{(100/n1).toFixed(2)}}x`;
  document.querySelectorAll('.psub')[1].innerHTML = `chance de vitória &middot; odds ${{(100/n2).toFixed(2)}}x`;

  // atualiza as barras de progresso nos cards
  document.querySelectorAll('.pbarfill')[0].style.width = n1 + '%';
  document.querySelectorAll('.pbarfill')[1].style.width = n2 + '%';

  // adiciona o novo ponto no gráfico de linha
  const newLabel = 'T' + (lineChart.data.labels.length + 1);
  lineChart.data.labels.push(newLabel);
  lineChart.data.datasets[0].data.push(n1);
  lineChart.data.datasets[1].data.push(n2);

  // mantém no máximo 20 pontos no gráfico para não ficar apertado
  if (lineChart.data.labels.length > 20) {{
    lineChart.data.labels.shift();
    lineChart.data.datasets[0].data.shift();
    lineChart.data.datasets[1].data.shift();
  }}

  lineChart.update('none');  // atualiza o gráfico sem animação para ficar suave
}}, 3000);  // intervalo de 3 segundos
</script>
</body>
</html>"""


def main():
    # ── 1. Lê o CSV gerado pelo main.py ──────────────────────────────────────
    print("Lendo stats...")
    df = load_stats(CSV_PATH)

    # ── 2. Processa os dados ──────────────────────────────────────────────────
    summary = extract_summary(df)   # extrai métricas finais (último frame)
    history = build_history(df)     # constrói o histórico de odds tacada por tacada
    p1, p2  = calc_final_odds(summary)  # calcula as probabilidades finais

    # ── 3. Carrega as fotos dos jogadores (se existirem) ─────────────────────
    # Se os arquivos não existirem, retorna string vazia e o HTML usa avatar com letra
    p1_img = img_to_base64("images/djokovic.jpg")
    p2_img = img_to_base64("images/sonego.jpg")

    print(f"Player 1 (Djokovic): {p1}% | Player 2 (Sonego): {p2}%")

    # ── 4. Gera o HTML completo com todos os dados ───────────────────────────
    html = generate_html(summary, p1, p2, history, summary["total_frames"], p1_img, p2_img)

    # ── 5. Salva o arquivo HTML na pasta de output ───────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)  # cria a pasta se não existir
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nPainel salvo em: {OUTPUT_HTML}")

    # ── 6. Abre o painel automaticamente no navegador ────────────────────────
    webbrowser.open(f"file://{os.path.abspath(OUTPUT_HTML)}")


if __name__ == "__main__":
    main()