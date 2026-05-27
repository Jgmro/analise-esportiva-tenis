
# Análise de Tênis - arrumar readme


## Introdução
Projeto voltado à análise esportiva preditiva aplicada ao tênis, utilizando visão computacional e aprendizado de máquina para extrair métricas de desempenho em partidas. A solução analisa vídeos para medir velocidade dos jogadores, velocidade dos golpes, quantidade de trocas de bola e índice de erros. O sistema utiliza o algoritmo YOLO para detecção de jogadores e bola, além de redes neurais convolucionais (CNNs) para identificação de pontos-chave da quadra. O projeto integra técnicas de detecção, rastreamento e análise esportiva, com foco em aplicações de analytics e IA no esporte.


## Modelos usados
* YOLO v8 - Detecção dos jogadores 
* YOLO    - para detecção de bolas de tênis
* CNN     - pontos-chave da quadra

* Modelo Treinado YOLOV5 : https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
* Modelo dos pontos-chave da quadra: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing

## Treinamento
* Detecção bola de tênis com YOLO: training/tennis_ball_detector_training.ipynb
* Pontos-Chave da quadra com Pytorch: training/tennis_court_keypoints_training.ipynb

## Requirements
* python3.8
* ultralytics
* pytroch
* pandas
* numpy 
* opencv
