# Reconhecimento de Gestos com Mãos usando MediaPipe e Random Forest

Este projeto implementa um sistema de reconhecimento de gestos das mãos em tempo real utilizando webcam, com as seguintes tecnologias:

- **MediaPipe** para detecção de mãos e extração de landmarks (pontos de articulação)
- **scikit-learn** para treinamento e predição de gestos usando Random Forest
- **OpenCV** para captura de vídeo, exibição da interface gráfica e escrita do texto reconhecido

---

## Funcionalidades

- Captura os pontos da mão pela webcam em tempo real.
- Classifica o gesto da mão baseado em um modelo treinado (Random Forest).
- Usa buffer de predições para aumentar a confiabilidade do reconhecimento.
- Exibe o texto dos gestos reconhecidos na tela, com um bloco de notas visual.
- Permite limpar o texto com a tecla espaço e sair com a tecla "x".

---

## Estrutura do projeto

- `gestos_dataset.json`: arquivo JSON contendo as amostras coletadas (coordenadas + label).
- `main.py`: código principal que carrega o dataset, treina o modelo e faz o reconhecimento em tempo real.
- `coletar_dados.py` (opcional): script para coletar e salvar novos dados de gestos via webcam.

---

## Requisitos

- Python 3.7+
- Bibliotecas Python:

```bash
pip install opencv-python mediapipe scikit-learn numpy
