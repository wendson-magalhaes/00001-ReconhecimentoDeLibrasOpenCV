import cv2
import mediapipe as mp
import json
import numpy as np
from collections import deque
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureCollector:
    def __init__(self):
        self.dataset = []
        self.current_gesture = None
        self.samples_collected = 0
        self.sample_buffer = deque(maxlen=30)  # Buffer para suavização
        self.last_sample_time = 0
        self.collection_active = False

    def normalize_landmarks(self, landmarks):
        """Normaliza as coordenadas relativas ao pulso"""
        wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        coords = []
        for lm in landmarks.landmark:
            rel_x = lm.x - wrist[0]
            rel_y = lm.y - wrist[1]
            rel_z = lm.z - wrist[2]
            coords.extend([rel_x, rel_y, rel_z])
        return coords

    def calculate_hand_size(self, landmarks):
        """Calcula o tamanho médio da mão para normalização adicional"""
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        return max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))

    def process_landmarks(self, landmarks):
        """Processa os landmarks com normalização e vetores característicos"""
        normalized = self.normalize_landmarks(landmarks)
        hand_size = self.calculate_hand_size(landmarks)
        
        # Adiciona vetores entre pontos-chave
        finger_tips = [4, 8, 12, 16, 20]
        finger_joints = [2, 5, 9, 13, 17]
        feature_vectors = []
        
        for tip, joint in zip(finger_tips, finger_joints):
            tip_coord = np.array([landmarks.landmark[tip].x, landmarks.landmark[tip].y, landmarks.landmark[tip].z])
            joint_coord = np.array([landmarks.landmark[joint].x, landmarks.landmark[joint].y, landmarks.landmark[joint].z])
            vector = (tip_coord - joint_coord) / hand_size
            feature_vectors.extend(vector)
        
        return normalized + feature_vectors

    def add_sample(self, landmarks):
        """Adiciona uma amostra ao buffer com verificação de qualidade"""
        processed = self.process_landmarks(landmarks)
        self.sample_buffer.append(processed)
        
        # Só permite amostras a cada 0.1 segundos para variedade
        current_time = time.time()
        if current_time - self.last_sample_time < 0.1:
            return False
        
        self.last_sample_time = current_time
        return True

    def save_gesture(self, gesture_name, num_samples=30):
        """Salva múltiplas amostras de um gesto"""
        if len(self.sample_buffer) < num_samples:
            print(f"Colete pelo menos {num_samples} amostras para o gesto '{gesture_name}'")
            return False
        
        # Pega as melhores amostras (remove outliers)
        samples = list(self.sample_buffer)[-num_samples:]
        
        for sample in samples:
            self.dataset.append({
                "coords": sample,
                "label": gesture_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        print(f"Gesto '{gesture_name}' salvo com {num_samples} amostras! Total no dataset: {len(self.dataset)}")
        self.sample_buffer.clear()
        return True

    def save_dataset(self, filename):
        """Salva o dataset completo"""
        with open(filename, "w") as f:
            json.dump(self.dataset, f, indent=2)
        print(f"Dataset salvo em '{filename}' com {len(self.dataset)} amostras")

def main():
    collector = GestureCollector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Feedback visual
            status_text = "Pressione:"
            instructions = [
                "1-9: Definir número de amostras",
                "s: Iniciar/Parar coleta",
                "x: Salvar e sair"
            ]

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if collector.collection_active:
                        if collector.add_sample(hand_landmarks):
                            collector.samples_collected += 1
                        
                        cv2.putText(image, f"Coletando: {collector.current_gesture}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image, f"Amostras: {collector.samples_collected}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Pressione 's' para começar a coletar", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Mostrar instruções
            y_offset = 90
            for i, line in enumerate([status_text] + instructions):
                cv2.putText(image, line, (10, y_offset + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Coletor de Dados Avançado', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if not collector.collection_active:
                    collector.current_gesture = input("Digite o nome do gesto: ")
                    collector.collection_active = True
                    collector.samples_collected = 0
                    print(f"Iniciando coleta para o gesto: {collector.current_gesture}")
                else:
                    if collector.save_gesture(collector.current_gesture):
                        collector.collection_active = False
            elif ord('1') <= key <= ord('9'):
                samples_to_collect = key - ord('0') * 10  # 1-9 -> 10-90 amostras
                print(f"Configurado para coletar {samples_to_collect} amostras por gesto")
            elif key == ord('x'):
                if collector.dataset:
                    collector.save_dataset("gestos_dataset_avancado.json")
                print("Saindo...")
                cap.release()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()