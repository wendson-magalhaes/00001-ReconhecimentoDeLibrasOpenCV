import cv2
import mediapipe as mp
import json
import numpy as np
import time
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Carregamento dos dados
with open("gestos_dataset_avancado.json", "r") as f:
    dados = json.load(f)

X = [exemplo['coords'] for exemplo in dados]
y = [exemplo['label'] for exemplo in dados]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo KNN mais sensível
clf = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
clf.fit(X_scaled, y)

# Constantes
NEUTRAL_GESTURE = "neutro"
DEBOUNCE_TIME = 0.8
GESTURE_BUFFER_SIZE = 10  # Aumentado para mais estabilidade
INITIAL_CONFIDENCE_THRESHOLD = 0.6

# Variáveis globais
current_text = ""
last_registered_time = 0
last_confirmed_gesture = None
gesture_buffer = deque(maxlen=GESTURE_BUFFER_SIZE)
gesture_confidence_thresholds = {label: INITIAL_CONFIDENCE_THRESHOLD for label in clf.classes_}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    """Converte para coordenadas relativas ao pulso"""
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    coords = []
    for lm in landmarks.landmark:
        rel_x = lm.x - wrist[0]
        rel_y = lm.y - wrist[1]
        rel_z = lm.z - wrist[2]
        coords.extend([rel_x, rel_y, rel_z])
    return coords

def preprocess_landmarks(landmarks):
    """Adiciona vetores característicos entre pontos-chave"""
    normalized = normalize_landmarks(landmarks)
    
    # Calcula vetores entre pontos-chave
    key_vectors = []
    finger_tips = [4, 8, 12, 16, 20]  # Pontas dos dedos
    finger_joints = [2, 5, 9, 13, 17]  # Junções dos dedos
    
    for tip, joint in zip(finger_tips, finger_joints):
        tip_coord = np.array([landmarks.landmark[tip].x, landmarks.landmark[tip].y, landmarks.landmark[tip].z])
        joint_coord = np.array([landmarks.landmark[joint].x, landmarks.landmark[joint].y, landmarks.landmark[joint].z])
        vector = tip_coord - joint_coord
        key_vectors.extend(vector)
    
    return normalized + key_vectors

def reconhecer_gesto(landmarks):
    """Função de reconhecimento com suavização e limiares adaptativos"""
    processed = preprocess_landmarks(landmarks)
    
    try:
        coords_scaled = scaler.transform([processed])
        probs = clf.predict_proba(coords_scaled)[0]
        
        # Suavização das probabilidades
        smoothed_probs = (probs + 0.1) / (1 + 0.1 * len(probs))
        
        max_prob_idx = np.argmax(smoothed_probs)
        confidence = smoothed_probs[max_prob_idx]
        predicted_label = clf.classes_[max_prob_idx]
        
        # Usa limiar adaptativo para cada gesto
        return predicted_label if confidence > gesture_confidence_thresholds[predicted_label] else None
    except:
        return None

def draw_notepad(frame, text, width, camera_height):
    """Desenha o bloco de notas na parte inferior"""
    notepad_height = 200
    expanded_frame = np.zeros((camera_height + notepad_height, width, 3), dtype=np.uint8)
    expanded_frame[:camera_height, :] = frame

    # Bloco de notas
    cv2.rectangle(expanded_frame, (20, camera_height+10), (width-20, camera_height+notepad_height-10), 
                 (240, 240, 240), -1)
    cv2.rectangle(expanded_frame, (20, camera_height+10), (width-20, camera_height+notepad_height-10), 
                 (200, 200, 200), 2)

    y_offset = camera_height + 50
    for i, line in enumerate(text.split('\n')):
        cv2.putText(expanded_frame, line, (30, y_offset + i*40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    return expanded_frame

def main():
    global current_text, last_registered_time, last_confirmed_gesture, gesture_confidence_thresholds

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height = int(cap.get(3)), int(cap.get(4))

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            current_time = time.time()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            confirmed_gesture = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    gesture_detected = reconhecer_gesto(hand_landmarks)
                    gesture_buffer.append(gesture_detected)

                    # Sistema de votação ponderada
                    if len(gesture_buffer) == GESTURE_BUFFER_SIZE:
                        weights = np.linspace(0.5, 1.5, GESTURE_BUFFER_SIZE)  # Pesos progressivos
                        weighted_votes = {}
                        
                        for g, w in zip(gesture_buffer, weights):
                            if g is not None:
                                weighted_votes[g] = weighted_votes.get(g, 0) + w
                        
                        if weighted_votes:
                            confirmed_gesture = max(weighted_votes, key=weighted_votes.get)
                            # Requer 60% de concordância ponderada
                            if weighted_votes[confirmed_gesture] / sum(weights) > 0.6:
                                confirmed_gesture = confirmed_gesture
                            else:
                                confirmed_gesture = None

                    # Mostrar gesto atual
                    if confirmed_gesture:
                        cv2.putText(frame, f"Gesto: {confirmed_gesture}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

                    # Registra novo gesto se atender aos critérios
                    if (confirmed_gesture 
                        and confirmed_gesture != NEUTRAL_GESTURE 
                        and confirmed_gesture != last_confirmed_gesture 
                        and current_time - last_registered_time > DEBOUNCE_TIME):

                        current_text += confirmed_gesture
                        last_registered_time = current_time
                        last_confirmed_gesture = confirmed_gesture

                    # Reset quando volta para neutro
                    if confirmed_gesture == NEUTRAL_GESTURE:
                        last_confirmed_gesture = None

            else:
                cv2.putText(frame, "Mostre a mao para a camera", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            final_frame = draw_notepad(frame, current_text, width, height)
            cv2.putText(final_frame, "Espaco: Limpar texto | X: Sair", (10, height + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv2.imshow("Digitação por Gestos", final_frame)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('x'):
                break
            elif key & 0xFF == ord(' '):
                current_text = ""
                last_confirmed_gesture = None
                gesture_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()