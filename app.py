import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import time

# Définition du modèle CNN
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Premier bloc
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Deuxième bloc
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Troisième bloc
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Dictionnaire des émotions et leurs messages associés
emotion_dict = {
    0: {"name": "Colère", "message": "Respirez profondément et prenez un moment pour vous calmer."},
    1: {"name": "Mépris", "message": "Essayez de voir les choses d'un autre point de vue."},
    2: {"name": "Dégoût", "message": "Concentrez-vous sur les aspects positifs de la situation."},
    3: {"name": "Peur", "message": "Vous êtes en sécurité, prenez votre temps pour vous apaiser."},
    4: {"name": "Bonheur", "message": "Votre sourire illumine la pièce ! Continuez ainsi !"},
    5: {"name": "Neutre", "message": "Vous semblez calme et posé."},
    6: {"name": "Tristesse", "message": "Chaque jour est une nouvelle opportunité. Gardez espoir !"},
    7: {"name": "Surprise", "message": "La vie est pleine de surprises positives !"}
}

# Configuration de la page Streamlit
st.set_page_config(page_title="Détecteur d'Émotions", layout="wide")

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .emotion-title {
        color: #333;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .emotion-message {
        color: #666;
        font-size: 18px;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.title("🎭 Détecteur d'Émotions en Temps Réel")

# Initialisation du modèle
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("cnn_emotion_model.pth", map_location=device))
    model.eval()
    return model, device

# Chargement du modèle
model, device = load_model()

# Transformations pour l'image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Chargement du classificateur Haar pour la détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Configuration de la webcam
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Flux Vidéo")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 😊 Émotion Détectée")
    emotion_placeholder = st.empty()
    message_placeholder = st.empty()

# Bouton pour démarrer/arrêter la détection
start_button = st.button("Démarrer la Détection")

if start_button:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la webcam. Veuillez vérifier vos permissions.")
        st.stop()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Erreur lors de la capture vidéo.")
                break

            # Détection des visages
            faces = detect_faces(frame)
            
            # Traitement de chaque visage détecté
            for (x, y, w, h) in faces:
                # Dessiner la bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extraire et prétraiter le visage
                face_img = frame[y:y+h, x:x+w]
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                
                # Prédiction de l'émotion
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion_idx = predicted.item()
                    
                # Afficher l'émotion sur l'image
                emotion_name = emotion_dict[emotion_idx]["name"]
                cv2.putText(frame, emotion_name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Mettre à jour l'affichage de l'émotion et du message
                with emotion_placeholder:
                    st.markdown(f"""
                    <div class="emotion-box">
                        <div class="emotion-title">{emotion_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with message_placeholder:
                    st.markdown(f"""
                    <div class="emotion-box">
                        <div class="emotion-message">{emotion_dict[emotion_idx]["message"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Afficher le flux vidéo
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Petite pause pour éviter une surcharge
            time.sleep(0.1)

    except Exception as e:
        st.error(f"Une erreur s'est produite : {str(e)}")
    
    finally:
        cap.release()

else:
    st.info("👆 Cliquez sur le bouton pour démarrer la détection d'émotions.")

