# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import tensorflow as tf
# import cv2
# import os

# app = Flask(__name__)

# #Loading Our Model
# model = tf.keras.models.load_model('./model/PhysioDeep.keras')

# IMG_SIZE = 224
# MAX_SEQ_LENGTH = 20
# NUM_FEATURES = 2048

# def build_feature_extractor():
#     base_model = keras.applications.EfficientNetB0(
#         weights="imagenet", 
#         include_top=False, 
#         pooling="avg",
#         input_shape=(IMG_SIZE, IMG_SIZE, 3)
#     )
#     preprocess_input = keras.applications.efficientnet.preprocess_input

#     inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
#     preprocessed = preprocess_input(inputs)
#     outputs = base_model(preprocessed)
#     return keras.Model(inputs, outputs, name="feature_extractor")

# feature_extractor = build_feature_extractor()

# def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     try:
#         while len(frames) < max_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = crop_center_square(frame)
#             frame = cv2.resize(frame, resize)
#             frame = frame[..., [2, 1, 0]]  # Convert BGR to RGB
#             frames.append(frame)
#         # Pad shorter videos with black frames
#         while len(frames) < max_frames:
#             frames.append(np.zeros_like(frames[0]))
#     finally:
#         cap.release()
#     return np.array(frames)

# # Function to crop video frames to a center square
# def crop_center_square(frame):
#     y, x = frame.shape[0:2]
#     min_dim = min(y, x)
#     start_x = (x // 2) - (min_dim // 2)
#     start_y = (y // 2) - (min_dim // 2)
#     return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# def preprocess_video(video_path):
#     frames = load_video(video_path)
#     frames = frames[None, ...]  # Add batch dimension

#     # Initialize placeholders for features and masks
#     temp_frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
#     temp_frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

#     # Extract features from the frames
#     video_length = min(MAX_SEQ_LENGTH, frames.shape[1])
#     for j in range(video_length):
#         temp_frame_features[0, j, :] = feature_extractor.predict(frames[0, j][None, ...])
#     temp_frame_mask[0, :video_length] = 1

#     return temp_frame_features, temp_frame_mask

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400
    
#     video = request.files['video']
#     video_path = os.path.join("uploads", video.filename)
#     video.save(video_path)

#     frame_features, frame_mask = preprocess_video(video_path)

#     prediction = model.predict([frame_features, frame_mask])[0]
#     result = 'FAKE' if prediction >= 0.51 else 'REAL'
#     confidence = float(prediction)

#     os.remove(video_path)
#     return jsonify({'result': result, 'confidence': confidence})

# if __name__ == '__main__':
#     if not os.path.exists('uploads'):
#         os.makedirs('uploads')
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Loading Our Model
model = tf.keras.models.load_model('./model/PhysioDeep.keras')

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1280

def build_feature_extractor():
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet", 
        include_top=False, 
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = base_model(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[..., [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)
        # Pad shorter videos with black frames
        while len(frames) < max_frames:
            frames.append(np.zeros_like(frames[0]))
    finally:
        cap.release()
    return np.array(frames)

# Function to crop video frames to a center square
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def preprocess_video(video_path):
    frames = load_video(video_path)
    frames = frames[None, ...]  # Add batch dimension

    # Initialize placeholders for features and masks
    temp_frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    temp_frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Extract features from the frames
    video_length = min(MAX_SEQ_LENGTH, frames.shape[1])
    for j in range(video_length):
        temp_frame_features[0, j, :] = feature_extractor.predict(frames[0, j][None, ...])
    temp_frame_mask[0, :video_length] = 1

    return temp_frame_features, temp_frame_mask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        print("No file is choosen")
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)

    frame_features, frame_mask = preprocess_video(video_path)

    prediction = model.predict([frame_features, frame_mask])[0]
    result = 'FAKE' if prediction >= 0.51 else 'REAL'
    confidence = float(prediction[0])

    os.remove(video_path)
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
