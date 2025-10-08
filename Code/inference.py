import numpy as np
import tensorflow as tf
from collections import Counter
from badminton_utils2    import extract_3d_landmarks_from_video, normalize_landmarks

def predict_shot_from_video(video_path, model, label_map, sequence_length=40, stride=5):
    """
    Runs inference on a video file to predict the badminton shot type.
    
    Args:
        video_path (str): Path to the input video file.
        model (tf.keras.Model): The trained classification model.
        label_map (dict): A dictionary mapping class indices to shot names.
        sequence_length (int): The length of the sequences the model was trained on.
        stride (int): The step size to use for the sliding window.
        
    Returns:
        str: The predicted shot type for the video.
    """
    print(f"Processing video: {video_path}...")
    
    # 1. Extract and preprocess landmarks from the video
    landmarks, _ = extract_3d_landmarks_from_video(video_path)
    if landmarks is None or len(landmarks) < sequence_length:
        return "Not enough data to classify"
        
    normalized_landmarks = normalize_landmarks(landmarks)
    frame_features = normalized_landmarks.reshape(normalized_landmarks.shape[0], -1)

    # 2. Use a sliding window to create sequences
    video_sequences = []
    for i in range(0, len(frame_features) - sequence_length + 1, stride):
        window = frame_features[i: i + sequence_length]
        video_sequences.append(window)
    
    if not video_sequences:
        return "Could not generate sequences from video"

    # 3. Make predictions on each sequence
    predictions = model.predict(np.array(video_sequences))
    predicted_class_indices = np.argmax(predictions, axis=1)

    # 4. Aggregate predictions with a majority vote
    if len(predicted_class_indices) == 0:
        return "Prediction failed"
        
    most_common_index = Counter(predicted_class_indices).most_common(1)[0][0]
    
    # 5. Map the index back to the shot name
    predicted_shot = label_map.get(most_common_index, "Unknown Shot")
    
    return predicted_shot

if __name__ == '__main__':
    # --- CONFIGURATION ---
    MODEL_PATH = "/home/smayan/Desktop/IPD/Code/badminton_shot_classifier_v3_sliced.h5"  # <--- CHANGE THIS to your model path
    VIDEO_PATH = "/home/smayan/Desktop/IPD/Data/forehand_net_shot/012.mp4"              # <--- CHANGE THIS to your video path
    SEQUENCE_LENGTH = 50# Must be the same as used in training
    
    # --- IMPORTANT ---
    # You MUST use the same label map that was generated during training.
    # The order of shot types matters.
    LABEL_MAP = {
        0: "clear",
        1: "drive",
        2: "drop",
        3: "smash"
        # ... add all your shot types here in the correct order
    }
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Run inference
    final_prediction = predict_shot_from_video(
        VIDEO_PATH, 
        model, 
        LABEL_MAP, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    print("\n" + "="*30)
    print(f"🚀 Final Predicted Shot: {final_prediction}")
    print("="*30)