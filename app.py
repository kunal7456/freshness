import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def process_image(model, image_path):
    """Run YOLO inference on an image and return the results."""
    image = Image.open(image_path)
    image_resized = image.resize((854, 480))  # Resize if needed

    # Perform inference
    results = model(image_resized)
    boxes = results[0].boxes.xyxy  # Bounding boxes
    confidences = results[0].boxes.conf  # Confidence scores
    class_ids = results[0].boxes.cls  # Class IDs
    labels = results[0].names  # Class labels

    # Prepare annotated image
    image_np = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = confidences[i].item()
        label = labels[int(class_ids[i].item())]

        # Draw bounding box and label
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, f"{label} {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Streamlit App Layout
st.title("YOLO Object Detection App")
st.markdown("Upload an image, and the app will detect objects using your pre-trained YOLO model.")

# Load the Pre-trained YOLO Model
model_path = "best_for_freshness.pt"  # Replace with your pre-trained model path
st.write(f"Loading model from `{model_path}`...")
model = YOLO(model_path)
st.success("Model loaded successfully!")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_image_path = "temp_uploaded_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show uploaded image
    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)

    # Process and display results
    output_image = process_image(model, temp_image_path)
    st.image(output_image, caption="Detected Objects", use_column_width=True)

    # Clean up temporary file (optional)
    # os.remove(temp_image_path)
else:
    st.warning("Please upload an image to proceed.")
