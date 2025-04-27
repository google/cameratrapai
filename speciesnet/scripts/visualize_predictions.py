import json
import cv2
import os

# === CONFIGURATION ===
PREDICTIONS_JSON_PATH = "/your/path/to/project/predictions.json"
OUTPUT_DIR = "/your/path/to/project/visualization"
FONT_SCALE = 1.2  # Bigger font
FONT_THICKNESS = 2
CONFIDENCE_THRESHOLD = 0.65  # Only show detections >= 65%

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load predictions
with open(PREDICTIONS_JSON_PATH, "r") as f:
    predictions_data = json.load(f)

# Process each image
for item in predictions_data["predictions"]:
    image_path = item["filepath"]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: could not load {image_path}")
        continue

    image_height, image_width = image.shape[:2]

    # Draw each detection if confidence is high enough
    for detection in item["detections"]:
        conf = detection["conf"]

        if conf < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence detections

        bbox = detection["bbox"]
        label = detection["label"]

        # Convert normalized bbox [x, y, width, height] -> absolute
        x1 = int(bbox[0] * image_width)
        y1 = int(bbox[1] * image_height)
        w = int(bbox[2] * image_width)
        h = int(bbox[3] * image_height)
        x2 = x1 + w
        y2 = y1 + h

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw classification label (predicted class)
        if "prediction" in item:
            # prediction format: id;mammalia;carnivora;viverridae;genetta;;genetta species
            full_prediction = item["prediction"]
            species_name = full_prediction.split(";")[-1]  # Extract species name
            text = f"{species_name} ({conf:.2f})"
        else:
            text = f"{label} ({conf:.2f})"

        # Text position
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        text_w, text_h = text_size

        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, (0, 0, 0), FONT_THICKNESS, lineType=cv2.LINE_AA)

    # Save visualized image
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_DIR, image_filename)
    cv2.imwrite(output_path, image)
    print(f"Saved {output_path}")
