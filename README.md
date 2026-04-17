# Handwriting OCR

## Project Intent

This project recognizes handwritten alphanumeric characters from an input image.
Its goal is to take a handwritten word or sequence of characters, isolate each
character region, run model inference per region, and return the final predicted
text in reading order.

At a high level, the project is designed for:

- Offline OCR inference on single images.
- Character-level recognition for digits and lowercase letters.
- Visual debugging through annotated output images with bounding boxes,
	predicted labels, and confidence scores.


## Workflow

The runtime workflow in [ocr_handwriting.py](ocr_handwriting.py) follows this sequence:

1. Parse inputs
- Required input image path.
- Optional model path and output image save path.

2. Load OCR model
- Load the TensorFlow model and obtain its serving signature for inference.

3. Read and prepare image
- Load the original color image.
- Convert to grayscale.
- Apply Otsu thresholding with inversion so foreground characters are white on
	black background.

4. Detect character candidates
- Find external contours in the thresholded image.
- Each contour is treated as a potential character region.
- Sort contours from left to right to preserve reading order.

5. Filter and extract ROIs
- Discard very small contours as noise.
- Clip contour coordinates to image bounds.
- Extract each grayscale character ROI.

6. Preprocess each ROI
- Resize ROI to 32x32.
- Normalize pixel values to the [0, 1] range.
- Expand dimensions to match model input shape (batch + channel dims).

7. Run inference and decode prediction
- Predict character class probabilities for each ROI.
- Select the top class with argmax.
- Map class index to character using the fixed label string:
	0123456789abcdefghijklmnopqrstuvwxyz

8. Build final OCR output
- Concatenate per-character predictions in sorted contour order.
- Print the recognized text string to the console.

9. Annotate and optionally save result image
- Draw bounding boxes and predicted labels with confidence on the original
	image.
- Save annotated image if output path is provided.

## Run Command
- install the essential libraries
- `python ocr_handwriting.py --model ocr_model.h5 --image images/umbc_address.png --output test_outputs/umbc_address_annotated.png`


## Output Behavior

- Console output includes processing logs and final OCR text.
- Optional image output includes region boxes and predicted label confidence.

