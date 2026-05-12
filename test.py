import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml
from ai_edge_litert.compiled_model import CompiledModel, HardwareAccelerator

script_dir = Path(__file__).parent.resolve()
print(f"Skriptverzeichnis: {script_dir}")
# ── Konfiguration ────────────────────────────────────────────
MODEL_PATH     = f"{script_dir}\\assets\\mymodel.tflite"
IMAGE_PATH     =  f"{script_dir}\\res\\p1.jpg"
IMAGE_NAME = "p1"
RESULT_DIR       = f"{script_dir}\\out_file\\"
INPUT_SIZE     = (320, 320)
CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.45

url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.yaml"
with urllib.request.urlopen(url) as f:
    coco = yaml.safe_load(f.read().decode())

COCO_CLASSES = coco['names']  # Liste in korrekter Reihenfolge

# ── Modell laden ─────────────────────────────────────────────
model          = CompiledModel.from_file(MODEL_PATH)

sig_index      = 0
sig_key   = list(model.get_signature_list().keys())[0]
input_buffers  = model.create_input_buffers(sig_index)
output_buffers = model.create_output_buffers(sig_index)

out_details  = model.get_output_tensor_details(sig_key)
out_shape    = out_details['Identity']['shape']       # ← direkt mit Name
num_elements = int(np.prod(out_shape))

print(input_buffers[0].get_tensor_details()['shape'])
print(output_buffers[0].get_tensor_details()['shape'])

# ── Bild vorverarbeiten ──────────────────────────────────────
def preprocess(image_path):
    img_orig = cv2.imread(image_path)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, img_orig

# ── Inferenz ─────────────────────────────────────────────────
def run_inference(img):
    input_buffers[0].write(img)
    start = time.perf_counter()
    model.run_by_index(sig_index, input_buffers, output_buffers)
    end = time.perf_counter()

    ms = (end - start) * 1000
    print(f"Inferenz: {ms:.1f} ms")

    return output_buffers[0].read(num_elements, np.float32).reshape(out_shape)

# ── Output dekodieren ────────────────────────────────────────
def decode_output(output, orig_shape):
    predictions = output[0]
    orig_h, orig_w = orig_shape[:2]
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        objectness = pred[4]

        if objectness < CONF_THRESHOLD:
            continue
        else:
            print(pred[4])
        class_probs = pred[5:]
        class_id    = np.argmax(class_probs)
        confidence  = objectness * class_probs[class_id]
        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    results = []
    for i in indices:
        x, y, w, h = boxes[i]
        results.append({
            "class":      COCO_CLASSES[class_ids[i]],
            "confidence": round(scores[i], 3),
            "box":        (x, y, x + w, y + h)
        })
    return results

# ── Visualisieren ────────────────────────────────────────────
def draw_results(img_orig, results):
    for det in results:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_orig, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    result_file = os.path.join(RESULT_DIR, f"{IMAGE_NAME}_result.jpg")
    cv2.imwrite(result_file, img_orig)
    print(f"Gespeichert: ergebnis.jpg")

# ── Main ─────────────────────────────────────────────────────
img, img_orig = preprocess(IMAGE_PATH)
output        = run_inference(img)
results       = decode_output(output, img_orig.shape)
draw_results(img_orig, results)

print(f"\n{len(results)} Objekt(e) erkannt:")
for det in results:
    print(f"  {det['class']:15s}  {det['confidence']:.1%}  Box: {det['box']}")