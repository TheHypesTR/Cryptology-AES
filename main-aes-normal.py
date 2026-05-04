import cv2
import numpy as np
from ultralytics import YOLO
from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto.Util.Padding import pad
import time
import psutil
import os
import torch

TEST_CORES = 4
TARGET_FPS = 10
VIDEO_PATH = 'aes_hd.mp4' 
YOLO_MODEL = 'yolov8n.pt' # 'yolov8n.pt'/'yolov8s.pt' / 'yolov8n-seg.pt'/'yolov8s-seg.pt' 
AES_MODE = 'CTR' # 'CTR'/'ECB'
AES_KEY = 128
CRYPT_KEY = b'1453269852165529'

p = psutil.Process(os.getpid())
p.cpu_affinity(list(range(TEST_CORES)))
torch.set_num_threads(TEST_CORES)


def encrypt_image_region(region):
    raw_bytes = region.tobytes()
    
    if AES_MODE == 'CTR':
        cipher = AES.new(CRYPT_KEY, AES.MODE_CTR, counter=Counter.new(AES_KEY))
        encrypted_bytes = cipher.encrypt(raw_bytes)
        
    elif AES_MODE == 'ECB':
        cipher = AES.new(CRYPT_KEY, AES.MODE_ECB)
        # ECB 16 byte bloklar ister. Bu yüzden veriyi pad'liyoruz.
        padded_bytes = pad(raw_bytes, AES.block_size)
        encrypted_padded = cipher.encrypt(padded_bytes)
        # Görüntü matrisinin bozulmaması için fazlalığı (padding kadar olan kısmı) atıyoruz
        encrypted_bytes = encrypted_padded[:len(raw_bytes)]

    # Byte dizisini tekrar görüntü matrisine çevir (Orijinal boyutlarla)
    encrypted_region = np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(region.shape)
    return encrypted_region

model = YOLO(YOLO_MODEL) 

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow('Sol: Orijinal | Sag: AES Sifreli', cv2.WINDOW_NORMAL)

original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps == 0: original_fps = 30
frame_skip_interval = max(1, int(original_fps / TARGET_FPS))

frame_counter = 0
processed_frame_count = 0
total_processing_time = 0
total_encryption_time = 0
total_encrypted_bytes = 0

print(f"--- TEST BAŞLADI ---")
print(f"Kullanılan Çekirdek: {TEST_CORES}")
print(f"Hedef FPS: {TARGET_FPS} (Orijinal: {original_fps})")
print(f"Şifreleme Modu: AES-{AES_MODE}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % frame_skip_interval != 0:
        continue

    encrypted_frame = frame.copy()
    t_start_frame = time.perf_counter()

    results = model.predict(frame, classes=[0], verbose=False)
    frame_encryption_time = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            human_roi = encrypted_frame[y1:y2, x1:x2]
            
            if human_roi.size > 0:
                t_start_enc = time.perf_counter()
                
                encrypted_frame[y1:y2, x1:x2] = encrypt_image_region(human_roi)
                
                t_end_enc = time.perf_counter()
                frame_encryption_time += (t_end_enc - t_start_enc)

    t_end_frame = time.perf_counter()
    frame_processing_time = t_end_frame - t_start_frame
    
    processed_frame_count += 1
    total_processing_time += frame_processing_time
    total_encryption_time += frame_encryption_time

    fps_achieved = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
    enc_ms = frame_encryption_time * 1000

    cv2.putText(encrypted_frame, f"Mode: AES-{AES_MODE} | Cores: {TEST_CORES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(encrypted_frame, f"Process FPS: {fps_achieved:.1f} / {TARGET_FPS}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(encrypted_frame, f"AES Time: {enc_ms:.2f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

    combined_frame = np.hstack((frame, encrypted_frame))
    cv2.imshow('Sol: Orijinal | Sag: AES Sifreli', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if processed_frame_count > 0:
    avg_proc_time = (total_processing_time / processed_frame_count) * 1000
    avg_enc_time = (total_encryption_time / processed_frame_count) * 1000
    
    print("\n--- TEST SONUÇLARI ---")
    print(f"Toplam İşlenen Kare: {processed_frame_count}")
    print(f"Kare Başına Ortalama Toplam İşlem: {avg_proc_time:.2f} ms")
    print(f"Kare Başına Ortalama AES Şifreleme: {avg_enc_time:.2f} ms")