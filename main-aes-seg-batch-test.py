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

YOLO_MODEL = 'yolov8n-seg.pt'  
CRYPT_KEY = b'1453269852165512'
OUTPUT_FILE = 'aes_sonuclari.txt'

VIDEO_PATHS_LIST = ['aes_vga_480p.mp4', 'aes_hd.mp4', 'aes_fhd.mp4']
TEST_CORES_LIST = [1, 2, 4]
TARGET_FPS_LIST = [10, 20]
AES_MODES_LIST = ['ECB', 'CTR']

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("OTOMATİK BATCH TEST BAŞLADI\n\n")

model = YOLO(YOLO_MODEL) 
test_no = 1

for video_path in VIDEO_PATHS_LIST:
    for fps in TARGET_FPS_LIST:
        for cores in TEST_CORES_LIST:
            for mode in AES_MODES_LIST:
                
                TEST_CORES = cores
                TARGET_FPS = fps
                AES_MODE = mode
                CURRENT_VIDEO = video_path
                
                p = psutil.Process(os.getpid())
                try:
                    p.cpu_affinity(list(range(TEST_CORES)))
                except AttributeError:
                    pass
                torch.set_num_threads(TEST_CORES)

                def encrypt_image_region(region, frame_id, object_id):
                    raw_bytes = region.tobytes()
                    if AES_MODE == 'CTR':
                        nonce = (frame_id << 16) | object_id
                        cipher = AES.new(CRYPT_KEY, AES.MODE_CTR, counter=Counter.new(128, initial_value=nonce))
                        encrypted_bytes = cipher.encrypt(raw_bytes)
                    elif AES_MODE == 'ECB':
                        cipher = AES.new(CRYPT_KEY, AES.MODE_ECB)
                        padded_bytes = pad(raw_bytes, AES.block_size)
                        encrypted_padded = cipher.encrypt(padded_bytes)
                        encrypted_bytes = encrypted_padded[:len(raw_bytes)]
                    return np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(region.shape)

                cap = cv2.VideoCapture(CURRENT_VIDEO)
                cv2.namedWindow('Test Ekrani', cv2.WINDOW_NORMAL)
                
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                if original_fps == 0: original_fps = 30
                frame_skip_interval = max(1, int(original_fps / TARGET_FPS))
                
                vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                frame_counter = 0
                processed_frame_count = 0
                total_processing_time = 0
                total_encryption_time = 0
                total_encrypted_bytes = 0

                print(f"[{test_no}] Başlıyor: Video={CURRENT_VIDEO}, Cores={TEST_CORES}, Mode={AES_MODE}")

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
                        if r.masks is None:
                            continue
                            
                        masks = r.masks.data.cpu().numpy()
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

                        for i, mask in enumerate(masks):
                            x1, y1, x2, y2 = boxes[i]
                            
                            p_val = 30
                            x1 = max(0, x1 - p_val)
                            y1 = max(0, y1 - p_val)
                            x2 = min(frame.shape[1], x2 + p_val)
                            y2 = min(frame.shape[0], y2 + p_val)
                            
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            mask_roi = mask_resized[y1:y2, x1:x2]
                            human_roi = encrypted_frame[y1:y2, x1:x2]
                            
                            bool_mask = mask_roi > 0.2
                            bool_mask_uint8 = bool_mask.astype(np.uint8)
                            kernel = np.ones((25, 25), np.uint8) 
                            dilated_mask = cv2.dilate(bool_mask_uint8, kernel, iterations=1)
                            bool_mask = dilated_mask > 0
                            
                            if np.any(bool_mask):
                                human_pixels = human_roi[bool_mask]
                                
                                t_start_enc = time.perf_counter()
                                total_encrypted_bytes += human_pixels.nbytes
                                encrypted_pixels = encrypt_image_region(human_pixels, frame_counter, i)
                                t_end_enc = time.perf_counter()
                                
                                frame_encryption_time += (t_end_enc - t_start_enc)
                                human_roi[bool_mask] = encrypted_pixels

                    t_end_frame = time.perf_counter()
                    
                    processed_frame_count += 1
                    total_processing_time += (t_end_frame - t_start_frame)
                    total_encryption_time += frame_encryption_time

                    combined_frame = np.hstack((frame, encrypted_frame))
                    
                    fps_achieved = 1.0 / (t_end_frame - t_start_frame) if (t_end_frame - t_start_frame) > 0 else 0
                    enc_ms = frame_encryption_time * 1000

                    cv2.putText(combined_frame, f"TEST: {test_no} | Cores: {TEST_CORES} | Mode: {AES_MODE} | Res: {vid_width}x{vid_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(combined_frame, f"FPS: {fps_achieved:.1f} / {TARGET_FPS} (Target)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(combined_frame, f"AES Time: {enc_ms:.2f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
                    cv2.imshow('Test Ekrani', combined_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Mevcut test atlandı.")
                        break
                    elif key == 27:
                        print("TÜM TESTLER İPTAL EDİLDİ.")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                cap.release()
                
                if processed_frame_count > 0:
                    avg_proc_time = (total_processing_time / processed_frame_count) * 1000
                    avg_enc_time = (total_encryption_time / processed_frame_count) * 1000
                    total_mb = total_encrypted_bytes / (1024 * 1024)
                    throughput_mb_s = total_mb / total_encryption_time if total_encryption_time > 0 else 0
                    
                    sonuc_metni = (
                        f"=============================================\n"
                        f"TEST SONUCU - {test_no}\n"
                        f"=============================================\n"
                        f"[TEST PARAMETRELERİ]\n"
                        f"Model Tipi         : {'Segmentasyon' if '-seg' in YOLO_MODEL else 'Dikdörtgen (BBox)'}\n"
                        f"Video Dosyası      : {CURRENT_VIDEO}\n"
                        f"Çözünürlük         : {vid_width} x {vid_height}\n"
                        f"Hedef FPS          : {TARGET_FPS} fps\n"
                        f"Şifreleme Modu     : AES-{AES_MODE}\n"
                        f"Kullanılan Çekirdek: {TEST_CORES}\n"
                        f"---------------------------------------------\n"
                        f"[PERFORMANS METRİKLERİ]\n"
                        f"Toplam İşlenen Kare      : {processed_frame_count}\n"
                        f"Toplam Şifrelenen Veri   : {total_mb:.2f} MB\n"
                        f"Ortalama Toplam İşlem    : {avg_proc_time:.2f} ms/kare\n"
                        f"Ortalama AES Süresi      : {avg_enc_time:.2f} ms/kare\n"
                        f"Şifreleme Verimi (Hızı)  : {throughput_mb_s:.2f} MB/s\n"
                        f"=============================================\n\n"
                    )
                    
                    print(f"Test {test_no} tamamlandı.")
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(sonuc_metni)
                
                test_no += 1 

cv2.destroyAllWindows()
print(f"\nTüm testler bitti. Sonuçlar '{OUTPUT_FILE}' dosyasına kaydedildi.")