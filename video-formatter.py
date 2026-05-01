import cv2
import time

# ==========================================
# DÖNÜŞTÜRME AYARLARI
# ==========================================
INPUT_VIDEO = 'aes_fhd.mp4'       # Küçültmek istediğin orijinal video
OUTPUT_VIDEO = 'aes_vga_360p_1.mp4' # Çıktı olarak alınacak yeni videonun adı

# 16:9 formatını bozmamak için 640x360 kullanıyoruz. 
# Eğer klasik VGA istersen 640 ve 480 yapabilirsin.
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
# ==========================================

cap = cv2.VideoCapture(INPUT_VIDEO)

# Orijinal videonun FPS ve toplam kare sayısını al
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps == 0: 
    fps = 30 # Eğer okuyamazsa varsayılan 30 fps kabul et

# Video kaydediciyi (Writer) hazırla
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

print(f"--- DÖNÜŞTÜRME BAŞLIYOR ---")
print(f"Girdi: {INPUT_VIDEO}")
print(f"Çıktı: {OUTPUT_VIDEO}")
print(f"Hedef Çözünürlük: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print(f"Toplam Kare: {total_frames}")

start_time = time.time()
frames_processed = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Kareyi hedef çözünürlüğe küçült
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # 2. Yeni dosyaya yaz
    out.write(resized_frame)

    frames_processed += 1
    
    # Süreci terminalden takip edebilmek için her 100 karede bir bilgi yazdır
    if frames_processed % 100 == 0:
        percent = (frames_processed / total_frames) * 100
        print(f"İşleniyor: %{percent:.1f} ({frames_processed}/{total_frames} kare)")

cap.release()
out.release()

elapsed_time = time.time() - start_time
print(f"\n--- İŞLEM TAMAMLANDI ---")
print(f"Geçen Süre: {elapsed_time:.2f} saniye")
print(f"Yeni video '{OUTPUT_VIDEO}' adıyla klasöre kaydedildi.")