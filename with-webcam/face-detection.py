import cv2

# Fungsi untuk mendeteksi wajah menggunakan Cascade Classifier
def detect_faces(frame):
    # Menggunakan Cascade Classifier untuk mendeteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Mengonversi citra ke skala abu-abu (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mendeteksi wajah dalam citra
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Menggambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame

# Membuka webcam
cap = cv2.VideoCapture(0)

while True:
    # Membaca setiap frame dari webcam
    ret, frame = cap.read()

    # Mendeteksi wajah dalam setiap frame
    frame_with_faces = detect_faces(frame)

    # Menampilkan frame dengan wajah yang terdeteksi
    cv2.imshow('Face Detection', frame_with_faces)

    # Keluar dari loop jika pengguna menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup webcam dan menghentikan penampilan frame
cap.release()
cv2.destroyAllWindows()
