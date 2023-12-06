import cv2

# Inisialisasi cascade classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mulai webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = video_capture.read()

    # Konversi frame ke abu-abu untuk meningkatkan kecepatan deteksi
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar wajah yang terdetek
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tampilkan frame hasil
    cv2.imshow('Face Detection', frame)

    # Hentikan program jika pengguna menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela tampilan
video_capture.release()
cv2.destroyAllWindows()
