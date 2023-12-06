import cv2

# Fungsi untuk mendeteksi wajah
def detect_faces(image_path):
    # Baca gambar dari file
    image = cv2.imread(image_path)
    
    # Menggunakan pre-trained classifier untuk wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Konversi gambar ke skala abu-abu (diperlukan untuk deteksi wajah)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Gambar kotak di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Tampilkan gambar yang sudah diolah
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan
image_path = 'pic.jpg'
detect_faces(image_path)
