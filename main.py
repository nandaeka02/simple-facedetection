from email.mime import image
import cv2
# import numpy as np

# Untuk mengimport module pengenalan wajah
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Untuk inisialisasi webcam, disesuaikan dengan webcam setiap laptop
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)


while True:
    # Untuk mendapatkan frame dan status webcam
    ret, frame = cap.read()

    # Untuk mengubah warna menjadi abu-abu / gray
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Untuk melakukan deteksi wajah dan mendapatkan data titik koordinat

    # Parameter scaleFactor direkomendasikan antara 1.05 sampai 1.4,
    # lebih kecil lebih jelas facedetectionnya,
    # namun semakin lambat videonya karena beban cpu semakin berat, begitupun sebaliknya

    # Parameter minNeightbors direkomendasikan antara 3 - 6
    # Semakin tinggi nilainya semakin sedikit mendeteksi wajah
    # Namun kualitas deteksi bagus, begitu sebaliknya

    # Sesuaikan settingan ini dengan kebutuhan

    faces = faceCascade.detectMultiScale(
        image=imgGray,
        scaleFactor=1.09,
        minNeighbors=5
    )
    
    # Untuk membuat frame wajah sesuai data titik koordinat diatas
    for (x,y,w,h) in faces:
        # Membuat frame wajah
        cv2.rectangle(
            img=frame,
            pt1=(x, y-30),
            pt2=(x+w, y+h),
            color=(141, 15, 245),
            thickness=2,
        )
        # Membuat filled rectangle untuk background text
        cv2.rectangle(
            img=frame,
            pt1=(x,y),
            pt2=(x+w, y-30),
            color=(141, 15, 245),
            thickness=-1,
        )
        # Membuat text yang akan ditampilkan
        cv2.putText(
            img=frame,
            text="Nanda",
            org=(x, y-10),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            thickness=1,
            color=(255,255,255)
        )

    # Untuk menampilkan frame yang sudah selesai diolah
    cv2.imshow("Video", frame)

    # Untuk mencegah close dengan tombol silang dan hanya bisa close dengan tombol q pada keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
