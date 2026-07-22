import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "ganti-dengan-secret-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@localhost/absensi_karyawan",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Face Recognition Configuration
    # Nilai confidence LBPH lebih rendah berarti kecocokan lebih baik.
    # Untuk dataset kecil, nilai threshold yang terlalu tinggi dapat menyebabkan false positive.
    FACE_CONFIDENCE_THRESHOLD = 45  # Confidence score harus <= threshold untuk dianggap valid
    FACE_MIN_FRAMES = 15  # Minimal frame untuk verification
    FACE_VOTING_PERCENTAGE = 70  # Minimal 70% frame harus memiliki label yang sama
    FACE_SAMPLE_AUGMENTATIONS = 3  # Jumlah variasi sampel wajah yang disimpan per registrasi
