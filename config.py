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
    FACE_CONFIDENCE_THRESHOLD = 45  # Confidence score harus <= threshold untuk dianggap valid
    FACE_MIN_FRAMES = 15  # Minimal frame untuk verification
    FACE_VOTING_PERCENTAGE = 70  # Minimal 70% frame harus memiliki label yang sama
