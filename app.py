from io import BytesIO
import base64
import os
from datetime import date, datetime

import cv2
import time
import sys

print("=" * 50)
print("PYTHON :", sys.executable)
print("CV2 :", cv2.__file__)
print("HAS FACE :", hasattr(cv2, "face"))
print("=" * 50)
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_file, jsonify
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import inspect, text
import traceback
from werkzeug.utils import secure_filename

from config import Config
from extensions import db
from models import Absensi, Karyawan

UPLOAD_FOLDER = "absensi_app/static/uploads"
FACES_FOLDER = os.path.join(UPLOAD_FOLDER, "faces")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_face_image(image, use_clahe=True):
    """
    Preprocess wajah untuk meningkatkan akurasi recognition.
    - Apply CLAHE untuk normalisasi lighting
    - Histogram equalization jika diperlukan
    - Konversi ke grayscale jika belum
    """
    # 1) Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Apply CLAHE to normalize local contrast
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    # 3) Gentle Gaussian blur to reduce high-frequency noise
    try:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    except Exception:
        # If blur fails for any reason, continue with current image
        pass

    # 4) Global histogram equalization to improve global contrast
    try:
        image = cv2.equalizeHist(image)
    except Exception:
        pass

    return image


def normalize_brightness_contrast(image, alpha=1.2, beta=30):
    """
    Normalisasi brightness dan contrast dengan formula:
    output = alpha * input + beta
    """
    
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


def augment_face_image(image, version):
    """Buat variasi gambar wajah untuk menambah keanekaragaman sampel."""
    if version == 0:
        return image

    augmented = image.copy()
    if version == 1:
        # Brightness sedikit lebih tinggi
        augmented = normalize_brightness_contrast(augmented, alpha=1.1, beta=15)
    elif version == 2:
        # Brightness sedikit lebih rendah
        augmented = normalize_brightness_contrast(augmented, alpha=0.9, beta=-10)
    elif version == 3:
        # Rotasi kecil untuk variasi sudut wajah
        h, w = augmented.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1)
        augmented = cv2.warpAffine(augmented, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return augmented


def majority_vote_recognition(predictions):
    """
    Gunakan majority voting untuk menentukan identitas.
    Input: list of (label, confidence) tuples
    Output: (most_common_label, average_confidence, count, total_frames) atau (None, 0.0, 0, 0)
    """
    from collections import Counter
    
    if not predictions:
        return None, 0.0, 0, 0
    
    # Hitung frekuensi label
    labels = [p[0] for p in predictions]
    label_counts = Counter(labels)
    
    # Ambil label yang paling sering muncul
    most_common_label, count = label_counts.most_common(1)[0]
    
    # Hitung rata-rata confidence untuk label terpilih
    confidences = [p[1] for p in predictions if p[0] == most_common_label]
    avg_confidence = sum(confidences) / len(confidences)
    
    return most_common_label, avg_confidence, count, len(predictions)


def ensure_face_image_column(app):
    inspector = inspect(db.engine)
    if "karyawan" in inspector.get_table_names():
        columns = [column["name"] for column in inspector.get_columns("karyawan")]
        if "face_image" not in columns:
            with db.engine.begin() as conn:
                conn.execute(text("ALTER TABLE karyawan ADD COLUMN face_image VARCHAR(255) NULL"))


def decode_face_image(face_b64):
    if not face_b64 or "," not in face_b64:
        return None

    try:
        _, encoded = face_b64.split(",", 1)
        image_data = base64.b64decode(encoded)
    except Exception:
        return None

    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def get_face_cascade():
    candidate_paths = []
    try:
        candidate_paths.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    candidate_paths.extend([
        os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"),
        os.path.join(BASE_DIR, "absensi_app", "static", "models", "haarcascade_frontalface_default.xml"),
        os.path.join(BASE_DIR, "absensi_app", "static", "models", "haarcascade_frontalface_alt2.xml"),
        os.path.join(BASE_DIR, "absensi_app", "static", "models", "haarcascade_frontalface_alt.xml"),
    ])

    for path in candidate_paths:
        if not path:
            continue
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    return None


def save_face_image(face_b64, filename):
    image = decode_face_image(face_b64)
    if image is None:
        return None

    classifier = get_face_cascade()
    if classifier is None:
        return "cascade_error"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return "no_face"
    if len(faces) > 1:
        return "multiple_faces"

    secure_name = secure_filename(filename)
    file_path = os.path.join(FACES_FOLDER, secure_name)
    _, encoded_img = cv2.imencode('.jpg', image)
    with open(file_path, 'wb') as f:
        f.write(encoded_img.tobytes())

    return secure_name


def save_face_sample(face_b64, karyawan_id, app=None):
    image = decode_face_image(face_b64)
    if image is None:
        return {"status": "invalid_image"}

    classifier = get_face_cascade()
    if classifier is None:
        return {"status": "cascade_error"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return {"status": "no_face"}
    if len(faces) > 1:
        return {"status": "multiple_faces"}

    faces_folder = app.config.get("FACES_FOLDER", FACES_FOLDER) if app else FACES_FOLDER
    karyawan_folder = os.path.join(faces_folder, f"karyawan_{karyawan_id}")
    os.makedirs(karyawan_folder, exist_ok=True)

    (x, y, w, h) = faces[0]
    face_roi_color = image[y:y + h, x:x + w]
    face_roi_color = cv2.resize(face_roi_color, (200, 200))

    existing_samples = [name for name in os.listdir(karyawan_folder) if name.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_number = len(existing_samples) + 1
    saved_files = []
    augmentations = app.config.get('FACE_SAMPLE_AUGMENTATIONS', 3) if app else 3

    for variation in range(augmentations):
        augmented_face = augment_face_image(face_roi_color, variation)
        sample_filename = f"{next_number:03d}.jpg"
        sample_path = os.path.join(karyawan_folder, sample_filename)
        sample_path_rel = f"karyawan_{karyawan_id}/{sample_filename}"
        _, encoded_img = cv2.imencode('.jpg', augmented_face)
        with open(sample_path, 'wb') as f:
            f.write(encoded_img.tobytes())
        saved_files.append(sample_path_rel)
        next_number += 1

    return {"status": "saved", "saved_count": len(saved_files), "path": saved_files[0]}


def get_face_image_url(karyawan, app=None):
    if not karyawan or not getattr(karyawan, 'face_image', None):
        return None

    face_image_value = karyawan.face_image.replace('\\', '/')
    faces_folder = app.config.get('FACES_FOLDER', FACES_FOLDER) if app else FACES_FOLDER
    candidate_file = os.path.join(faces_folder, face_image_value)

    if os.path.isfile(candidate_file):
        return url_for('static', filename=f'uploads/faces/{face_image_value}')

    candidate_dir = os.path.join(faces_folder, face_image_value)
    if os.path.isdir(candidate_dir):
        image_files = sorted([
            name for name in os.listdir(candidate_dir)
            if name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if image_files:
            image_rel = f"{face_image_value}/{image_files[0]}"
            return url_for('static', filename=f'uploads/faces/{image_rel}')

    return None


def load_face_training_data():
    """
    Load training data dengan support untuk:
    1. New format: FACES_FOLDER/karyawan_{id}/ dengan multiple samples
    2. Old format: FACES_FOLDER/face_{id}.jpg (backward compatibility)
    """
    images = []
    labels = []
    label_map = {}

    karyawan_list = Karyawan.query.all()
    for karyawan in karyawan_list:
        # Coba baca dari subfolder (new format)
        karyawan_face_dir = os.path.join(FACES_FOLDER, f"karyawan_{karyawan.id}")
        loaded_from_dir = False
        
        if os.path.exists(karyawan_face_dir) and os.path.isdir(karyawan_face_dir):
            for sample_file in sorted(os.listdir(karyawan_face_dir)):
                if sample_file.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG')):
                    sample_path = os.path.join(karyawan_face_dir, sample_file)
                    try:
                        image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
                        if image is not None:
                            image = cv2.resize(image, (200, 200))
                            # Apply preprocessing
                            image = preprocess_face_image(image, use_clahe=True)
                            image = normalize_brightness_contrast(image, alpha=1.2, beta=30)
                            images.append(image)
                            labels.append(karyawan.id)
                            loaded_from_dir = True
                    except Exception as e:
                        print(f"Error preprocessing {sample_path}: {e}")
        
        # Fallback ke old format untuk backward compatibility
        if not loaded_from_dir and karyawan.face_image:
            face_path = os.path.join(FACES_FOLDER, karyawan.face_image)
            if os.path.exists(face_path):
                try:
                    image = cv2.imread(face_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.resize(image, (200, 200))
                        image = preprocess_face_image(image, use_clahe=True)
                        image = normalize_brightness_contrast(image, alpha=1.2, beta=30)
                        images.append(image)
                        labels.append(karyawan.id)
                except Exception as e:
                    print(f"Error preprocessing {face_path}: {e}")
        
        # Pastikan karyawan ada di label_map jika ada sampel
        if karyawan.id in labels:
            label_map[karyawan.id] = karyawan

    return images, labels, label_map


def build_face_recognizer():
    images, labels, label_map = load_face_training_data()

    print("DEBUG IMAGES:", len(images))
    print("DEBUG LABELS:", len(labels))
    print("DEBUG HAS FACE:", hasattr(cv2, "face"))

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(images, np.array(labels))

    return recognizer, label_map

    recognizer.train(images, np.array(labels))
    return recognizer, label_map


def recognize_face_from_image(recognizer, label_map, face_b64, app=None):
    image = decode_face_image(face_b64)
    if image is None:
        return {"status": "invalid_image"}

    if app is None:
        threshold = 45.0
    else:
        threshold = app.config.get('FACE_CONFIDENCE_THRESHOLD', 45.0)

    face_cascade = get_face_cascade()
    if face_cascade is None:
        return {"status": "cascade_error"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return {"status": "no_face"}
    if len(faces) > 1:
        return {"status": "multiple_faces"}

    (x, y, w, h) = faces[0]
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (200, 200))
    face_roi = preprocess_face_image(face_roi, use_clahe=True)
    face_roi = normalize_brightness_contrast(face_roi, alpha=1.2, beta=30)

    label, confidence = recognizer.predict(face_roi)
    if float(confidence) <= threshold and label in label_map:
        return {"status": "recognized", "label": label, "confidence": float(confidence)}

    return {"status": "unknown", "confidence": float(confidence)}


def generate_excel(absensi_records):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Laporan Absensi"

    headers = ["Tanggal", "Nama Karyawan", "Jam Masuk", "Jam Pulang", "Keterangan"]
    sheet.append(headers)

    for record in absensi_records:
        sheet.append([
            record.tanggal.strftime("%d-%m-%Y"),
            record.karyawan.nama if record.karyawan else "-",
            record.jam_masuk.strftime("%H:%M:%S") if record.jam_masuk else "",
            record.jam_pulang.strftime("%H:%M:%S") if record.jam_pulang else "",
            record.keterangan or "",
        ])

    for column_cells in sheet.columns:
        length = max(len(str(cell.value or "")) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = length + 4

    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output


def generate_pdf(absensi_records, start, end):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Laporan Absensi Karyawan", styles["Title"]))
    elements.append(Paragraph(
        f"Periode: {start.strftime('%d-%m-%Y')} sampai {end.strftime('%d-%m-%Y')}",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 12))

    data = [["Tanggal", "Nama Karyawan", "Jam Masuk", "Jam Pulang", "Keterangan"]]
    for record in absensi_records:
        data.append([
            record.tanggal.strftime("%d-%m-%Y"),
            record.karyawan.nama if record.karyawan else "-",
            record.jam_masuk.strftime("%H:%M:%S") if record.jam_masuk else "",
            record.jam_pulang.strftime("%H:%M:%S") if record.jam_pulang else "",
            record.keterangan or "",
        ])

    table = Table(data, repeatRows=1, colWidths=[3.2 * cm, 4.5 * cm, 3 * cm, 3 * cm, 4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    return buffer


def create_default_user():
    admin = Karyawan.query.filter_by(username="admin").first()

    if admin is None:
        admin = Karyawan(
            nama="Administrator",
            username="admin"
        )
        db.session.add(admin)

    admin.set_password("admin123")

    db.session.commit()


def create_app():
    app = Flask(
        __name__,
        template_folder="absensi_app/templates",
        static_folder="absensi_app/static",
    )
    app.config.from_object(Config)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["FACES_FOLDER"] = FACES_FOLDER

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["FACES_FOLDER"], exist_ok=True)
    db.init_app(app)

    with app.app_context():
        db.create_all()
        create_default_user()
        ensure_face_image_column(app)

    @app.route("/", methods=["GET", "POST"])
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = Karyawan.query.filter_by(username=username).first()

            if user and user.check_password(password):
                session["user_id"] = user.id
                session["user_name"] = user.nama
                flash("Login berhasil. Selamat datang, {}!".format(user.nama), "success")
                return redirect(url_for("dashboard"))

            flash("Username atau password salah.", "danger")

        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        user = Karyawan.query.get(session["user_id"])
        history_user = user
        if session.get("recognized_karyawan_id"):
            recognized_history = Karyawan.query.get(session["recognized_karyawan_id"])
            if recognized_history:
                history_user = recognized_history
            else:
                session.pop("recognized_karyawan_id", None)

        absensi_list = Absensi.query.filter_by(karyawan_id=history_user.id).order_by(Absensi.tanggal.desc(), Absensi.id.desc()).limit(10).all()
        total_absensi = Absensi.query.count()
        total_karyawan = Karyawan.query.count()
        today = date.today()
        total_absensi_today = Absensi.query.filter_by(tanggal=today).count()
        jumlah_hadir = total_absensi_today
        jumlah_belum_hadir = max(total_karyawan - jumlah_hadir, 0)

        return render_template(
            "dashboard.html",
            user=user,
            history_user=history_user,
            absensi_list=absensi_list,
            total_absensi=total_absensi,
            total_karyawan=total_karyawan,
            total_absensi_today=total_absensi_today,
            jumlah_hadir=jumlah_hadir,
            jumlah_belum_hadir=jumlah_belum_hadir,
            sekarang=datetime.now(),
        )

    @app.route("/absensi", methods=["GET", "POST"])
    def absensi():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        user = Karyawan.query.get(session["user_id"])
        today = date.today()
        hari_ini = Absensi.query.filter_by(karyawan_id=user.id, tanggal=today).first()

        if request.method == "POST":
            action = request.form.get("action")

            if action == "face":
                face_b64 = request.form.get("face_image", "").strip()
                if not face_b64:
                    message = "Foto wajah belum diambil."
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return jsonify({"status": "error", "message": message})
                    flash(message, "warning")
                    return redirect(url_for("absensi"))

                try:
                    recognizer, label_map = build_face_recognizer()
                except ImportError as e:
                    message = str(e)
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return jsonify({"status": "error", "message": message})
                    flash(message, "danger")
                    return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

                if recognizer is None:
                    message = "Belum ada wajah terdaftar. Silakan lakukan registrasi wajah terlebih dahulu."
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return jsonify({"status": "error", "message": message})
                    flash(message, "warning")
                    return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

                result = recognize_face_from_image(recognizer, label_map, face_b64, app)
                if result["status"] == "recognized":
                    recognized_id = result["label"]
                    recognized_karyawan = label_map.get(recognized_id)
                    confidence = result["confidence"]

                    if not recognized_karyawan:
                        message = "Wajah dikenali tetapi data karyawan tidak ditemukan."
                        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                            return jsonify({"status": "error", "message": message})
                        flash(message, "danger")
                        return redirect(url_for("absensi"))

                    existing_absensi = Absensi.query.filter_by(
                        karyawan_id=recognized_karyawan.id,
                        tanggal=today,
                    ).first()

                    if not existing_absensi:
                        absensi = Absensi(
                            karyawan_id=recognized_karyawan.id,
                            tanggal=today,
                            jam_masuk=datetime.now().time(),
                            keterangan="Hadir",
                        )
                        db.session.add(absensi)
                        db.session.commit()
                        session["recognized_karyawan_id"] = recognized_karyawan.id
                        hari_ini_data = {
                            "tanggal": today.strftime('%d-%m-%Y'),
                            "jam_masuk": absensi.jam_masuk.strftime('%H:%M:%S'),
                            "jam_pulang": '-',
                            "keterangan": absensi.keterangan,
                        }
                        message = f"Absensi masuk berhasil untuk {recognized_karyawan.nama}. Confidence: {confidence:.2f}"
                        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                            return jsonify({
                                "status": "success",
                                "message": message,
                                "recognized_nama": recognized_karyawan.nama,
                                "hari_ini": hari_ini_data,
                            })
                        flash(message, "success")
                        return redirect(url_for("dashboard"))

                    if existing_absensi and not existing_absensi.jam_pulang:
                        existing_absensi.jam_pulang = datetime.now().time()
                        db.session.commit()
                        session["recognized_karyawan_id"] = recognized_karyawan.id
                        hari_ini_data = {
                            "tanggal": today.strftime('%d-%m-%Y'),
                            "jam_masuk": existing_absensi.jam_masuk.strftime('%H:%M:%S') if existing_absensi.jam_masuk else '-',
                            "jam_pulang": existing_absensi.jam_pulang.strftime('%H:%M:%S'),
                            "keterangan": existing_absensi.keterangan,
                        }
                        message = f"Absensi pulang berhasil untuk {recognized_karyawan.nama}. Confidence: {confidence:.2f}"
                        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                            return jsonify({
                                "status": "success",
                                "message": message,
                                "recognized_nama": recognized_karyawan.nama,
                                "hari_ini": hari_ini_data,
                            })
                        flash(message, "success")
                        return redirect(url_for("dashboard"))

                    session["recognized_karyawan_id"] = recognized_karyawan.id
                    hari_ini_data = {
                        "tanggal": today.strftime('%d-%m-%Y'),
                        "jam_masuk": existing_absensi.jam_masuk.strftime('%H:%M:%S') if existing_absensi.jam_masuk else '-',
                        "jam_pulang": existing_absensi.jam_pulang.strftime('%H:%M:%S') if existing_absensi.jam_pulang else '-',
                        "keterangan": existing_absensi.keterangan,
                    }
                    message = f"Absensi hari ini untuk {recognized_karyawan.nama} sudah lengkap (masuk dan pulang)."
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return jsonify({
                            "status": "warning",
                            "message": message,
                            "recognized_nama": recognized_karyawan.nama,
                            "hari_ini": hari_ini_data,
                        })
                    flash(message, "warning")
                    return redirect(url_for("dashboard"))

                if result["status"] == "no_face":
                    message = "Tidak ada wajah yang terdeteksi. Pastikan hanya satu orang berada di depan kamera."
                elif result["status"] == "multiple_faces":
                    message = "Lebih dari satu wajah terdeteksi. Pastikan hanya satu orang berada di depan kamera."
                elif result["status"] == "invalid_image":
                    message = "Gambar tidak valid. Coba ambil foto lagi."
                else:
                    message = "Wajah tidak terdaftar. Silakan registrasi terlebih dahulu."

                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"status": "error", "message": message})
                flash(message, "danger")
                return redirect(url_for("absensi"))

            if action == "masuk":
                if hari_ini:
                    flash("Anda sudah melakukan absen masuk hari ini.", "warning")
                else:
                    absensi = Absensi(karyawan_id=user.id, tanggal=today, jam_masuk=datetime.now().time(), keterangan="Hadir")
                    db.session.add(absensi)
                    db.session.commit()
                    flash("Absensi masuk berhasil.", "success")
                return redirect(url_for("absensi"))

            if action == "pulang":
                if not hari_ini:
                    flash("Silakan absen masuk terlebih dahulu sebelum pulang.", "warning")
                elif hari_ini.jam_pulang:
                    flash("Anda sudah melakukan absen pulang hari ini.", "warning")
                else:
                    hari_ini.jam_pulang = datetime.now().time()
                    db.session.commit()
                    flash("Absensi pulang berhasil.", "success")
                return redirect(url_for("absensi"))

        return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

    @app.route("/laporan", methods=["GET"])
    def laporan_absensi():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        export = request.args.get("export")

        if start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                flash("Format tanggal salah.", "danger")
                return redirect(url_for("laporan_absensi"))
        else:
            end = date.today()
            start = end.replace(day=1)
            start_date = start.isoformat()
            end_date = end.isoformat()

        absensi_records = Absensi.query.filter(Absensi.tanggal.between(start, end)).order_by(Absensi.tanggal.desc()).all()

        if export == "excel":
            excel_data = generate_excel(absensi_records)
            return send_file(
                excel_data,
                download_name=f"laporan_absensi_{start_date}_sd_{end_date}.xlsx",
                as_attachment=True,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if export == "pdf":
            pdf_data = generate_pdf(absensi_records, start, end)
            return send_file(
                pdf_data,
                download_name=f"laporan_absensi_{start_date}_sd_{end_date}.pdf",
                as_attachment=True,
                mimetype="application/pdf",
            )

        return render_template(
            "laporan_absensi.html",
            absensi_records=absensi_records,
            start_date=start_date,
            end_date=end_date,
        )

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Anda telah logout.", "info")
        return redirect(url_for("login"))

    @app.route("/karyawan")
    def daftar_karyawan():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan_list = Karyawan.query.all()
        return render_template("karyawan_list.html", karyawan_list=karyawan_list)

    @app.route("/registrasi-wajah")
    def registrasi_wajah():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan_list = Karyawan.query.all()
        return render_template("registrasi_wajah_list.html", karyawan_list=karyawan_list)

    @app.route("/registrasi-wajah/<int:karyawan_id>", methods=["GET", "POST"])
    def registrasi_wajah_detail(karyawan_id):
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan = Karyawan.query.get(karyawan_id)
        if not karyawan:
            flash("Data karyawan tidak ditemukan.", "warning")
            return redirect(url_for("registrasi_wajah"))

        if request.method == "POST":
            face_b64 = request.form.get("face_image", "").strip()
            if not face_b64:
                flash("Foto wajah belum diambil. Silakan ambil foto terlebih dahulu.", "warning")
                return redirect(url_for("registrasi_wajah_detail", karyawan_id=karyawan_id))

            result = save_face_sample(face_b64, karyawan_id, app)
            if result.get("status") != "saved":
                message = "Gagal memproses foto wajah. Pastikan wajah terlihat jelas."
                if result.get("status") == "no_face":
                    message = "Tidak ada wajah yang terdeteksi. Pastikan satu wajah terlihat jelas."
                elif result.get("status") == "multiple_faces":
                    message = "Lebih dari satu wajah terdeteksi. Pastikan hanya satu wajah di depan kamera."
                elif result.get("status") == "cascade_error":
                    message = "Model deteksi wajah OpenCV tidak tersedia. Periksa instalasi OpenCV dan file haarcascade."
                flash(message, "danger")
                return redirect(url_for("registrasi_wajah_detail", karyawan_id=karyawan_id))

            saved_info = result.get('path')
            if saved_info:
                karyawan.face_image = saved_info
                db.session.commit()
                flash("Wajah berhasil diregistrasi.", "success")
            else:
                flash("Terjadi kesalahan saat menyimpan foto wajah.", "danger")
            return redirect(url_for("registrasi_wajah"))

        face_image_url = get_face_image_url(karyawan, app)
        return render_template("registrasi_wajah_detail.html", karyawan=karyawan, face_image_url=face_image_url)

    @app.route("/karyawan/tambah", methods=["GET", "POST"])
    def tambah_karyawan():
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        if request.method == "POST":
            nama = request.form.get("nama", "").strip()
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            no_telepon = request.form.get("no_telepon", "").strip()
            jabatan = request.form.get("jabatan", "").strip()
            password = request.form.get("password", "")

            if not nama or not username or not password:
                flash("Nama, username, dan password wajib diisi.", "danger")
                return redirect(url_for("tambah_karyawan"))

            if Karyawan.query.filter_by(username=username).first():
                flash("Username sudah digunakan.", "danger")
                return redirect(url_for("tambah_karyawan"))

            karyawan = Karyawan(
                nama=nama,
                username=username,
                email=email,
                no_telepon=no_telepon,
                jabatan=jabatan,
            )
            karyawan.set_password(password)

            if "foto" in request.files:
                file = request.files["foto"]
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(f"{username}_{file.filename}")
                    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                    karyawan.foto = filename

            db.session.add(karyawan)
            db.session.commit()

            flash(f"Karyawan {nama} berhasil ditambahkan.", "success")
            return redirect(url_for("daftar_karyawan"))

        return render_template("karyawan_add.html")

    @app.route("/karyawan/<int:id>/edit", methods=["GET", "POST"])
    def edit_karyawan(id):
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan = Karyawan.query.get_or_404(id)

        if request.method == "POST":
            karyawan.nama = request.form.get("nama", "").strip()
            karyawan.email = request.form.get("email", "").strip()
            karyawan.no_telepon = request.form.get("no_telepon", "").strip()
            karyawan.jabatan = request.form.get("jabatan", "").strip()

            password = request.form.get("password", "").strip()
            if password:
                karyawan.set_password(password)

            if "foto" in request.files:
                file = request.files["foto"]
                if file and file.filename and allowed_file(file.filename):
                    if karyawan.foto and os.path.exists(
                        os.path.join(app.config["UPLOAD_FOLDER"], karyawan.foto)
                    ):
                        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], karyawan.foto))
                    filename = secure_filename(f"{karyawan.username}_{file.filename}")
                    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                    karyawan.foto = filename

            db.session.commit()
            flash(f"Karyawan {karyawan.nama} berhasil diperbarui.", "success")
            return redirect(url_for("daftar_karyawan"))

        return render_template("karyawan_edit.html", karyawan=karyawan)

    @app.route("/karyawan/<int:id>/hapus")
    def hapus_karyawan(id):
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan = Karyawan.query.get_or_404(id)
        nama = karyawan.nama
 
        # Hapus data absensi terkait, file foto, dan file wajah secara transactional
        try:
            # Hapus semua absensi yang terkait dengan karyawan ini
            Absensi.query.filter_by(karyawan_id=karyawan.id).delete()

            # Hapus file foto jika ada
            if karyawan.foto:
                foto_path = os.path.join(app.config["UPLOAD_FOLDER"], karyawan.foto)
                if os.path.exists(foto_path):
                    try:
                        os.remove(foto_path)
                    except OSError:
                        # Jika file tidak bisa dihapus, lanjutkan
                        print(f"Gagal menghapus file foto: {foto_path}")

            # Hapus file wajah (faces) jika ada
            if getattr(karyawan, "face_image", None):
                face_path = os.path.join(app.config.get("FACES_FOLDER", FACES_FOLDER), karyawan.face_image)
                if os.path.exists(face_path):
                    try:
                        os.remove(face_path)
                    except OSError:
                        print(f"Gagal menghapus file wajah: {face_path}")

            # Hapus objek karyawan
            db.session.delete(karyawan)

            # Commit transaksi
            db.session.commit()

            flash(f"Karyawan {nama} berhasil dihapus.", "success")
            return redirect(url_for("daftar_karyawan"))

        except Exception as e:
            # Debug: tampilkan error dan stack trace lengkap
            print(str(e))
            traceback.print_exc()

            # Rollback transaksi database
            try:
                db.session.rollback()
            except Exception as rb_e:
                print("Rollback error:", rb_e)
                traceback.print_exc()

            flash("Terjadi kesalahan saat menghapus karyawan. Periksa log untuk detail.", "danger")
            return redirect(url_for("daftar_karyawan"))


    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
