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
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_file
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
import traceback
from werkzeug.utils import secure_filename

from config import Config
from extensions import db
from models import Absensi, Karyawan

UPLOAD_FOLDER = "absensi_app/static/uploads"
FACES_FOLDER = os.path.join(UPLOAD_FOLDER, "faces")
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


def save_face_image(face_b64, filename):
    header, encoded = face_b64.split(",", 1)
    try:
        image_data = base64.b64decode(encoded)
    except Exception:
        return None

    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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


def recognize_face_from_camera(recognizer, label_map, app=None, show_window=False):
    """
    Recognize wajah dengan strict confidence threshold dan multi-frame verification.
    Untuk alur web, jalankan mode silent (tanpa jendela OpenCV) supaya kamera ditutup dengan rapi.
    """
    if app is None:
        threshold = 45.0
        min_frames = 15
        voting_percentage = 70
    else:
        threshold = app.config.get('FACE_CONFIDENCE_THRESHOLD', 45.0)
        min_frames = app.config.get('FACE_MIN_FRAMES', 15)
        voting_percentage = app.config.get('FACE_VOTING_PERCENTAGE', 70)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"status": "camera_error"}

    try:
        if not show_window:
            valid_predictions = []
            all_predictions = []

            for _ in range(min_frames):
                ret, frame = cap.read()
                if not ret:
                    return {"status": "camera_error"}

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                )

                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y + h, x:x + w]
                    try:
                        face_roi = cv2.resize(face_roi, (200, 200))
                        face_roi = preprocess_face_image(face_roi, use_clahe=True)
                        face_roi = normalize_brightness_contrast(face_roi, alpha=1.2, beta=30)

                        label, confidence = recognizer.predict(face_roi)
                        all_predictions.append((label, float(confidence)))

                        if float(confidence) <= threshold:
                            valid_predictions.append((label, float(confidence)))
                    except Exception as e:
                        print(f"Error predicting: {e}")
                elif len(faces) > 1:
                    print("Lebih dari 1 wajah terdeteksi, skip frame ini")

            if len(valid_predictions) == 0:
                return {"status": "unknown", "confidence": 0.0}

            best_label, avg_confidence, vote_count, total_valid_frames = majority_vote_recognition(valid_predictions)
            voting_percentage_result = (vote_count / len(valid_predictions)) * 100 if valid_predictions else 0

            if voting_percentage_result >= voting_percentage and avg_confidence <= threshold and best_label in label_map:
                return {
                    "status": "recognized",
                    "label": best_label,
                    "confidence": float(avg_confidence),
                }

            return {"status": "unknown", "confidence": float(avg_confidence)}

        result = {"status": "cancel"}
        display_name = "SIAP"
        display_confidence = 0.0
        display_status = "MENUNGGU"

        while True:
            ret, frame = cap.read()
            if not ret:
                result = {"status": "camera_error"}
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Tekan SPACE untuk mulai, ESC untuk batal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Nama: {display_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {display_confidence:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status_color = (0, 255, 0) if display_status == "VALID" else (0, 0, 255) if display_status == "UNKNOWN" else (255, 255, 255)
            cv2.putText(frame, f"Status: {display_status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("Absensi Wajah", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                result = {"status": "cancel"}
                break

            if key == 32:
                if len(faces) == 0:
                    result = {"status": "no_face"}
                    break
                elif len(faces) > 1:
                    result = {"status": "multiple_faces"}
                    break
                else:
                    valid_predictions = []
                    all_predictions = []
                    frame_count = 0
                    unknown_start = None

                    while frame_count < min_frames:
                        ret, frame = cap.read()
                        if not ret:
                            result = {"status": "camera_error"}
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                        )

                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        progress_text = f"Mengambil sample... {frame_count + 1}/{min_frames}"
                        cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.imshow("Absensi Wajah", frame)
                        cv2.waitKey(50)

                        if len(faces) == 1:
                            (x, y, w, h) = faces[0]
                            face_roi = gray[y:y + h, x:x + w]
                            try:
                                face_roi = cv2.resize(face_roi, (200, 200))
                                face_roi = preprocess_face_image(face_roi, use_clahe=True)
                                face_roi = normalize_brightness_contrast(face_roi, alpha=1.2, beta=30)

                                label, confidence = recognizer.predict(face_roi)
                                all_predictions.append((label, float(confidence)))

                                if float(confidence) <= threshold:
                                    valid_predictions.append((label, float(confidence)))
                                    frame_count += 1
                                    unknown_start = None
                                else:
                                    print(f"[SKIP] Frame: Confidence {confidence:.2f} > Threshold {threshold:.2f} - UNKNOWN")
                                    if unknown_start is None:
                                        unknown_start = time.time()
                                    else:
                                        elapsed = time.time() - unknown_start
                                        if elapsed >= 2.0:
                                            return {"status": "unknown", "confidence": float(confidence)}
                            except Exception as e:
                                print(f"Error predicting: {e}")
                                continue
                        elif len(faces) > 1:
                            print("Lebih dari 1 wajah terdeteksi, skip frame ini")

                    if result.get("status") == "camera_error":
                        break

                    if len(valid_predictions) == 0:
                        return {"status": "unknown", "confidence": 0.0}

                    best_label, avg_confidence, vote_count, total_valid_frames = majority_vote_recognition(valid_predictions)
                    voting_percentage_result = (vote_count / len(valid_predictions)) * 100 if valid_predictions else 0

                    if voting_percentage_result >= voting_percentage and avg_confidence <= threshold and best_label in label_map:
                        result = {
                            "status": "recognized",
                            "label": best_label,
                            "confidence": float(avg_confidence),
                        }
                    else:
                        result = {"status": "unknown", "confidence": float(avg_confidence)}
                    break

        return result
    finally:
        cap.release()
        cv2.destroyAllWindows()


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
        absensi_list = Absensi.query.filter_by(karyawan_id=user.id).order_by(Absensi.tanggal.desc()).limit(10).all()
        total_absensi = Absensi.query.count()
        total_karyawan = Karyawan.query.count()
        today = date.today()
        total_absensi_today = Absensi.query.filter_by(tanggal=today).count()
        jumlah_hadir = total_absensi_today
        jumlah_belum_hadir = max(total_karyawan - jumlah_hadir, 0)

        return render_template(
            "dashboard.html",
            user=user,
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

        # Face recognition automatically starts when accessing halaman absensi.
        try:
            recognizer, label_map = build_face_recognizer()
        except ImportError as e:
            flash(str(e), "danger")
            return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

        if recognizer is None:
            flash(
                "Belum ada wajah terdaftar. Silakan lakukan registrasi wajah terlebih dahulu.",
                "warning",
            )
            return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

        result = recognize_face_from_camera(recognizer, label_map, app, show_window=True)

        if result["status"] == "cancel":
            flash("Absensi dibatalkan oleh pengguna.", "warning")
            return redirect(url_for("absensi"))

        if result["status"] == "camera_error":
            flash("Kamera tidak dapat diakses. Pastikan webcam terpasang dan tidak digunakan aplikasi lain.", "danger")
            return render_template("absensi.html", user=user, hari_ini=hari_ini, today=today)

        if result["status"] == "no_face":
            flash("Tidak ada wajah yang terdeteksi. Pastikan hanya satu orang berada di depan kamera.", "danger")
            return redirect(url_for("absensi"))

        if result["status"] == "multiple_faces":
            flash("Lebih dari satu wajah terdeteksi. Pastikan hanya satu orang berada di depan kamera.", "danger")
            return redirect(url_for("absensi"))

        if result["status"] == "recognition_error":
            flash("Terjadi kesalahan saat mengenali wajah. Coba lagi.", "danger")
            return redirect(url_for("absensi"))

        if result["status"] == "unknown":
            flash(
                "Wajah tidak dikenali. Silakan registrasi terlebih dahulu.",
                "danger",
            )
            return redirect(url_for("absensi"))

        if result["status"] == "recognized":
            recognized_id = result["label"]
            recognized_karyawan = label_map.get(recognized_id)
            confidence = result["confidence"]

            if not recognized_karyawan:
                flash("Wajah dikenali tetapi data karyawan tidak ditemukan.", "danger")
                return redirect(url_for("absensi"))

            existing_absensi = Absensi.query.filter_by(
                karyawan_id=recognized_karyawan.id,
                tanggal=today,
            ).first()

            # Jika belum ada absensi hari ini, buat record baru dengan jam_masuk
            if not existing_absensi:
                absensi = Absensi(
                    karyawan_id=recognized_karyawan.id,
                    tanggal=today,
                    jam_masuk=datetime.now().time(),
                    keterangan="Hadir",
                )
                db.session.add(absensi)
                db.session.commit()
                flash(
                    f"Absensi masuk berhasil untuk {recognized_karyawan.nama}. Confidence: {confidence:.2f}",
                    "success",
                )
                return redirect(url_for("dashboard"))

            # Jika sudah ada record dan jam_pulang masih kosong, isi jam_pulang
            if existing_absensi and not existing_absensi.jam_pulang:
                existing_absensi.jam_pulang = datetime.now().time()
                db.session.commit()
                flash(
                    f"Absensi pulang berhasil untuk {recognized_karyawan.nama}. Confidence: {confidence:.2f}",
                    "success",
                )
                return redirect(url_for("dashboard"))

            # Jika sudah ada record dan jam_pulang sudah terisi, tolak
            if existing_absensi and existing_absensi.jam_pulang:
                flash(
                    f"Absensi hari ini untuk {recognized_karyawan.nama} sudah lengkap (masuk dan pulang).",
                    "warning",
                )
                return redirect(url_for("dashboard"))

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

    @app.route("/registrasi-wajah/<int:karyawan_id>")
    def registrasi_wajah_detail(karyawan_id):
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("login"))

        karyawan = Karyawan.query.get_or_404(karyawan_id)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Kamera tidak ditemukan. Pastikan webcam terpasang dan tidak digunakan aplikasi lain.", "danger")
            return redirect(url_for("registrasi_wajah"))

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Target: 30-50 sampel dengan berbagai sudut (default 40)
        target_samples = 40
        instructions = [
            "Hadap lurus ke kamera (10 sampel)",
            "Hadap ke KANAN (8 sampel)",
            "Hadap ke KIRI (8 sampel)",
            "Hadap sedikit KE ATAS (7 sampel)",
            "Hadap sedikit KE BAWAH (7 sampel)"
        ]
        instruction_samples = [10, 8, 8, 7, 7]
        
        samples_collected = []
        current_instruction = 0
        current_instruction_count = 0
        result_status = "waiting"
        
        while current_instruction < len(instructions) and len(samples_collected) < target_samples:
            ret, frame = cap.read()
            if not ret:
                result_status = "camera_error"
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display instructions
            total_needed = instruction_samples[current_instruction]
            instruction_text = instructions[current_instruction]
            progress_text = f"Sampel {current_instruction_count}/{total_needed} | Total: {len(samples_collected)}/{target_samples}"
            
            cv2.putText(frame, instruction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, progress_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Tekan SPACE untuk capture otomatis, ESC untuk batal", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Registrasi Wajah - Multiple Samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                result_status = "cancel"
                break
            
            if key == 32:  # SPACE - mulai auto capture
                # Mulai auto-capture untuk instruksi saat ini
                auto_capture_count = 0
                while auto_capture_count < instruction_samples[current_instruction] and len(samples_collected) < target_samples:
                    ret, frame = cap.read()
                    if not ret:
                        result_status = "camera_error"
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                    
                    # Draw rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Status info
                    instruction_text = instructions[current_instruction]
                    progress_text = f"Sampel {auto_capture_count + 1}/{instruction_samples[current_instruction]} | Total: {len(samples_collected) + 1}/{target_samples}"
                    
                    cv2.putText(frame, instruction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, progress_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "Capturing...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("Registrasi Wajah - Multiple Samples", frame)
                    cv2.waitKey(100)  # Delay untuk variasi frame
                    
                    # Capture jika ada exactly 1 wajah
                    if len(faces) == 1:
                        (x, y, w, h) = faces[0]
                        face_sample = frame[y:y + h, x:x + w]
                        samples_collected.append(face_sample)
                        auto_capture_count += 1
                        print(f"Captured sample {len(samples_collected)}: instruction {current_instruction}, photo {auto_capture_count}")
                    elif len(faces) > 1:
                        print(f"Skipped frame: {len(faces)} wajah terdeteksi")
                    else:
                        print(f"Skipped frame: tidak ada wajah")
                    
                    # User bisa cancel dengan ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        result_status = "cancel"
                        break
                
                if result_status == "cancel":
                    break
                
                # Pindah ke instruksi berikutnya
                current_instruction += 1
                current_instruction_count = 0

        cap.release()
        cv2.destroyAllWindows()

        if result_status == "cancel":
            flash("Registrasi wajah dibatalkan.", "warning")
            return redirect(url_for("registrasi_wajah"))
        
        if result_status == "camera_error":
            flash("Terjadi kesalahan saat mengakses kamera.", "danger")
            return redirect(url_for("registrasi_wajah"))
        
        if len(samples_collected) < 30:  # Minimal 30 sampel
            flash(f"Hanya berhasil mengumpulkan {len(samples_collected)} sampel. Minimal 30 sampel diperlukan. Coba lagi.", "danger")
            return redirect(url_for("registrasi_wajah"))

        # Simpan sampel ke folder karyawan
        karyawan_face_dir = os.path.join(app.config["FACES_FOLDER"], f"karyawan_{karyawan_id}")
        try:
            os.makedirs(karyawan_face_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {karyawan_face_dir}: {e}")
            flash("Gagal membuat folder penyimpanan. Coba lagi.", "danger")
            return redirect(url_for("registrasi_wajah"))

        # Simpan setiap sampel
        saved_count = 0
        try:
            for idx, sample in enumerate(samples_collected):
                # Preprocess sample before saving to ensure consistency with training
                try:
                    sample_proc = cv2.resize(sample, (200, 200))
                except Exception:
                    sample_proc = sample

                sample_proc = preprocess_face_image(sample_proc, use_clahe=True)
                sample_proc = normalize_brightness_contrast(sample_proc, alpha=1.2, beta=30)

                sample_filename = f"{idx+1:03d}.jpg"
                sample_path = os.path.join(karyawan_face_dir, sample_filename)
                # cv2.imwrite handles grayscale images
                success = cv2.imwrite(sample_path, sample_proc)
                if success:
                    saved_count += 1
        except Exception as e:
            print(f"Error saving samples: {e}")

        if saved_count == 0:
            flash("Gagal menyimpan sampel wajah. Pastikan folder dapat ditulis.", "danger")
            return redirect(url_for("registrasi_wajah"))

        # Update face_image column sebagai reference
        karyawan.face_image = f"karyawan_{karyawan_id}"
        db.session.commit()
        
        flash(f"Wajah berhasil diregistrasi dengan {saved_count} sampel.", "success")
        return redirect(url_for("registrasi_wajah"))

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
