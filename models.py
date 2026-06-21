from datetime import date, datetime

from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


class Karyawan(db.Model):
    __tablename__ = "karyawan"

    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    no_telepon = db.Column(db.String(20), nullable=True)
    jabatan = db.Column(db.String(100), nullable=True)
    foto = db.Column(db.String(255), nullable=True)
    face_image = db.Column(db.String(255), nullable=True)
    absensi = db.relationship("Absensi", backref="karyawan", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Absensi(db.Model):
    __tablename__ = "absensi"

    id = db.Column(db.Integer, primary_key=True)
    karyawan_id = db.Column(db.Integer, db.ForeignKey("karyawan.id"), nullable=False)
    tanggal = db.Column(db.Date, nullable=False, default=date.today)
    jam_masuk = db.Column(db.Time, nullable=False, default=lambda: datetime.now().time())
    jam_pulang = db.Column(db.Time, nullable=True)
    keterangan = db.Column(db.String(120), default="Hadir")
