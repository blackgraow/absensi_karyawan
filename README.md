# Aplikasi Absensi Karyawan

Aplikasi web sederhana untuk absensi karyawan menggunakan Flask, SQLAlchemy, Bootstrap 5, dan MySQL.

## Fitur
- ✅ Halaman Login dengan autentikasi password
- ✅ Dashboard dengan ringkasan absensi
- ✅ **Manajemen Karyawan** (CRUD)
  - Tambah karyawan baru
  - Edit data karyawan
  - Hapus karyawan
  - Daftar karyawan dengan foto
- ✅ Upload foto wajah karyawan
- ✅ Struktur proyek rapi dan profesional
- ✅ Template responsive dengan Bootstrap 5
- ✅ Sidebar navigation

## Struktur Folder
```
absensi-karyawan/
├── absensi_app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── uploads/
│   │       └── (foto karyawan tersimpan di sini)
│   └── templates/
│       ├── base.html
│       ├── login.html
│       ├── dashboard.html
│       ├── karyawan_list.html
│       ├── karyawan_add.html
│       └── karyawan_edit.html
├── app.py
├── config.py
├── extensions.py
├── models.py
├── requirements.txt
├── .env
└── README.md
```

## Tech Stack
- **Backend**: Flask 2.3+
- **Database**: MySQL dengan SQLAlchemy ORM
- **Frontend**: Bootstrap 5, Jinja2 Templates
- **Upload**: Werkzeug FileStorage
- **Security**: werkzeug.security (password hashing)

## Pengaturan Database
1. Pastikan MySQL sudah terpasang dan berjalan.
2. Buat database `absensi_karyawan`:
   ```sql
   CREATE DATABASE absensi_karyawan CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```
3. Perbarui file `.env` jika perlu:
   ```
   SECRET_KEY=ubah_secret_anda
   DATABASE_URL=mysql+pymysql://root:password@localhost/absensi_karyawan
   ```

## Cara Menjalankan di VS Code

### 1. Setup Environment
```powershell
# Buka folder proyek di VS Code
# Terminal: Ctrl + `

# Aktivkan virtual environment (jika belum)
.\venv\Scripts\Activate

# Instal dependencies
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```powershell
python app.py
```

### 3. Akses Aplikasi
- Buka browser dan kunjungi: `http://127.0.0.1:5000`
- Default login:
  - **Username**: admin
  - **Password**: admin123

## Menu Aplikasi

### Dashboard
- Ringkasan total absensi
- Waktu saat ini
- Total karyawan
- Riwayat absensi 10 terakhir

### Manajemen Karyawan
- **Daftar Karyawan**: Lihat semua karyawan dengan foto
- **Tambah Karyawan**: Form untuk menambah karyawan baru
  - Isi: Nama, Username, Password, Email, Telepon, Jabatan
  - Upload foto wajah (PNG, JPG, JPEG, GIF)
- **Edit Karyawan**: Ubah data dan foto karyawan
- **Hapus Karyawan**: Hapus karyawan (foto otomatis dihapus)

## Login Credentials
| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |

## File Uploads
- Folder upload: `absensi_app/static/uploads/`
- Format file: PNG, JPG, JPEG, GIF
- Nama file: `{username}_{original_filename}`

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'flask'"
```powershell
pip install -r requirements.txt
```

### Error: "Can't connect to MySQL"
- Pastikan MySQL service berjalan
- Verifikasi credentials di `.env`
- Pastikan database sudah dibuat

### Error: "Table already exists"
```sql
DROP TABLE IF EXISTS absensi;
DROP TABLE IF EXISTS karyawan;
```
Kemudian jalankan `python app.py` lagi untuk recreate tables.

## Development Tips
- Gunakan `debug=True` di `app.run()` untuk development
- Set `SQLALCHEMY_TRACK_MODIFICATIONS = False` di config
- Gunakan `werkzeug.security` untuk password hashing
- Selalu validasi input dari user

## Pengembangan Lebih Lanjut
- Tambah fitur pencatatan masuk/pulang real-time
- Laporan absensi mingguan/bulanan
- Export ke Excel/PDF
- Notifikasi email
- Dashboard admin untuk manajemen user
- Two-factor authentication

---

**Dibuat dengan ❤️ menggunakan Flask, Bootstrap 5, dan MySQL**

