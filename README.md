# Gemclip

Aplikasi lokal untuk mengubah video YouTube panjang menjadi klip pendek. Hasilnya bisa vertikal (9:16), persegi (1:1), atau lanskap (16:9).

Cara kerjanya:
- **Caption YouTube** dipakai sebagai transkrip jika video punya caption. Jika tidak, **Gemini** mentranskrip audionya.
- **Gemini** memilih momen terbaik, lalu menulis judul, hook, caption, dan hashtag.
- **Gemini** juga melacak posisi wajah pembicara, sehingga area crop mengikuti orang yang berbicara.
- **ffmpeg** dan **OpenCV** merender hasilnya menjadi MP4 dengan caption yang dibakar ke video.

Semua pemrosesan AI berjalan lewat Gemini API. Tidak ada model AI yang dijalankan di komputermu.

## Yang dibutuhkan

- Python 3.10 atau lebih baru
- ffmpeg, dan dapat diakses dari terminal (`ffmpeg -version` harus berhasil)
- Deno atau Node.js, dipakai yt-dlp untuk mengunduh video YouTube
- API key Gemini dari [Google AI Studio](https://aistudio.google.com/apikey)

Instal ffmpeg dan Deno:
- Windows: `winget install Gyan.FFmpeg` dan `winget install DenoLand.Deno`
- macOS: `brew install ffmpeg deno`
- Linux: `sudo apt install ffmpeg`, lalu ikuti panduan di [deno.land](https://deno.land) untuk Deno

## Menjalankan

1. Ekstrak folder ini, lalu jalankan:
   - **Windows:** double-click `run.bat`
   - **macOS / Linux:** `./run.sh`
2. Pada kali pertama, skrip membuat `.env` dari `.env.example`. Isi `GEMINI_API_KEY` di dalamnya, lalu jalankan ulang.
3. Buka **http://127.0.0.1:8765**, tempel link YouTube, atur opsi, lalu klik **Buat klip viral sekarang**.

Hasil tersimpan di folder `output/<id-job>/`:
- `clip_01.mp4`, `clip_02.mp4`, ... (siap diunggah)
- `clip_01.srt`, ... (caption dalam format SRT, bisa dipakai di editor video atau platform yang menerima subtitle)
- `clip_01.txt`, ... (judul, caption, dan hashtag untuk disalin)
- `clips.json` (metadata semua klip, termasuk transkrip per klip)

Video asli yang diunduh disimpan di folder cache, bukan di folder job. Lihat bagian Cache di bawah.

## Opsi di halaman web

| Opsi | Keterangan |
|---|---|
| **Tautan YouTube** | Hanya link YouTube. Upload file lokal belum didukung. |
| **Rasio aspek** | 9:16 (Shorts, TikTok, Reels), 1:1 (Instagram, LinkedIn), atau 16:9 (YouTube, X). Crop mengikuti wajah pembicara untuk semua rasio. |
| **Gaya caption** | Hormozi (tebal, kuning, muncul per kata), Neon glow (cyan bercahaya), atau Minimalist (teks putih sederhana). |
| **Jumlah klip** | 2 sampai 6 di halaman web. Lewat API boleh 1 sampai 20. Gemini bisa mengembalikan lebih sedikit jika tidak ada momen yang layak. |
| **Target durasi klip** | 15 sampai 30 detik, 30 sampai 60 detik (default), atau 60 sampai 90 detik. |
| **Cara analisis** | **Transkrip** (default): Gemini hanya membaca transkrip, sekitar 20 ribu token untuk video 1 jam. **Video**: Gemini menonton video, sekitar 1 juta token untuk video 1 jam, dengan konteks visual dan nada yang lebih kaya. |
| **Mulai / Selesai** | Rentang waktu yang dianalisis, misalnya `10:00` dan `35:00`. Kosongkan untuk seluruh video. |
| **Instruksi untuk AI** | Teks bebas untuk mengarahkan pemilihan momen, misalnya "fokus pada tips keuangan" atau "hindari bagian sponsor". |

Jika instruksi tidak cocok dengan isi video, Gemini bisa mengembalikan daftar kosong dan aplikasi memberi tahu kamu untuk mengubah instruksi atau rentang waktu.

Lewat API (`POST /api/jobs`), durasi minimum dan maksimum bisa diatur sendiri antara 5 dan 180 detik.

## Rasio dan gaya caption

| Rasio | Ukuran hasil | Cocok untuk |
|---|---|---|
| 9:16 | 1080 x 1920 | YouTube Shorts, TikTok, Reels |
| 1:1 | 1080 x 1080 | Instagram feed, LinkedIn |
| 16:9 | 1920 x 1080 | YouTube biasa, X |

Area crop diambil dari video sumber, lalu diskalakan ke ukuran di atas. Untuk 16:9 dari video 16:9, tidak ada yang dipotong, jadi pelacakan wajah tidak mengubah apa pun.

Gaya caption mengatur warna, ketebalan, dan efek teks. Hook ditampilkan di bagian atas klip. Caption juga disimpan sebagai file `.srt` per klip, dengan waktu per kata yang sama.

## Cara analisis: transkrip atau video

Transkrip diambil dari caption YouTube jika ada. Jika tidak, Gemini mentranskrip audio. Pilihan analisis di bawah tidak mengubah sumber transkrip, hanya cara Gemini memilih momen.

- **Transkrip** (default): transkrip dikirim sebagai teks dalam satu permintaan. Setiap kalimat diberi nomor, dan Gemini menjawab dengan nomor kalimat, bukan waktu. Batas klip karena itu selalu jatuh di awal dan akhir kalimat. Ini cepat dan murah, dan video tidak perlu diunggah untuk pemilihan momen. Kelemahannya, Gemini tidak melihat visual atau nada suara, sehingga momen yang bergantung pada tampilan layar atau ekspresi bisa terlewat.
- **Video**: Gemini menonton video secara langsung. Ini lebih kaya konteks, tetapi jauh lebih mahal dan lebih lambat.

Sebagai perkiraan kasar, untuk video 20 menit mode transkrip memakai sekitar 7 ribu token untuk pemilihan momen, sedangkan mode video memakai sekitar 320 hingga 360 ribu token.

## Video panjang (misalnya 1 jam)

Video dan audio dipecah menjadi potongan sekitar 15 menit. Setiap potongan dikirim ke Gemini secara terpisah:
- **Transkripsi** memproses audio per potongan, lalu hasilnya digabung menjadi satu transkrip.
- **Mode video** menganalisis setiap potongan video (dengan tumpang tindih 1 menit), lalu semua kandidat digabung dan dipilih ulang. Pemecahan ini perlu karena satu jam video bisa mendekati 1 juta token, sehingga tidak muat dalam satu permintaan.

Beberapa hal yang perlu diperhatikan:
- **Biaya token**: satu jam audio diperkirakan sekitar 100 ribu token untuk transkripsi, ditambah pemilihan momen. Mode video menambah satu juta token lagi. Di free tier, kuota harian bisa habis lebih cepat. Cek di AI Studio.
- **Waktu proses** sebagian besar ada di Gemini dan jaringan. Komputermu hanya mengunduh dan merender.
- **Waktu render** tetap sebanding dengan jumlah klip, bukan panjang video.

## Cache unduhan dan transkrip

Unduhan dan transkrip disimpan di `output/cache/<id-video>/` dan dipakai ulang:
- Job kedua untuk video yang sama tidak mengunduh ulang dan tidak mentranskrip ulang. Mengubah jumlah klip, durasi, rasio, gaya caption, atau instruksi juga tidak memicu ulang.
- Transkrip disimpan per sumber (caption YouTube atau model Gemini) dan per rentang. Mengubah rentang **Mulai / Selesai** atau `GEMINI_MODEL` memerlukan transkrip baru.
- Transkrip kosong tidak disimpan, supaya kesalahan sesaat tidak terus terulang.
- Cache tidak dihapus otomatis. Video 1 jam di 720p bisa berukuran ratusan MB, jadi hapus folder `output/cache/<id-video>` yang sudah tidak dipakai.
- Link yang tidak dikenali sebagai video YouTube tidak di-cache.

## Konfigurasi (`.env`)

| Variabel | Default | Keterangan |
|---|---|---|
| `GEMINI_API_KEY` | (kosong) | Wajib. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model untuk transkripsi, pemilihan momen, dan pelacakan wajah. Jika muncul error "model not found", cek nama model terbaru di AI Studio. |
| `WORK_DIR` | `output` | Folder hasil. |
| `CLIP_MIN_SEC` / `CLIP_MAX_SEC` | `30` / `60` | Durasi klip default untuk pemanggilan API. Halaman web punya preset durasi sendiri. |
| `DEFAULT_CLIP_COUNT` | `10` | Jumlah klip default untuk pemanggilan API (maksimum 20). Halaman web memulai dari 4 klip. |
| `MAX_VIDEO_HEIGHT` | `720` | Resolusi maksimum video yang diunduh. |
| `USE_YOUTUBE_CAPTIONS` | `1` | Pakai caption YouTube lebih dulu. Isi `0` untuk selalu memakai Gemini. |

## Catatan biaya, privasi, dan batasan

- **Gemini:** free tier punya kuota dan batas harian. Paid tier tidak membatasi durasi video. Cek di AI Studio sebelum memproses banyak video.
- **Perkiraan token pelacakan wajah:** satu frame dikirim tiap 2 detik, jadi klip 60 detik mengirim sekitar 30 frame. Perkiraan kasarnya 8 sampai 15 ribu token per klip 60 detik.
- **Privasi:** audio, video (atau potongannya), dan frame dikirim ke Google untuk diproses. Unduhan dan render tetap berjalan di komputermu.
- **Ketepatan caption karaoke:** caption otomatis YouTube punya waktu per kata, dan dipakai bila tersedia. Caption manual dan transkrip Gemini hanya punya waktu per kalimat, jadi waktu tiap kata diperkirakan dari panjang kata dan bisa sedikit meleset.
- **Caption YouTube:** tidak semua video punya caption, dan caption otomatis bisa salah ketik. Jika caption tidak ada atau gagal diambil, aplikasi memakai Gemini.
- **Pelacakan wajah:** wajah yang sangat kecil, membelakangi kamera, atau sering keluar frame bisa membuat crop kembali ke tengah. Jika pelacakan gagal, klip tetap dibuat dengan crop tengah.
- **Gerak kamera:** crop diam selama wajah masih berada dalam zona toleransi 15% dari lebar crop, lalu bergerak pelan mengikuti wajah. Ini meredam getaran dari deteksi.
- **Sumber video:** hanya video YouTube publik yang didukung.
- **Hak cipta:** hanya klip dari video milikmu atau yang diizinkan pemiliknya. Mengunduh video dari YouTube juga dapat melanggar Terms of Service YouTube. Tanggung jawab penggunaan ada padamu.

## Pemecahan masalah

- **"GEMINI_API_KEY belum diisi"**: isi `.env`, lalu jalankan ulang aplikasi.
- **Gagal mengunduh dari YouTube**: perbarui yt-dlp dengan `pip install -U yt-dlp`, dan pastikan Deno atau Node.js sudah terpasang.
- **"ffmpeg tidak ditemukan"**: pasang ffmpeg dan pastikan dapat dipanggil dari terminal.
- **"Tidak ada ucapan yang terdeteksi"**: rentang waktu yang dipilih mungkin tidak berisi ucapan, atau audionya terlalu sunyi.
- **"Gemini gagal memproses berkas"**: coba lagi beberapa menit kemudian. Jika terus terjadi, cek status layanan Gemini.
- **"Gemini tidak menemukan momen yang cocok"**: longgarkan instruksi, perluas rentang waktu, atau ubah durasi klip.
- **Proses sangat lambat**: persempit rentang waktu, atau kurangi jumlah klip.
- **Hasil pemilihan momen kurang bagus**: coba model Gemini yang lebih besar di `GEMINI_MODEL`.
- **Video lama sudah berubah atau ingin mengunduh ulang**: hapus folder `output/cache/<id-video>`. Job berikutnya akan mengunduh dan mentranskrip dari awal.
- **Ingin selalu memakai Gemini untuk transkrip**: isi `USE_YOUTUBE_CAPTIONS=0` di `.env`, lalu jalankan ulang aplikasi.

## Pengembangan

```
pip install -r requirements-dev.txt
pytest
```

Tes yang butuh ffmpeg dilewati otomatis jika ffmpeg tidak ditemukan. Tes tidak memanggil Gemini atau YouTube sungguhan. Pemanggilan Gemini diuji dengan klien tiruan.

## Struktur kode

| File | Tugas |
|---|---|
| `gemclip/cache.py` | Simpan dan pakai ulang unduhan dan transkrip per ID video |
| `gemclip/download.py` | Unduh video dengan yt-dlp |
| `gemclip/gemini.py` | Klien Gemini dengan timeout, retry untuk error sementara, unggah berkas, dan hapus berkas sementara |
| `gemclip/ytcaptions.py` | Ambil caption YouTube lewat yt-dlp, baca penanda waktu per kata, dan buang baris yang diulang |
| `gemclip/transcribe.py` | Transkrip ucapan dengan Gemini per potongan audio, lalu bagi waktu per kata |
| `gemclip/chunks.py` | Bagi rentang waktu, potong video untuk analisis, dan ambil audio untuk transkripsi |
| `gemclip/moments.py` | Kirim tiap potongan ke Gemini, validasi jawaban, gabungkan dan sesuaikan batas klip ke batas kata |
| `gemclip/reframe.py` | Lacak posisi wajah lewat Gemini, hitung rasio crop, dan buat posisi crop yang halus |
| `gemclip/captions.py` | Buat file caption ASS dengan efek karaoke, dan file SRT |
| `gemclip/render.py` | Potong segmen dan render klip dengan ffmpeg |
| `gemclip/pipeline.py` | Jalankan seluruh proses untuk satu job |
| `gemclip/server.py` | Server web lokal (FastAPI) |
| `gemclip/static/index.html` | Antarmuka web (satu berkas, tanpa CDN) |
