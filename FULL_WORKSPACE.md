# Bản workspace đầy đủ

Mã nguồn nằm trên `main`. Bản sao đầy đủ được lưu trong
[GitHub Release workspace-full-2026-10-07](https://github.com/piigii/Golf_Scoring/releases/tag/workspace-full-2026-10-07),
gồm model, dataset/video, hai môi trường Python, cache, source bên thứ ba và output
có trong workspace tại thời điểm đóng gói. Python standalone macOS ARM64 được
kèm một bản dùng chung để khôi phục hai venv. Git metadata và thông tin đăng nhập
không nằm trong archive.

Archive gzip được chia thành các phần 1 GiB để đáp ứng giới hạn từng asset của
GitHub Releases. `workspace-manifest.json` ghi thứ tự, dung lượng và SHA-256 mỗi
phần; `SHA256SUMS` hỗ trợ kiểm tra thủ công. Release có manifest môi trường
`python-environments.json` ghi phiên bản package của cả hai môi trường.

## Khôi phục

Dùng Python 3.11.8+ hoặc 3.12+ và đủ dung lượng cho archive tải về cùng dữ liệu
được giải nén. Đích phải là thư mục mới/rỗng:

```bash
git clone https://github.com/piigii/Golf_Scoring.git
cd Golf_Scoring
python3 scripts/restore_workspace.py --destination ../Golf_Scoring-full
```

Script tải đủ các phần, kiểm tra SHA-256 rồi giải nén trực tiếp, không tạo thêm
một file archive ghép lớn. Download đã kiểm tra hợp lệ được dùng lại khi chạy lại.
Bản giải nén không kèm lịch sử Git; lịch sử source vẫn ở repository vừa clone.

Trên **macOS Apple Silicon**, có thể chuyển đường dẫn hai venv sang Python kèm theo:

```bash
python3 scripts/restore_workspace.py --destination ../Golf_Scoring-full --relocate-python
../Golf_Scoring-full/Golf_Detection/.venv/bin/python ../Golf_Scoring-full/Golf_Detection/scripts/doctor.py
```

Venv và Python kèm theo là snapshot **macOS ARM64**, không chạy trực tiếp trên
Linux/Windows/NVIDIA. Khi đổi nền tảng, dùng model/dataset đã khôi phục và tạo
venv CUDA mới theo [PORTABILITY.md](Golf_Detection/PORTABILITY.md).
Không thể upload một venv macOS để dùng trực tiếp như venv CUDA.

## Tình trạng dữ liệu

- `Dataset/archive.zip` là file rỗng (0 byte) trong workspace nguồn.
- `Golf_Detection/Dataset/Unconfirmed 843452.crdownload` là file tải dở,
  khoảng 2,96 GB. Archive giữ nguyên file này, không biến nó thành dataset hoàn chỉnh.
- `workspace-manifest.json` liệt kê các archive rỗng/download dở được phát hiện.

## Tạo snapshot mới

```bash
python3 scripts/package_workspace.py --output /tmp/golf-workspace-backup --runtime /path/to/standalone/python
```

Chỉ định đường dẫn Python standalone bằng `python -c 'import sys; print(sys.base_prefix)'`
trong venv nguồn. Không đặt output backup bên trong workspace.
