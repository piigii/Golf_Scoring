# Chạy VLM trên Apple Metal và NVIDIA CUDA

Xem [../PORTABILITY.md](../PORTABILITY.md) để tạo môi trường theo GPU.
`infer.py` và `detect_club_videos.py` nhận `--backend auto|metal|cuda`,
`--model`, `--mode hybrid|fast|slow`, `--max-tokens`, `--image-token-limit`.
`auto` chọn Metal trên macOS arm64, CUDA trên máy có NVIDIA GPU.
Backend CUDA dùng checkpoint NVIDIA gốc, không nạp trọng số đã chuyển sang MLX.

```bash
python vlm/download_model.py --backend metal  # hoặc cuda
python vlm/infer.py --backend metal --image vlm/examples/golf_impact.jpg
python vlm/detect_club_videos.py --backend metal --input videos_160 --count 2 --output vlm/outputs/metal_run
python vlm/export_performance.py --run vlm/outputs/metal_run
```

Thay `metal` bằng `cuda` và chọn thư mục output mới khi đổi backend/model.
Model được tải theo commit đã ghim. CUDA dùng mã model của NVIDIA thông qua
`trust_remote_code=True`, offline sau khi tải; SDPA không cần MagiAttention.
Video mặc định dùng `mp4v`; có thể chọn `--codec avc1` nếu có encoder H.264.
Resume kiểm tra hash model, source và cấu hình; không trộn lượt chạy khác nhau.
Output cũ trước khi thêm backend cần thư mục mới khi chạy inference lại.
Báo cáo vẫn đọc được kết quả MLX lịch sử.

`probe_club.py` là công cụ thử nghiệm MLX riêng, chỉ dành cho Apple Metal.
Các ghi chú dưới đây mô tả lượt chạy MLX lịch sử, không phải hướng dẫn CUDA.

---

# LocateAnything chạy trên Mac bằng MLX

Model: [eadx/LocateAnything-3B-MLX](https://huggingface.co/eadx/LocateAnything-3B-MLX),
bản BF16 người dùng yêu cầu. Trọng số nằm trong `vlm/models/LocateAnything-3B-MLX`.
Môi trường Python riêng: `vlm/.venv`, độc lập với môi trường GolfPose/SwingNet.

Đã kiểm thử thành công ngày 07/10/2026 trên Apple M1 Pro, RAM 16 GiB,
macOS 26.3, Python 3.11.16 ARM, MLX 0.32.3, MLX-VLM 0.7.6.
Ảnh Impact 354×492: nạp model 5,02 giây, suy luận 8,71 giây, peak memory MLX
9,44 GB (8,79 GiB). Đây là một lượt thử ảnh đơn, không phải benchmark trung bình
hoặc mức RAM toàn hệ thống. Model trả box bao quanh golfer; kết quả nằm ở
`vlm/outputs/smoke_test/`, trạng thái cài đặt ở `vlm/setup_status.json`.

Checkpoint có sáu tensor projector dùng tên cũ. `checkpoint_compat.py` ánh xạ
`mlp1_0/1/3` sang `layer_norm/linear_1/linear_2` khi nạp. Không sửa checkpoint,
không bỏ tensor hoặc nạp model ngẫu nhiên; kiểm tra trọng số vẫn `strict=True`.
Vì vậy hãy dùng `vlm/infer.py` để chạy bản checkpoint này với môi trường đã cài.

## Chạy ảnh đơn

Từ thư mục gốc repository:

```bash
vlm/.venv/bin/python vlm/infer.py \
  --image vlm/examples/golf_impact.jpg \
  --prompt "Locate a single instance that matches the following description: the golfer." \
  --output vlm/outputs/golfer
```

Định vị đầu hoặc tay là các truy vấn riêng cần đánh giá bằng ảnh thật, ví dụ:

```bash
vlm/.venv/bin/python vlm/infer.py \
  --image vlm/examples/golf_impact.jpg \
  --prompt "Point to: the golfer's head." \
  --output vlm/outputs/head
```

Script dùng model local, không tải thêm model khi suy luận. Mặc định xử lý một
ảnh, `--image-token-limit 1024`, `--max-tokens 128`, temperature 0. Giới hạn token
ảnh thay đổi trên processor lúc chạy, không sửa checkpoint đã tải.
Các tùy chọn thêm: `--model`, `--output`, `--max-tokens`, `--image-token-limit`.

Output mỗi lượt: `result.json` (raw output, box/point, thời gian, peak memory MLX),
`response.txt`, `annotated.jpg`. Tọa độ được giải mã theo thứ tự x/y chuẩn của
NVIDIA: box là x1/y1/x2/y2, point là x/y, chuẩn hóa 0–1000. Không có score độ
tin cậy đã hiệu chuẩn cho box/point. Khi không parse được tọa độ hợp lệ, script
giữ raw output để kiểm tra; không tạo điểm giả.

## Detect gậy trên từng frame của hai video đầu

```bash
vlm/.venv/bin/python -u vlm/detect_club_videos.py --count 2
```

Script sắp xếp tên video theo số và chọn `videos_160/0.mp4`, `1.mp4`.
Mỗi frame gốc được đưa riêng vào LocateAnything với prompt `golf club`,
chế độ PBD `hybrid`, tối đa 48 token đầu ra. Không dùng tracker, nội suy
box hoặc bỏ qua frame. Có thể chạy lại cùng lệnh để tiếp tục từ kết quả
đã lưu nếu bị gián đoạn; cấu hình và SHA-256 video phải khớp.
Trên Mac RAM 16 GB, script giới hạn cache MLX 256 MiB và giữ tối đa 8 GiB
bộ nhớ GPU thường trú để giảm swap; chỉ áp dụng cho tiến trình này.

Output mặc định trong `vlm/outputs/golf_club_first2/`:

- `golf_club_first2.mp4`: video tổng hợp có box màu vàng, số frame và thời gian.
- `0/annotated.mp4`, `1/annotated.mp4`: video riêng ở FPS nguồn.
- `0/detections.jsonl`, `1/detections.jsonl`: raw response và box x1/y1/x2/y2
  theo pixel ảnh nguồn, thời gian suy luận của từng frame.
- `0/summary.json`, `1/summary.json`: số frame, số frame có box, thời gian chạy.
- `run_config.json`, `batch_status.json`: cấu hình và tiến độ/kết quả kiểm tra.

Box chiếm trên 80% diện tích ảnh được giữ nguyên và đánh dấu cần xem lại.
Đây chỉ là quy tắc hỗ trợ kiểm tra; số frame có box không phải độ chính xác.
Nguồn 160×160 khiến gậy mảnh hoặc chuyển động nhanh khó quan sát. Video
được phóng lớn để xem, không bổ sung chi tiết ảnh đầu vào cho model.
Tùy chọn `--input`, `--output`, `--count` cho phép thay nguồn và thư mục kết quả.

Lượt thử ngày 07/10/2026 hoàn tất: video `0.mp4` có 138 frame, `1.mp4` có
324 frame; tổng 462/462 frame được suy luận riêng. Video ghép H.264 960×720,
30 FPS, dài 15,4 giây, đã decode kiểm tra đủ frame và không bỏ frame nguồn.
Model trả box hợp lệ về định dạng ở cả 462 frame, nhưng 8 frame có box gần
phủ toàn ảnh (3 ở clip 0, 5 ở clip 1). Một số box khác cũng có thể lệch hoặc
chưa bao đủ gậy; chưa có nhãn chuẩn để đo IoU/độ chính xác. Tổng thời gian
suy luận các frame đã lưu khoảng 14,9 phút, không tính nạp model, lượt bị
ngắt do swap và xuất video. Peak bộ nhớ MLX khoảng 8,91 GiB.
`verification.json` ghi các kiểm tra; `detections.csv` tổng hợp tọa độ và
`review_examples.jpg` đối chiếu ví dụ định vị được gậy với box cần xem lại.

Xuất báo cáo hiệu năng từ kết quả đã lưu (không chạy lại model):

```bash
.venv/bin/python vlm/export_performance.py
```

Lệnh dùng NumPy/Matplotlib có sẵn trong môi trường pipeline. Thư mục
`vlm/outputs/golf_club_first2/performance/` gồm báo cáo HTML độc lập,
Markdown, JSON, CSV tổng hợp/từng frame và biểu đồ PNG/PDF. Thời gian đo
là thời gian dự đoán đã lưu cho từng frame, không bao gồm nạp model, phần
bị ngắt chưa lưu và xuất video. Báo cáo nêu rõ lượt chạy có thay đổi cấu
hình bộ nhớ khi tiếp tục; chưa phải benchmark có điều kiện đồng nhất.

## Tái lập môi trường và tải lại

Cần macOS hỗ trợ Metal, Python ARM native >=3.10. Môi trường đã tạo dùng Python
3.11. Không chạy qua Rosetta. Model BF16 khoảng 7 GB; trên RAM 16 GB nên chạy
ảnh đơn và tránh nạp đồng thời các model lớn khác.

```bash
uv venv --python 3.11 vlm/.venv
uv pip install --python vlm/.venv/bin/python -r vlm/requirements.lock.txt
vlm/.venv/bin/python vlm/download_model.py
```

`model_manifest.json` ghi commit và SHA-256 của các shard sau khi tải đầy đủ.
Downloader có cache các đoạn tải trong `vlm/.cache/ranges`; tải tiếp được khi
bị gián đoạn. Kiểm tra SHA-256 toàn shard trước khi cài trọng số vào thư mục model.
Model, cache, output và môi trường không đưa vào Git.

## Phạm vi

Đây là model grounding trên ảnh và trả box/point 2D, chưa phải tracker video,
model khớp ngón tay hoặc skeleton 3D. Kết quả suy luận cần kiểm tra trước khi
dùng để sửa nhãn GolfPose.

Model giữ giấy phép từ NVIDIA: giới hạn dùng cho nghiên cứu/đánh giá phi
thương mại. Xem [LICENSE gốc](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE).
Mã chạy MLX dùng [MLX-VLM](https://github.com/Blaizzy/mlx-vlm).
