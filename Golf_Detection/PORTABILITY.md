# Cài đặt và chuyển máy

Repository phát hành nằm ở **Golf_Scoring/**. Các lệnh Python dưới đây chạy
trong **Golf_Detection/**, dùng Python **3.11**. Chạy lệnh Git từ thư mục
Golf_Scoring cha. Repository GolfDB nested trên máy cũ được giữ riêng và không đưa lên GitHub.
Giữ môi trường pipeline và VLM riêng vì phiên bản Transformers/Hugging Face
của MLX và CUDA khác nhau. Không copy `.venv` giữa các máy.

## Kiến trúc

- `golf_pipeline.py`: HRNet 2D → MixSTE 3D → SwingNet; một người/một cú swing đã crop.
- `batch_golf_pipeline.py`: xử lý nhiều video, kiểm tra kết quả và ghép video.
- `train.py`, `eval.py`, `test_video.py`: SwingNet gốc, thêm `--device`.
- `vlm/infer.py`, `vlm/detect_club_videos.py`: LocateAnything cho ảnh/từng frame, có resume.
- `vlm/export_performance.py`: JSON/CSV/HTML/Markdown/PNG/PDF từ thời gian đã lưu.
- `device.py`: PyTorch auto chọn CUDA → MPS → CPU. VLM auto chọn Metal trên macOS arm64 hoặc CUDA; không có CPU adapter.

## Apple Silicon: macOS, Python ARM64

Pipeline dùng PyTorch MPS (Metal). MLX dùng Metal riêng cho VLM.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install 'setuptools<81' numpy==1.26.4
python -m pip install chumpy==0.70 --no-build-isolation
python -m pip install -r requirements/metal.txt
python scripts/fetch_golfpose.py
python scripts/doctor.py
python golf_pipeline.py --video test_video.mp4 --device mps --output outputs/metal_pose
```

Đặt checkpoint theo mục bên dưới trước khi chạy inference. Nếu một toán tử MPS
chưa được PyTorch hỗ trợ, có thể thử `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`;
một số toán tử khi đó chạy CPU. Dùng `--device cpu` nếu cần chạy hoàn toàn CPU.
`auto` không tự retry inference sau lỗi GPU. Độ dài sequence và batch của training
vẫn cần giảm nếu thiếu bộ nhớ.

VLM: dùng terminal mới hoặc `deactivate`, tạo môi trường riêng:

```bash
python3.11 -m venv vlm/.venv
source vlm/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r vlm/requirements-metal.txt
python vlm/download_model.py --backend metal
python vlm/infer.py --backend metal --image vlm/examples/golf_impact.jpg --output vlm/outputs/metal_image
python vlm/detect_club_videos.py --backend metal --input videos_160 --count 2 --output vlm/outputs/metal_run
python vlm/export_performance.py --run vlm/outputs/metal_run
```

MLX BF16 cần đủ unified memory; máy M1 Pro 16 GiB đã chạy checkpoint này.
Adapter giới hạn wired memory tối đa 8 GiB và cache 256 MiB trong tiến trình.
Khuyến nghị macOS 15+ để dùng wired memory; không chạy Python Intel qua Rosetta.

## NVIDIA: Linux hoặc WSL2

Cài NVIDIA driver và kiểm tra `nvidia-smi`. Ví dụ dưới dùng wheel CUDA 12.4,
Torch 2.6.0 / torchvision 0.21.0 theo [hướng dẫn PyTorch](https://docs.pytorch.org/get-started/previous-versions/).
Chọn wheel tương thích driver/GPU; CUDA 12.4/Torch 2.6 không phù hợp mọi GPU mới,
đặc biệt Blackwell. Các tổ hợp Torch mới hơn chưa được kiểm chứng với pipeline này.
Native Windows chưa được kiểm chứng; dùng WSL2 cho dependency VLM `decord`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install 'setuptools<81' numpy==1.26.4
python -m pip install chumpy==0.70 --no-build-isolation
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/cuda.txt
python scripts/fetch_golfpose.py
python scripts/doctor.py
python golf_pipeline.py --video test_video.mp4 --device cuda --output outputs/cuda_pose
python batch_golf_pipeline.py --input videos_160 --count 20 --device cuda --output outputs/cuda_batch
```

VLM trong môi trường riêng:

```bash
python3.11 -m venv vlm/.venv-cuda
source vlm/.venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r vlm/requirements-cuda.txt
python scripts/doctor.py
python vlm/download_model.py --backend cuda
python vlm/infer.py --backend cuda --image vlm/examples/golf_impact.jpg --output vlm/outputs/cuda_image
python vlm/detect_club_videos.py --backend cuda --input videos_160 --count 2 --output vlm/outputs/cuda_run
python vlm/export_performance.py --run vlm/outputs/cuda_run
```

Backend CUDA dùng [checkpoint và API chính thức NVIDIA](https://huggingface.co/nvidia/LocateAnything-3B),
SDPA, BF16 nếu GPU hỗ trợ, FP16 nếu không. Không cần build MagiAttention/FlashAttention.
VRAM thực tế phụ thuộc ảnh và token; giảm `--image-token-limit` khi thiếu bộ nhớ.
Không dùng checkpoint MLX cho CUDA. Kết quả hai backend có thể khác nhau.
`--max-tokens 48` phục vụ single-object golf-club; tăng nếu cần nhiều box để tránh cắt output.

## Model, nguồn ngoài và dữ liệu

`python scripts/fetch_golfpose.py` clone mã nguồn tại commit
`0b749c6df4bb7f060ab42fa3dbd71de91bde17c1`, giữ checkout hiện có nếu đúng commit,
báo lỗi nếu khác commit để bảo vệ thay đổi local. Source không được commit lại.

Đặt ba checkpoint trong `models/` (tải từ link ở [PIPELINE.md](PIPELINE.md),
hoặc copy từ máy hiện tại):

```text
models/golfpose_golfer_hrnetw48.pth
models/golfpose_17+5_39.2_32.3_62.8.bin
models/swingnet_1800.pth.tar
```

Inference không cần dataset hoặc MobileNet pretrain riêng. Training cần
`mobilenet_v2.pth.tar`, `data/train_split_*.pkl` và `data/videos_160/` theo README gốc.
Batch mặc định đọc `videos_160/`; dùng `--input /path/to/videos` để đổi.
Chỉ `test_video.mp4` và ảnh ví dụ nhỏ được giữ trong source.

Downloader VLM ghim revision Metal `010e23e085e4a3ce41ba1666b0c6ce356b737e14`,
CUDA `c32291ca5e996f5a7a485845b4f57a233936bba0`. Có thể ghi đè bằng `--revision COMMIT`.
Download manifest lưu revision và SHA-256. CUDA tải cả mã model NVIDIA và
chạy qua `trust_remote_code=True`. Inference dùng model local, không tự tải online.
Để chuyển offline, copy cả thư mục model (config, tokenizer, processor, mã Python,
trọng số), không chỉ file safetensors. Sao lưu phiên bản dependency bằng
`python -m pip freeze > requirements-local.txt` trên máy chạy thành công.
Các lock cũ là snapshot macOS, không dùng để cài CUDA.

Video export mặc định `mp4v` để tương thích wheel OpenCV trên Linux/macOS.
`--codec avc1` dùng khi máy có H.264 encoder. Chọn output mới khi thay backend,
model hoặc cấu hình. Batch pose có thể tái sử dụng output đã hoàn tất; chọn folder
mới khi đo hiệu năng thiết bị khác. VLM kiểm tra hash/config trước resume.

## Kiểm tra

Trong môi trường pipeline:

```bash
python -m unittest discover -s tests -v
python scripts/doctor.py
```

Tests kiểm tra CPU, chọn thiết bị, API adapter CUDA bằng mock, parse tọa độ,
shape pose/events và xuất video. GPU smoke test SwingNet tự chạy khi GPU truy cập
được; skip nếu không có. Mock CUDA không chứng minh inference CUDA thành công.
Trong môi trường VLM, chạy single-image ở trên trước khi xử lý nhiều video.
Báo cáo performance đo thời gian frame đã lưu, không phải end-to-end benchmark,
và chỉ ghi peak memory của tiến trình cuối khi resume.

Đã kiểm chứng ngày 07/10/2026 trên Apple M1 Pro:

- Suite 13 test: 12 đạt trong sandbox, 1 GPU test skip do sandbox không thấy GPU.
  Chạy lại suite 5 test thiết bị ngoài sandbox: 5/5 đạt, gồm SwingNet trên MPS.
- Pipeline đầy đủ với checkpoint thật trên MPS, clip 8 frame: pose 2D/3D hữu hạn,
  giữ đủ frame và xuất video/events. Clip ngắn dùng để kiểm tra thực thi, không đánh giá độ chính xác mốc swing.
- VLM ảnh thật bằng Metal/MLX: nạp checkpoint, hybrid generation và parse box thành công.
- VLM video Metal 8 frame: suy luận từng frame, xuất và decode đủ 8 frame MP4;
  chạy lại xác nhận cache 8/8 frame, không suy luận lại.
- Báo cáo legacy 462 frame xuất lại thành công, số tổng không đổi. Fixture CUDA
  synthetic với 1 và 3 video xuất đủ định dạng, không mang thông tin M1/MLX.
- CUDA adapter đã kiểm tra giao diện bằng mock và đối chiếu mã/checkpoint NVIDIA;
  **chưa chạy inference trên NVIDIA GPU**. Cài đặt từ môi trường trống và training
  đầy đủ trên từng nền tảng chưa được kiểm chứng trong phiên này.

## Đưa lên GitHub

`.gitignore` loại `.venv` và `vlm/.venv*`, model, dataset/video bulk, cache, log, outputs,
`.DS_Store` và cấu hình máy. Giữ nguyên giấy phép/attribution của GolfDB và
các model/source upstream; xem README gốc và LICENSE trong GolfPose/checkpoint.
Không có thay đổi nào cấp lại giấy phép cho checkpoint của bên thứ ba.

```bash
cd /path/to/Golf_Scoring
git status --short
git add .
git diff --cached --stat
git commit -m "Support Apple Metal and NVIDIA CUDA execution"
```

Repository này đã có Git history. Đặt remote GitHub của bạn nếu cần:

```bash
git remote -v
# Chỉ khi chưa có origin:
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin HEAD
```

Máy mới clone repository, cài theo nền tảng, tải source GolfPose và checkpoint,
rồi chạy doctor và smoke test. Không copy thư mục sibling `../GolfPose` hoặc
`../Dataset`; pipeline mặc định lấy mọi asset từ root repository.
