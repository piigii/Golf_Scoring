# Golf Scoring

Workspace phân tích golf swing, chạy trên Apple Silicon (Metal) và NVIDIA CUDA.

| Thư mục | Nội dung |
|---|---|
| `Golf_Detection/` | GolfPose 2D → MixSTE 3D → SwingNet; LocateAnything VLM; báo cáo hiệu năng |
| `GolfPose/` | Hướng dẫn chuẩn bị checkpoint pose |
| `Dataset/` | Hướng dẫn chuẩn bị dữ liệu/video |

## Bắt đầu

```bash
cd Golf_Detection
```

Làm theo [hướng dẫn cài đặt và chạy](Golf_Detection/PORTABILITY.md):

- Apple: PyTorch `--device mps`, VLM `--backend metal`.
- NVIDIA: PyTorch `--device cuda`, VLM `--backend cuda`.
- CPU: pipeline PyTorch `--device cpu`; xuất báo cáo từ kết quả đã lưu.

[Xem tổng quan pipeline](Golf_Detection/README.md),
[pose và swing events](Golf_Detection/PIPELINE.md),
[VLM](Golf_Detection/vlm/README.md).

Model, dataset, môi trường Python, cache và output không nằm trong Git;
chuẩn bị lại theo tài liệu trên mỗi máy. CUDA adapter chưa được kiểm chứng
bằng inference trên NVIDIA GPU thực tế. Pipeline MPS và VLM Metal đã được
kiểm tra trên Apple M1 Pro.

## Git

Chạy lệnh Git tại root `Golf_Scoring/`:

```bash
git status
git add .
git commit -m "Describe changes"
git push origin main
```

Các thành phần GolfDB/GolfPose/LocateAnything giữ giấy phép và attribution
của tác giả gốc; xem README trong từng thành phần và model upstream.
