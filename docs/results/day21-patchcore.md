# Bàn giao PatchCore ngày 21/11/2026

Bản này ghi thời điểm trước calibration. Phần tích hợp H02/H03, ngưỡng và custom
test được cập nhật tại [kết quả đánh giá chung](shared-evaluation-day21.md).
Gói ZIP uncalibrated cũ giữ nguyên; metadata của artifact làm việc hiện đã được
calibrate bằng evaluator chung.

## Đã chạy thật trên máy Triệu

Python 3.11.9, torch 2.10.0+cu128, Anomalib 2.7.0, GTX 1650 4 GB.
Hai model cùng ResNet-18 ImageNet, layer2/layer3, input 256×256, coreset 1%, 9 neighbors.
Weights chính thức được tải sau khi Triệu đồng ý; hash được kiểm tra trước khi fit.

| Category | Good fit | Normal val | Toàn bộ custom val | Memory bank | Fit giây | Peak CUDA allocated MiB |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| tile | 184 | 46 | 58 | 1884×384 | 19.27 | 730.4 |
| metal_nut | 176 | 44 | 57 | 1802×384 | 16.36 | 698.2 |

Thời gian fit tính từ lúc bắt đầu trích đặc trưng đến khi chọn xong coreset, chưa
gồm khởi tạo backbone, tải weights, export checkpoint hoặc validation. Peak CUDA
allocated là bộ nhớ tensor PyTorch, không phải tổng VRAM/RAM của máy.
Không có kết quả test, ngưỡng đã calibrate hoặc metric chất lượng trong lần bàn giao này.

## Artifact và output local

| Category | Run ID | Checkpoint SHA-256 |
| --- | --- | --- |
| tile | `20261121-202249-seed42-holdout-crack` | `5c6e240bb70aee605b303c79abfa4a2df665ba0535f4c5f6566779b711e15147` |
| metal_nut | `20261121-202715-seed42-holdout-scratch` | `7abe64cff55ca2455fdab377bb7874e290056f6265f8f6945d3aa4fa014fb612` |

Ghép đường dẫn theo quy ước sau, với category và run ID ở bảng:

```text
artifacts/custom_comparison/patchcore/<category>/<run_id>/
outputs/custom_comparison/patchcore/<category>/<run_id>/
outputs/custom_comparison/patchcore/<category>/<run_id>/raw-validation/
```

Artifact chứa `model.ckpt`, `metadata.json`, `manifest.csv`. Không cần cache
pretrained hoặc kết nối mạng để load/predict. Checkpoint khoảng 14 MB/category.
Metadata có trạng thái `uncalibrated`, hash manifest, config, preprocessing,
phiên bản thư viện, tài nguyên đo và danh sách ảnh đã dùng fit.

Fit tile chạy trước khi commit code; fit metal_nut chạy sau commit T03 nhưng config
T05 chưa commit. Metadata giữ đúng SHA lúc chạy và `dirty: true`, không sửa thành
clean sau đó. Code fit/adapter không đổi giữa các lượt full fit và bản code bàn giao.
Gói source và chỉ mục bàn giao local ở `outputs/day21/handoff/` ghi commit code
khi đóng gói; không coi SHA trước run là đủ mô tả toàn bộ source đã chạy.

Output có `config.yaml`, `environment.txt`, `fit-summary.json`, `fit-samples.csv`.
Thư mục `raw-validation` có CSV score, map float32 trên size gốc, preview mỗi loại
lỗi validation và good. Cùng khoảng màu cho cả category/run, không min-max mỗi ảnh.
Ảnh lỗi nằm trong folder test gốc của MVTec nhưng được chọn theo **custom val**;
crack/scratch trong custom test chưa được predict hoặc dùng để chỉnh cấu hình.

## Bằng chứng kiểm tra

- 36 unit/integration tests đạt; lint đạt; notebook đúng schema và cell compile
  trên Python 3.11. Format được kiểm tra cho code mới; giữ nguyên file H01.
- Smoke tile: fit 4 ảnh good, lưu/load và dự đoán trong tiến trình mới.
- Full artifact tile và metal_nut load trong tiến trình mới, xuất lần lượt 58/57
  raw predictions; kiểm tra shape/GT bằng H01 cho từng ảnh.
- Load/predict CPU khi chặn kết nối mạng cho một ảnh good và một ảnh lỗi validation
  mỗi category; so với map CUDA đã lưu bằng atol 1e-3, rtol 1e-4. Chi tiết sai số
  ở `outputs/day21/offline-tile.json` và `offline-metal-nut.json`.
- Không chạy U-Net, calibrate, metric chính thức hoặc thử nghiệm held-out; không
  tự sửa/merge H02/H03. Không có thay đổi GitHub hoặc upload Releases.

## Vai trò của phần đã làm

Fit tạo bộ nhớ hình dạng/bề mặt chuẩn của sản phẩm. Lưu đủ model và memory bank
để lần sau mở app không phải fit lại. Adapter đưa mọi output về giao diện chung,
cho evaluator và app của Hùng dùng mà không biết chi tiết PatchCore. Chạy category
thứ hai bằng config chứng minh đây là một pipeline chung, không phải hai notebook
train riêng. Raw validation giúp phát hiện lỗi kỹ thuật trước khi khóa thí nghiệm.

## Phần còn chờ Hùng

| Phần | Cần nhận | Để làm gì |
| --- | --- | --- |
| H03 #11 / PR #17 | Sửa chặn protocol/fold/test IDs và thứ tự quantile; tích hợp vào main | Dùng chung calibration/metrics mà không rò rỉ dữ liệu |
| H02 #10 / PR #18 | Sửa kiểm tra config fold khớp manifest; có tests và hướng dẫn | Tin được identity và metadata của baseline đối chứng |
| U-Net tile artifact | Releases asset, checksum, config, provenance, output tham chiếu, lệnh chạy | Triệu load lại và so sánh trên cùng test |
| Kiểm tra chéo | Hùng load artifact PatchCore sau khi hai bên thống nhất loader | Nghiệm thu T03, tránh chỉ chạy được trên máy Triệu |

Trình tự sau khi Hùng sẵn sàng và lệnh chạy nằm trong
[hướng dẫn tích hợp](../patchcore-guide.md#khi-hùng-sửa-xong-h02h03-và-upload-artifact-tile).
Đầu việc tiếp theo là nối loader → calibration normal val → kiểm tra U-Net offline
→ khóa cấu hình → đánh giá chung. Giữ nguyên split và không dùng test để đổi model.

T03 #3 còn phần tích hợp/đánh giá/kiểm tra chéo. T05 #5 mới có PatchCore metal_nut,
chưa có EfficientAD. Không đóng issue hay xem các mốc dài hạn là đã hoàn thành.
