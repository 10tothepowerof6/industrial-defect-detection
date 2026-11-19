# Bàn giao phần ngày 19/11/2026 của Triệu

## Kết quả đã kiểm tra

- Python 3.11.9 trong `.venv`, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128.
- GPU GTX 1650, CUDA runtime 12.8. Matmul và convolution chạy thật trên GPU.
- Import PatchCore, EfficientAD và U-Net thành công. Chưa tải pretrained weights.
- Reader kiểm tra được 682 ảnh và 177 mask. Không phát hiện ảnh hỏng, mask sai
  kích thước, mask thiếu/mồ côi hoặc ảnh trùng theo nội dung pixel đã decode.
- Tile có 347 ảnh 840×840. Metal nut có 335 ảnh 700×700.
- Triệu đã xác nhận giữ riêng crack và scratch trước khi tạo manifest.
- Có schema prediction và ví dụ giả cho Hùng kiểm tra H01.

Báo cáo môi trường và audit đầy đủ nằm trong `outputs/day19/`, đang được Git ignore.
Preview audit ghép ba cột ảnh RGB, mask thật, overlay. Các preview lỗi dùng
glue_strip và bent, không sử dụng crack/scratch để chọn cấu hình model.

## Manifest đã tạo

Seed 42, normal train lấy floor 80% train/good gốc, phần còn lại vào val. Với các
loại lỗi được phép học, lấy floor 60% train, floor 20% val, phần dư vào test.
Trong custom fold, toàn bộ loại lỗi giữ riêng chỉ vào test.

| Manifest | Train | Val | Test | Good train dùng cho anomaly |
| --- | ---: | ---: | ---: | ---: |
| official_ad_tile_baseline | 184 | 46 | 117 | 184 |
| official_ad_metal_nut_baseline | 176 | 44 | 115 | 176 |
| custom_comparison_tile_crack | 222 | 58 | 67 | 184 |
| custom_comparison_metal_nut_scratch | 217 | 57 | 61 | 176 |

CSV và metadata ở `splits/seed42/`. Mỗi metadata có hash CSV và fingerprint ảnh
dataset. Chỉ có hai fold custom đã chốt cho deadline, chưa tạo mọi fold dài hạn.
U-Net dùng toàn bộ dòng train, anomaly lọc `is_anomalous == 0`. Cả hai vẫn đọc
cùng manifest và cùng test IDs.

## Hùng cần review những gì?

1. Mở `outputs/day19/audit/tile_glue_strip.png` và `metal_nut_bent.png` để xác nhận
   mask thật khớp ảnh. Xem thêm hai preview good có mask toàn 0.
2. Kiểm tra reader trả RGB uint8 `[H,W,3]` và mask uint8 0/1 `[H,W]`. Nếu truyền
   size thì thứ tự là `(height,width)`, mask resize bằng nearest-neighbor.
3. Kiểm tra custom manifest không có crack/scratch trong train hoặc val, normal
   split khớp với official manifest. Các file hiện ghi `pending_hung_review`.
4. Dùng prediction giả để bắt đầu H01. `RawPrediction` có score chưa calibrate.
   `Prediction` thêm hai ngưỡng và tự suy ra quyết định cùng mask bằng phép `>`.
   Map phải là float32 hữu hạn ở kích thước ảnh gốc.
5. H03 sẽ chọn ngưỡng bằng normal validation và tính metric. Ví dụ giả chỉ minh
   họa schema, không dùng làm kết quả thực nghiệm.

Hùng chưa review nên T01/T02 vẫn cần nghiệm thu chung. Chưa có model được train,
checkpoint thật, ngưỡng đã calibrate hoặc kết quả đánh giá model trong bàn giao này.
T02 mới có prediction contract. Artifact loader và metadata từ model thật sẽ được
hoàn thiện cùng adapter PatchCore ở T03, không tạo loader trả kết quả giả.

## Kiểm thử đã chạy

- 16 kiểm thử tự động đạt: ghép ảnh/mask, dữ liệu hỏng, ảnh trùng, chia tập,
  rò rỉ lỗi giữ riêng, tái tạo manifest và kiểm tra prediction/ngưỡng.
- `ruff check` và `ruff format --check` đạt; `pip check` không báo dependency hỏng.
- Tạo lại bốn manifest từ dataset thật ở `outputs/day19/reproduced_splits/`.
  Cả tám file CSV/JSON khớp từng byte với bản trong `splits/seed42/`.
- Kiểm tra hash CSV, đọc ảnh/mask qua manifest ở cả train/val/test của bốn bộ.
  Normal split giống nhau giữa official và custom; crack/scratch chỉ ở test.
- Ví dụ prediction giả tạo map 32×48, 96 pixel vượt ngưỡng và quyết định DEFECT.
  Đây chỉ là dữ liệu thử giao diện, không phải kết quả model.

GPU đã chạy phép tính thật nhưng chưa có lượt fit/predict PatchCore hoặc U-Net.
Khả năng chạy một cấu hình train cụ thể trên VRAM 4 GB sẽ được kiểm tra ở bước sau.

## Lệnh chạy lại

Chạy từ gốc repo, dùng môi trường đã cài. Với audit/split/example đã tồn tại, chọn
thư mục output mới để kiểm tra thay vì ghi đè. Hai lần tạo split cùng seed và dữ
liệu phải cho cùng bytes CSV/JSON.

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
.\.venv\Scripts\python.exe scripts/audit_dataset.py --dataset-root datasets --output-dir outputs/recheck/audit
.\.venv\Scripts\python.exe scripts/create_splits.py --dataset-root datasets --seed 42 --tile-held-out crack --metal-nut-held-out scratch --output-dir outputs/recheck/splits
.\.venv\Scripts\python.exe scripts/preview_prediction.py
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

T02 được phát triển trên branch local `feat/shared-inference-contract`, dựa trên
commit T01 ở `feat/mvtec-data-manifests`, để cả dữ liệu và prediction dùng được
cùng nhau trước khi có PR. Khi tạo PR sau này, tích hợp T01 trước rồi đưa T02 lên
main mới nhất. Các commit hiện chỉ ở local.
