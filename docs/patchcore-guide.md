# Chạy PatchCore và nối phần của Hùng

## Phần đã triển khai

T03 dùng lõi `PatchcoreModel` của Anomalib 2.7.0, không dùng Lightning Engine.
Lý do là cần kiểm soát chính xác ảnh fit, preprocessing và bước chọn ngưỡng.
Không có optimizer hay backpropagation: model trích đặc trưng ảnh good rồi dùng
k-center greedy chọn các patch đại diện vào memory bank.

Baseline dành cho GTX 1650 4 GB dùng ResNet-18 pretrained ImageNet, layer2/layer3,
ảnh 256×256, batch 4, coreset 1% và 9 neighbors. Đây là cấu hình tiết kiệm tài nguyên,
khác backbone Wide ResNet-50 của cấu hình PatchCore thường dùng. Kết quả custom
test được ghi riêng; không coi tên thuật toán là bằng chứng đạt kết quả từ paper.

Reader trả RGB; resize bilinear rồi chuẩn hóa bằng ImageNet mean/std. BatchNorm
của backbone luôn ở eval mode khi trích đặc trưng. Không augmentation, crop hoặc
min-max từng ảnh. Điểm ảnh là native score có neighbor reweighting của Anomalib.
Score map được đưa về kích thước ảnh gốc trước khi chuyển cho evaluator.

## File cần tìm

| Việc | File |
| --- | --- |
| Fit | `scripts/train_anomaly.py` |
| Kiểm tra config/manifest và lưu kết quả fit | `src/industrial_defect_detection/models/anomaly/train.py` |
| Load offline và raw prediction | `src/industrial_defect_detection/models/anomaly/patchcore.py` |
| Xuất map/preview validation hoặc một ảnh | `scripts/predict.py` |
| Cấu hình | `configs/anomaly/patchcore_tile.yaml`, `configs/anomaly/patchcore_metal_nut.yaml` |
| Notebook gọi script | `notebooks/anomaly/patchcore_baseline.ipynb` |

`scripts/predict.py` dùng loader chung cho PatchCore và U-Net. Mặc định xuất raw
score/map để giữ tương thích với notebook. Thêm `--calibrated` để xuất quyết định,
mask và overlay bằng ngưỡng trong artifact; artifact chưa calibrate sẽ bị từ chối.

## Điều kiện và lệnh chạy

Dùng môi trường đã cài theo `requirement.txt`. Không tự động tải/cài thư viện hoặc
weights. File pretrained chính thức là
`https://download.pytorch.org/models/resnet18-f37072fd.pth`, đặt tại
`artifacts/backbones/resnet18-f37072fd.pth`. Code kiểm tra SHA-256 prefix `f37072fd`.
Weights chỉ cần lúc fit; artifact xuất ra chứa đủ backbone đã dùng và memory bank.

Manifest phải có hash khớp sidecar và được duyệt `approved_by_hung`. Các thay đổi
trạng thái duyệt của ngày 21 đang có sẵn ở working tree, không thuộc commit code
PatchCore này. Khi bàn giao repo sang máy khác, đưa cả trạng thái nghiệm thu đã
duyệt sang; không tự đổi trạng thái nếu chưa có review.

```powershell
.\.venv\Scripts\python.exe scripts/train_anomaly.py --config configs/anomaly/patchcore_tile.yaml --smoke-test
.\.venv\Scripts\python.exe scripts/train_anomaly.py --config configs/anomaly/patchcore_tile.yaml
.\.venv\Scripts\python.exe scripts/train_anomaly.py --config configs/anomaly/patchcore_metal_nut.yaml
.\.venv\Scripts\python.exe scripts/predict.py --artifact-dir <artifact_dir> --manifest splits/seed42/custom_comparison_tile_crack.csv --dataset-root datasets --device cuda --output-dir <new_output_dir>
.\.venv\Scripts\python.exe scripts/predict.py --artifact-dir <artifact_dir> --image <image_path> --device cpu --output-dir <new_output_dir>
```

Mỗi run dùng ID theo giờ Asia/Bangkok; không ghi đè run hoặc output đã tồn tại.
Smoke dùng 4 ảnh good, được ghi `smoke_test: true`, không đem làm baseline cuối.
Fit đầy đủ tile dùng 184 good train; 46 good val dành cho calibration sau này.
Config fold/category/protocol phải khớp manifest. Train không decode ảnh val/test;
chỉ đọc danh sách đường dẫn để kiểm tra manifest và chọn good train.
Ảnh lỗi validation custom vẫn có đường dẫn `tile/test/...` hoặc
`metal_nut/test/...` theo layout MVTec gốc. Quyết định dùng ảnh theo cột `split`
trong manifest; đường dẫn đó không có nghĩa đang chạy custom test.

Output:

- `artifacts/<protocol>/patchcore/<category>/<run_id>/`: `model.ckpt`, `metadata.json`,
  `manifest.csv`. Checkpoint gồm trọng số backbone và memory bank, có checksum.
- `outputs/<protocol>/patchcore/<category>/<run_id>/`: config, environment, fit summary
  và danh sách chính xác ảnh đã fit. Peak VRAM là allocated bytes do PyTorch đo,
  không phải tổng VRAM hoặc peak RAM của cả tiến trình.
- Output raw prediction: CSV score, `maps/*.npz` chứa float32 map gốc và preview.
  Cùng một khoảng màu cho cả lượt validation; khoảng màu chỉ phục vụ hiển thị.
  Không xuất quyết định PASS/DEFECT, mask dự đoán hoặc metric khi chưa calibrate.
  Thời gian predict ghi trong CSV chỉ để kiểm tra, chưa phải benchmark latency.

H02/H03/H04 đã được tích hợp trên branch local `feat/patchcore-shared-evaluation`.
PatchCore đã calibrate bằng normal val và đánh giá bằng evaluator H03; U-Net tile
đã được tải từ Releases, kiểm tra hash, load offline và tái hiện cùng custom test.
Xem [kết quả tích hợp](results/shared-evaluation-day21.md) để biết số liệu và việc
còn chờ Hùng. Bản bàn giao raw trước đó vẫn được giữ như lịch sử chạy.

## Dùng calibration và evaluator chung

H02/H03/H04 đã vào main qua PR #17 và #19. Loader có cả `patchcore` và
`unet_resnet18`; helper run/provenance và overlay được dùng chung. CLI raw cũ vẫn
hoạt động; `--calibrated` yêu cầu ngưỡng hợp lệ và xuất thêm quyết định/mask.

```powershell
.\.venv\Scripts\python.exe scripts/calibrate.py --artifact-dir <artifact_dir> --dataset-root datasets --manifest <training_manifest_csv> --device cuda
.\.venv\Scripts\python.exe scripts/predict.py --artifact-dir <artifact_dir> --image <image_path> --output-dir <new_output_dir> --device cuda --calibrated
# Khóa config, checkpoint và ngưỡng trước khi chạy test.
.\.venv\Scripts\python.exe scripts/evaluate.py --artifact-dir <artifact_dir> --dataset-root datasets --manifest <training_manifest_csv> --device cuda
```

Calibration chỉ dùng normal validation từ manifest đã ghi trong artifact;
manifest khác hash bị chặn. Evaluate chặn sai protocol/fold và ảnh test trùng
train/val. Quantile theo thứ tự CLI → config artifact → mặc định 0.95/0.995.
Không dùng `--force` để đổi ngưỡng sau khi xem test rồi coi đó là kết quả chưa thăm dò.

Artifact cũ không có ngưỡng vẫn predict raw được. Checkpoint PatchCore chứa đủ
backbone và memory bank để load offline. U-Net tile được tải từ Release
`unet-tile-holdout-crack-v1`, kiểm tra SHA-256 trước khi load. Không cần train lại
U-Net để tái hiện.

Cả hai PatchCore và U-Net tile đã được chạy evaluator chung; config/metadata/hash
khóa trước lượt test tại `outputs/day21/frozen-evaluation/lock.json`.
Xem [kết quả và giới hạn](results/shared-evaluation-day21.md).

## Phần tiếp theo cần Hùng

1. Upload artifact/output U-Net metal_nut và checksum để Triệu tái hiện cùng 61
   test IDs. Số trong PR H04 mới là số Hùng công bố, chưa phải kết quả local.
2. Hùng load hai artifact PatchCore đã calibrate và chạy lại ví dụ/evaluation để
   nghiệm thu chéo. Gói local chưa được publish lên Releases trong lượt này.
3. Xử lý vấn đề score viền của U-Net metal_nut ở PR riêng, giữ baseline cũ. Chọn
   quy tắc mới trên validation, calibrate lại và ghi rõ đã xem test baseline.
4. H05 nối app với loader/`calibrated_predict`; đối chiếu một ảnh giữa CLI và app,
   cache đúng category/model và hiển thị score/ngưỡng, không gọi score là confidence.

Không đặt trước kết luận anomaly luôn tốt hơn supervised hoặc supervised chắc
chắn bỏ sót lỗi mới. Hai hướng cần so sánh cả quyết định cấp ảnh và chất lượng mask.

## Phạm vi issue

T03 #3 đã có fit/load/calibrate/evaluate tile, còn kiểm tra chéo và bàn giao cho Hùng.
T05 #5 ở đợt này chỉ làm PatchCore metal_nut. EfficientAD thuộc T04 và phần còn lại
của T05; mọi fold unseen và study lượng ảnh good vẫn là T06/T07. Không đóng các
issue dài hạn dựa trên kết quả một phần cho deadline.
