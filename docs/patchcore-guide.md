# Chạy PatchCore và nối phần của Hùng

## Phần đã triển khai

T03 dùng lõi `PatchcoreModel` của Anomalib 2.7.0, không dùng Lightning Engine.
Lý do là cần kiểm soát chính xác ảnh fit, preprocessing và bước chọn ngưỡng.
Không có optimizer hay backpropagation: model trích đặc trưng ảnh good rồi dùng
k-center greedy chọn các patch đại diện vào memory bank.

Baseline dành cho GTX 1650 4 GB dùng ResNet-18 pretrained ImageNet, layer2/layer3,
ảnh 256×256, batch 4, coreset 1% và 9 neighbors. Đây là cấu hình tiết kiệm tài nguyên,
khác backbone Wide ResNet-50 của cấu hình PatchCore thường dùng. Chất lượng phải
được đo sau; không coi tên thuật toán là bằng chứng đạt kết quả từ paper.

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
| Cấu hình | `configs/anomaly/patchcore_tile.yaml`, sau đó config metal_nut ở T05 |
| Notebook gọi script | `notebooks/anomaly/patchcore_baseline.ipynb` |

`scripts/predict.py` hiện dành cho raw PatchCore, chưa phải CLI inference cuối cho
mọi model. Khi H03 merge, đổi điểm load model sang loader chung và giữ các lệnh
đang dùng tương thích. Không thêm một registry cạnh tranh với `PREDICTORS` của Hùng.

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
.\.venv\Scripts\python.exe scripts/predict.py --artifact-dir <artifact_dir> --manifest splits/seed42/custom_comparison_tile_crack.csv --dataset-root datasets --device cuda --output-dir <new_output_dir>
.\.venv\Scripts\python.exe scripts/predict.py --artifact-dir <artifact_dir> --image <image_path> --device cpu --output-dir <new_output_dir>
```

Mỗi run dùng ID theo giờ Asia/Bangkok; không ghi đè run hoặc output đã tồn tại.
Smoke dùng 4 ảnh good, được ghi `smoke_test: true`, không đem làm baseline cuối.
Fit đầy đủ tile dùng 184 good train; 46 good val dành cho calibration sau này.
Config fold/category/protocol phải khớp manifest. Train không decode ảnh val/test;
chỉ đọc danh sách đường dẫn để kiểm tra manifest và chọn good train.

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

## Khi Hùng sửa xong H02/H03 và upload artifact tile

1. Review lại H03: evaluate phải chặn sai protocol/fold và test trùng dữ liệu phát
   triển. Quantile phải theo thứ tự CLI → config artifact → mặc định, lưu giá trị
   thật. Review H02: config fold phải khớp held_out_defect, có regression tests.
2. Tích hợp H03 trước H02 vì H02 dựa trên H03. Đưa cả hai về main mới nhất theo
   trạng thái PR lúc đó; không merge hoặc cherry-pick cả một branch một cách mù quáng.
3. Dùng lại `runs.py`, overlay và loader của H03 sau khi review. Helper trong
   `models/anomaly/train.py` hiện chỉ phục vụ PatchCore trong lúc chờ; gom phần
   thực sự trùng vào helper chung khi tích hợp, giữ kiểm tra provenance và đường dẫn.
4. Đăng ký `PatchcorePredictor` vào `inference/loader.py` của Hùng bằng lazy import.
   Constructor đã nhận `(artifact_dir, device=None)`, có `metadata`, `model_id`,
   `category`, `run_id`; `predict(uint8 RGB)` trả `RawPrediction` tại size ảnh gốc.
5. Chạy lại test loader/CLI, gồm load PatchCore và U-Net, sai category/fold/manifest,
   thiếu checkpoint, checksum sai và artifact chưa calibrate. Xác nhận evaluator
   không chuẩn hóa score khác nhau giữa các model.
6. Calibrate PatchCore tile bằng 46 good val và metal_nut bằng 44 good val. Dùng
   quantile ảnh 0.95/pixel 0.995 đã chốt; giữ raw map và image score native. Lưu
   ngưỡng, quantile, số mẫu/pixel và manifest hash trong metadata theo schema H03.
7. Tải artifact U-Net tile từ Releases sau khi kiểm tra checksum Hùng cung cấp.
   Load offline, kiểm tra preprocessing/fold/hash và thử một ảnh validation. So
   với output tham chiếu Hùng bàn giao; sai số số thực giữa CPU/GPU cần được ghi rõ.
   Không cần train lại chỉ vì bắt đầu tích hợp trên máy Triệu.
8. Hùng cần bàn giao thêm config thực tế, commit/dirty status, checksum, lệnh load,
   calibration/evaluate, metrics và predictions.csv. Nếu H03 sửa ảnh hưởng ngưỡng
   hoặc sample IDs, chạy lại phần bị ảnh hưởng và giữ rõ phiên bản số liệu cũ/mới.
9. Khóa config/model/ngưỡng rồi đánh giá PatchCore bằng cùng custom test IDs với
   U-Net: tile 67 ảnh, metal_nut 61 ảnh khi U-Net category đó sẵn sàng. Tách good,
   known và held-out; AUROC held-out cần có good làm lớp âm. Không đánh giá U-Net
   custom trên official test có chứa ảnh từng dùng train/val.
10. Dùng evaluator chung xuất metric, predictions, mask và overlay; đo latency có
    warm-up/CUDA synchronization. Chọn ví dụ tốt và failure cases cho slide.
    Kết quả test đã dùng để sửa cấu hình phải ghi là thăm dò.

Hùng chưa có artifact không chặn fit/load/raw prediction của PatchCore. H03 chưa
sửa chặn nghiệm thu calibration/metrics chung, không chặn phần lõi anomaly.
Chưa có kết quả nào chứng minh model nào tốt hơn; không đặt trước kết luận U-Net
chắc chắn bỏ sót crack/scratch.

## Phạm vi issue

T03 #3 mới hoàn tất phần fit/load/raw prediction khi chưa có H03 và kiểm tra chéo.
T05 #5 ở đợt này chỉ làm PatchCore metal_nut. EfficientAD thuộc T04 và phần còn lại
của T05; mọi fold unseen và study lượng ảnh good vẫn là T06/T07. Không đóng các
issue dài hạn dựa trên kết quả một phần cho deadline.
