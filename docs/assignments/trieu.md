# Phân công của Triệu

## 1. Vai trò và mục tiêu

Triệu phụ trách hướng anomaly detection, dữ liệu chung và giao diện inference.
Mục tiêu là fit mô hình từ ảnh good, phát hiện ảnh đáng nghi và chỉ ra vùng lỗi.
Hùng review các phần Triệu làm, đồng thời dùng lại reader và inference trong
supervised, evaluator và app.

Đọc [hợp đồng kỹ thuật chung](../development-contract.md) trước khi bắt đầu.
Các đường dẫn, schema, split, ngưỡng và cách lưu kết quả lấy từ tài liệu đó.
Đọc [phân công của Hùng](hung.md) để tìm phần evaluator, U-Net và demo.
Xem [workflow](../workflow.md) để biết đầu việc nào cần làm trước và khi nào gửi
artifact hoặc kết quả cho Hùng.

## 2. Mô hình và phạm vi

| Mức | Mô hình hoặc thử nghiệm | Khi nào xong |
| --- | --- | --- |
| Bắt buộc | PatchCore | Fit, lưu, load, calibrate và đánh giá được cả hai category |
| Bắt buộc | EfficientAD | Cùng luồng trên, có so sánh chất lượng và tốc độ với PatchCore |
| Bắt buộc | Unseen defect | Đánh giá từng loại lỗi giữ riêng bằng test chung với Hùng |
| Bắt buộc | Lượng ảnh good | So sánh các mức 25%, 50%, 100% với val/test cố định |
| Mở rộng ưu tiên đầu | Dinomaly | Bắt đầu sau khi các phần bắt buộc của cả nhóm đã hoàn chỉnh |

Ban đầu làm một pipeline xuyên suốt trên tile. Sau đó thêm metal_nut qua config.
Không dùng kết quả test để quyết định ngưỡng hoặc chọn checkpoint.

## 3. Bản đồ code Triệu cần biết

| Muốn tìm | Đường dẫn dự kiến |
| --- | --- |
| Audit ảnh và mask | `scripts/audit_dataset.py`, package `data/` |
| Tạo manifest | `scripts/create_splits.py`, `splits/` |
| Train anomaly | `scripts/train_anomaly.py` |
| Logic PatchCore/EfficientAD | `src/industrial_defect_detection/models/anomaly/` |
| Cấu hình mô hình | `configs/anomaly/` |
| Notebook chạy cloud | `notebooks/anomaly/` |
| Load model và kiểu prediction | `src/industrial_defect_detection/inference/` |
| Tính metric do Hùng làm | `src/industrial_defect_detection/evaluation/` |
| Demo do Hùng làm | `app/streamlit_app.py` |

Triệu cần kiểm tra model của mình đi qua evaluator và app của Hùng bằng giao diện
chung. Khi cần tìm cách Hùng train U-Net, xem `scripts/train_supervised.py` và
`configs/supervised/`, rồi đi vào package `models/supervised/`.

## 4. Dữ liệu và protocol

Triệu chịu trách nhiệm tạo manifest chính thức để cả hai cùng dùng. Hùng kiểm tra
mask và khả năng dùng manifest để train U-Net. Chốt audit trước khi chốt split.

- `official_ad` giữ test gốc, chỉ dành cho benchmark anomaly của nhóm.
- `custom_comparison` có baseline và từng fold giữ riêng một loại lỗi.
- Anomaly chỉ fit bằng good của train, cùng normal train với U-Net.
- Calibrate bằng normal val. Khi dùng Anomalib, kiểm tra datamodule, callback,
  postprocessor và evaluator để test không tham gia lựa chọn trong lúc fit.
- Lưu hash manifest, seed, config, môi trường và commit của từng run.

## 5. Các đầu việc và tên Git đã chốt

Mỗi hàng là một PR. Cột cuối dùng cho commit chính và title PR. Các commit phụ
trong lúc phát triển vẫn dùng prefix mô tả đúng việc, sau đó squash theo title này.

| Mã | Mốc | Branch | Commit chính và title PR |
| --- | --- | --- | --- |
| T01 | M1 | `feat/mvtec-data-manifests` | `feat(data): add MVTec audit and shared split manifests` |
| T02 | M1 | `feat/shared-inference-contract` | `feat(inference): define shared prediction and artifact contracts` |
| T03 | M2 | `feat/patchcore-tile-baseline` | `feat(anomaly): add reproducible PatchCore tile baseline` |
| T04 | M3 | `feat/efficientad-baseline` | `feat(anomaly): add EfficientAD training and inference` |
| T05 | M3 | `feat/anomaly-metal-nut` | `feat(anomaly): support metal nut across anomaly pipelines` |
| T06 | M4 | `feat/anomaly-unseen-evaluation` | `feat(anomaly): evaluate held-out defect folds` |
| T07 | M4 | `feat/anomaly-data-size-study` | `feat(anomaly): add normal training data size study` |
| T08 | M5 | `docs/anomaly-results-handoff` | `docs(anomaly): document results and reproducible handoff` |
| T09 | M6, optional | `feat/dinomaly-extension` | `feat(anomaly): add Dinomaly comparison` |

### T01 - Kiểm tra dữ liệu và tạo split

Đầu vào là dataset tải xong và giải nén theo layout của README gốc. Viết audit đếm
ảnh theo nhóm, phát hiện file hỏng, mask thiếu/sai kích thước và ảnh trùng. Xuất số
liệu để Hùng mở overlay kiểm tra ghép mask. Không sửa trực tiếp dữ liệu gốc.

Viết reader trả ảnh, mask và metadata theo hợp đồng. Viết tạo split seed 42,
official và custom, gồm cả các fold giữ riêng lỗi. Áp dụng cách làm tròn ổn định,
lưu số lượng từng split và hash manifest. Giữ normal split giống nhau cho mọi model.

Hoàn thành khi Hùng đọc cùng manifest, nhìn đúng ảnh/mask và xác nhận không có ảnh
trùng giữa train/val/test của mỗi fold. Có kiểm tra nhỏ cho ghép mask, ảnh good,
split separation và tái tạo manifest. Bàn giao script, CSV, metadata và báo cáo audit.

### T02 - Chốt prediction và cách lưu/load model

Bắt đầu song song phần cuối T01. Tạo `Prediction` trong `inference/contracts.py`,
định nghĩa metadata schema và các kiểm tra shape, finite score, category/schema.
Chuẩn bị một prediction giả để Hùng phát triển evaluator và overlay ngay.

Chốt loader chung theo model ID. Mỗi model cung cấp adapter để loader gọi được.
Ngưỡng và quy tắc preprocessing phải đọc từ artifact, không hardcode trong app.
Viết `scripts/predict.py` khi adapter model thật đầu tiên có ở T03.

Hoàn thành khi Hùng tạo được một prediction hợp lệ từ U-Net giả và dùng cùng
cấu trúc để tính thử metric. Kiểm tra thiếu field, sai kích thước và sai category
báo lỗi rõ ràng. T03/H02 sẽ hoàn thiện đăng ký adapter thật theo schema này.

### T03 - PatchCore trên tile chạy xuyên suốt

Cần T01/T02. Tạo config PatchCore tile, entry point train anomaly và notebook gọi
entry point đó. Làm rõ image resize, backbone, tỷ lệ chọn memory bank và device.
Trước tiên fit một lần nhỏ để kiểm tra lưu/load, rồi mới chạy đầy đủ.

Kiểm tra Anomalib thực sự chỉ đọc good train và không dùng test để chọn ngưỡng.
Lưu toàn bộ memory bank cần thiết. Hoàn thiện adapter PatchCore và predict CLI.
Khi H03 sẵn sàng, dùng calibration/evaluator chung để xuất bảng và heatmap.

Hoàn thành khi Hùng tải artifact vào môi trường mới, dự đoán được cùng ảnh và
chạy lại evaluation bằng hướng dẫn. Bàn giao config, notebook, link artifact/output,
checksum và ví dụ good/lỗi. Ghi rõ nếu mới smoke test hoặc đã chạy đầy đủ.

### T04 - EfficientAD theo cùng luồng

Cần T03 để tái sử dụng entry point và adapter pattern. Thêm config/model adapter,
không tạo một dự án train khác. Ghi rõ teacher weights, dữ liệu phụ mà implementation
cần nếu có, số bước train và cách chọn checkpoint. Dữ liệu phụ phải khai báo nguồn,
cách tải và tách khỏi MVTec test. Thực hiện smoke test trước khi chạy dài.

Hoàn thành khi EfficientAD trên tile đi qua train, save/load, calibrate, evaluate
và predict giống PatchCore. So sánh ban đầu trên cùng test. Bàn giao gói artifact
đủ tài nguyên để Hùng load mà không phụ thuộc cache weights trên máy Triệu.

### T05 - Mở rộng hai mô hình sang metal_nut

Cần T03/T04. Thêm config metal_nut và chạy riêng từng model. Kiểm tra hướng ảnh,
resize và mask sau khi đưa map về ảnh gốc. Giữ cùng reader, trainer entry point và
evaluator. Hoàn thiện cả config custom baseline và official cho hai category.

Hoàn thành khi có bốn cặp model/category, bảng official và custom được phân biệt,
Hùng chọn được từng artifact trong app. Ghi bộ nhớ và thời gian cùng môi trường đo.

### T06 - Đánh giá lỗi chưa thấy cho anomaly

Cần manifest fold T01 và evaluator H03. Fit anomaly bằng normal train cố định.
Với mỗi fold, kiểm tra manifest hash, chạy test bằng ngưỡng hợp lệ và lưu kết quả
riêng. Tái sử dụng model đã fit khi dữ liệu/config không đổi, ghi rõ quan hệ này.

Phối hợp H06 để cùng test IDs từng fold. Tổng hợp recall lỗi giữ riêng, lỗi đã biết
và false-positive rate trên good. Hoàn thành khi đủ mọi loại lỗi của hai category,
không chọn riêng fold tốt, và Hùng đối chiếu được sample IDs với nhánh supervised.

### T07 - Khảo sát lượng ảnh good

Cần pipeline và baseline ổn định. Dùng các tập con lồng nhau 25%, 50%, 100% từ
normal train. Tập nhỏ phải nằm trong tập lớn. Val/test giữ nguyên. Fit và calibrate
lại từng model theo cùng quy tắc. Áp dụng cho cả hai mô hình anomaly và hai category.

Ghi số ảnh thực tế sau làm tròn, seed, metric, bộ nhớ và latency. Nếu có thời gian,
lặp seed để xem kết quả có nhạy với tập con đã chọn không. Hoàn thành khi có bảng
và hình thể hiện ảnh hưởng của số ảnh good, cùng ví dụ lỗi khó khi dữ liệu ít.

### T08 - Kết quả anomaly và bàn giao

Chọn model demo dựa trên chất lượng, false positives, latency và bộ nhớ đo được.
Ghi lý do chọn theo từng category nếu kết quả khác nhau. Trình bày cả lỗi bỏ sót
và báo nhầm. Tách kết quả official/custom, tránh trộn số từ hai protocol.

Cập nhật `docs/results/` với bảng đã review, link tải, checksum và lệnh tái hiện.
Triệu chạy lại một artifact U-Net theo tài liệu Hùng. Hùng chạy lại artifact anomaly.
Hoàn thành khi hai người cùng xác nhận luồng tải artifact tới đánh giá hoạt động.

### T09 - Dinomaly nếu đủ thời gian

Chỉ nhận sau M5. Tích hợp qua config/adapter có sẵn, chạy thử VRAM trước khi train.
Ghi backbone, pretrained weights, resolution và chi phí chạy. Đánh giá cùng protocol
và evaluator. Nếu không phù hợp tài nguyên, ghi kết quả thử và lý do dừng.

## 6. Cách chạy và nơi nhận kết quả

Dùng các lệnh trong hợp đồng, thay bằng config/category tương ứng. Khi triển khai,
mỗi PR bổ sung ví dụ đã chạy thật, bao gồm cách chạy trên Colab/Kaggle.

```powershell
python scripts/train_anomaly.py --config configs/anomaly/patchcore_tile.yaml
python scripts/calibrate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python scripts/evaluate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python scripts/predict.py --artifact-dir <artifact_dir> --image <image_path>
```

Model nằm trong `artifacts/<protocol>/<model_id>/<category>/<run_id>/`. Config, số
liệu và ảnh kết quả nằm trong `outputs/` theo cùng cấu trúc. Gửi Hùng link gói tải,
checksum và lệnh chạy qua PR/chỉ mục kết quả. Các folder local đang được ignore.

## 7. Phụ thuộc và cách phối hợp

Triệu cần Hùng cung cấp calibration, evaluator, overlay và app. Trong lúc chờ,
Triệu kiểm tra raw prediction và lưu/load bằng adapter. Hùng cần reader, manifest,
prediction schema và artifact mẫu từ Triệu để tích hợp.

Nếu phát hiện thay đổi làm Hùng phải sửa code, trao đổi tại issue/PR trước và cập
nhật hợp đồng cùng thay đổi. Gửi Hùng một artifact nhỏ ngay khi T03 chạy được để
test sớm. Với file dùng chung, có một PR phụ trách thay đổi tại một thời điểm.

## 8. Checklist bàn giao của Triệu

- [ ] Hùng biết lệnh fit model, config nào được dùng và normal train lấy từ đâu.
- [ ] Artifact load được trên môi trường mới, có preprocessing và ngưỡng.
- [ ] Metric dùng evaluator chung, test không tham gia chọn model/ngưỡng.
- [ ] Có link tải và checksum, config, environment, commit và manifest hash.
- [ ] Có đủ hai category, các fold unseen và khảo sát lượng ảnh good.
- [ ] Có failure cases, số liệu tài nguyên và giới hạn của kết quả.
- [ ] Hùng đã chạy lại ví dụ anomaly, Triệu đã chạy lại ví dụ U-Net.
- [ ] PR đã được review, các ví dụ lệnh và đường dẫn khớp code đã merge.
