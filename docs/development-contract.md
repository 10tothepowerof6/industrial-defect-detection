# Hợp đồng kỹ thuật chung

## 1. Tài liệu này dùng để làm gì?

Triệu phụ trách anomaly detection. Hùng phụ trách supervised segmentation. Hai
người cùng làm một hệ thống kiểm tra sản phẩm trên `tile` và `metal_nut`.
Tài liệu này chốt cách đặt code, đọc dữ liệu, chạy mô hình và bàn giao kết quả để
người còn lại có thể tìm đúng chỗ và chạy lại được.

Đây là thiết kế sẽ triển khai. Các script, config và định dạng được mô tả bên dưới
cần được tạo qua các đầu việc trong bản phân công. Hiện repo mới có bộ khung.

- [Phân công của Triệu](assignments/trieu.md)
- [Phân công của Hùng](assignments/hung.md)
- [Workflow và phụ thuộc công việc](workflow.md)

Quy tắc dùng chung được sửa tại tài liệu này. Hai bản phân công dẫn về đây.
Khi đổi tên trường, đường dẫn hay cách tính kết quả, người sửa cập nhật code,
tài liệu và ví dụ bị ảnh hưởng trong cùng PR. Người còn lại review trước khi merge.

## 2. Mô hình và trách nhiệm

| Phần việc | Người chính | Người review | Phạm vi bắt buộc |
| --- | --- | --- | --- |
| Dữ liệu, kiểm tra ảnh/mask, chia tập | Triệu | Hùng | Hai category, manifest dùng chung |
| Anomaly detection | Triệu | Hùng | PatchCore, EfficientAD |
| Supervised segmentation | Hùng | Triệu | U-Net, encoder ResNet-18 pretrained ImageNet |
| Giao diện inference và lưu artifact | Triệu | Hùng | Cách load và predict thống nhất |
| Đánh giá và xuất bảng kết quả | Hùng | Triệu | Cả ba mô hình dùng một evaluator |
| Streamlit | Hùng | Triệu | Một app, chọn category và model |
| Thử nghiệm lỗi chưa thấy | Triệu điều phối | Cả hai | Mỗi người chạy mô hình mình phụ trách |
| Khảo sát lượng ảnh good | Triệu | Hùng | Các mức 25%, 50%, 100% |
| Tài liệu kết quả và bàn giao | Cả hai | Review chéo | Chạy lại được phần của nhau |

Ưu tiên khoảng 70% đầu tư nghiên cứu cho anomaly detection. Hùng làm thêm evaluator
và app để khối lượng tổng thể giữa hai người cân đối. Dinomaly là mở rộng đầu tiên.
SegFormer-B0 và video chỉ bắt đầu khi các phần bắt buộc đã bàn giao xong.

Người phụ trách chính nhận việc, giữ tính nhất quán của phần đó và xử lý phản hồi.
Người còn lại có thể sửa khi cần, qua một PR có review của người phụ trách.

## 3. Bản đồ dự án

Các đường dẫn đều tính từ gốc repository.

| Đường dẫn | Nội dung | Người chính |
| --- | --- | --- |
| `datasets/mvtec_ad/` | Dữ liệu gốc ở máy hoặc notebook runtime | Triệu |
| `splits/` | CSV chia tập và metadata chia tập | Triệu |
| `src/industrial_defect_detection/data/` | Đọc ảnh, mask, manifest, kiểm tra dữ liệu | Triệu |
| `src/industrial_defect_detection/models/anomaly/` | Logic tích hợp PatchCore, EfficientAD | Triệu |
| `src/industrial_defect_detection/models/supervised/` | U-Net, loss và vòng train supervised | Hùng |
| `src/industrial_defect_detection/inference/` | Cấu trúc prediction, load artifact, adapter model | Triệu |
| `src/industrial_defect_detection/evaluation/` | Calibration, metrics, xuất bảng kết quả | Hùng |
| `src/industrial_defect_detection/visualization/` | Overlay và màu hiển thị chung | Hùng |
| `configs/anomaly/` | Config cho PatchCore, EfficientAD | Triệu |
| `configs/supervised/` | Config cho U-Net | Hùng |
| `scripts/` | Các lệnh gọi code trong package | Theo từng script |
| `notebooks/anomaly/` | Notebook chạy trên Colab/Kaggle cho anomaly | Triệu |
| `notebooks/supervised/` | Notebook chạy trên Colab/Kaggle cho supervised | Hùng |
| `artifacts/` | Model và thông tin đủ để load lại | Theo mô hình |
| `outputs/` | Log, prediction, bảng số và hình của từng lần chạy | Theo mô hình |
| `app/` | App Streamlit dùng chung | Hùng |
| `docs/results/` | Kết quả đã review để trình bày trên GitHub | Cả hai |

Notebook làm các việc chọn môi trường, gắn dataset, gọi lệnh và xem kết quả.
Logic đọc dữ liệu, train và evaluate nằm trong package để chạy được cả local lẫn
Colab/Kaggle. Đường dẫn dataset được truyền qua config hoặc tham số.

## 4. Dữ liệu và cách chia tập

### 4.1. Kiểm tra trước khi train

Giữ nguyên ảnh và mask tải về. Không chuyển ảnh giữa các thư mục gốc để chia tập.
Việc chọn ảnh được mô tả bằng manifest.

Triệu viết kiểm tra ảnh đọc được, kích thước ảnh/mask, ghép đúng tên mask, số lượng
theo category và loại lỗi. Kiểm tra ảnh trùng bằng nội dung file. Hùng mở một số
overlay ảnh/mask để xác nhận vùng lỗi khớp vị trí. Chốt báo cáo audit trước khi train.

Ảnh `good` có nhãn ảnh bằng 0 và mask toàn 0. Ảnh lỗi có nhãn bằng 1. Mask gốc được
chuyển về 0/1. Khi resize mask, dùng nearest-neighbor để giữ nhãn pixel.

### 4.2. Manifest

Mỗi manifest chỉ thuộc một protocol, category và fold. Mỗi ảnh xuất hiện một lần
trong manifest đó. Ảnh có thể xuất hiện lại ở fold khác theo cách chia đã chốt.

| Cột CSV | Ý nghĩa |
| --- | --- |
| `sample_id` | Định danh ổn định, dùng đường dẫn ảnh tương đối làm ID |
| `image_path` | Đường dẫn tương đối từ `datasets/mvtec_ad/` |
| `mask_path` | Đường dẫn mask tương đối, để trống với ảnh good |
| `category` | `tile` hoặc `metal_nut` |
| `defect_type` | `good` hoặc tên thư mục loại lỗi gốc |
| `is_anomalous` | 0 hoặc 1 |
| `split` | `train`, `val` hoặc `test` |
| `protocol` | `official_ad` hoặc `custom_comparison` |
| `held_out_defect` | Tên loại lỗi giữ riêng, trống ở baseline thông thường |

Metadata đi kèm ghi `schema_version`, seed, tỷ lệ chia, số lượng từng nhóm, nguồn
dataset và SHA-256 của CSV. Mặc định seed 42. Tạo manifest theo thứ tự file ổn định
trước khi shuffle. Kiểm tra trùng ảnh và trùng nội dung giữa các split trong mỗi fold.

### 4.3. Hai protocol

| Protocol | Cách dùng dữ liệu | Dùng để trả lời |
| --- | --- | --- |
| `official_ad` | Giữ toàn bộ test gốc. Chia ảnh good trong train gốc thành 80% train và 20% val | Hai mô hình anomaly hoạt động thế nào khi giữ test gốc? |
| `custom_comparison` | Dùng cùng normal train/val ở trên, normal test lấy từ test/good gốc. Ảnh lỗi được chia riêng như bên dưới | Ba mô hình khác nhau thế nào khi supervised được học lỗi? |

Trong custom baseline, chia ảnh lỗi của từng loại theo 60% train, 20% val, 20% test.
Chia số lượng nguyên theo cách làm tròn có ghi trong metadata. Mỗi loại cần có ảnh
ở đủ ba nhóm. Nếu audit phát hiện nhóm quá ít hoặc ảnh trùng khiến cách chia không
khả thi, cả hai review và cập nhật cách chia trước khi chạy mô hình.

U-Net dùng toàn bộ dòng train hợp lệ. PatchCore/EfficientAD chỉ dùng dòng train
good. Cả ba dùng cùng ảnh good để fit và cùng tập test trong một phép so sánh.
Ảnh val lỗi đã biết có thể dùng để chọn checkpoint U-Net. Ghi rõ sự khác biệt này.

Các bảng official và custom được trình bày riêng. Official ở đây có giữ lại normal
validation nên phải ghi số ảnh thực sự dùng để fit khi đối chiếu paper.

### 4.4. Lỗi chưa từng thấy

Lần lượt giữ riêng từng loại lỗi của mỗi category. Toàn bộ ảnh loại đó chỉ vào
test. U-Net train bằng good và các loại lỗi còn lại. Không dùng loại giữ riêng để
chọn checkpoint, ngưỡng, config, hoặc chọn ví dụ tốt nhất trước khi chốt mô hình.

Để các fold ổn định, tạo một lần cách chia ảnh của từng loại lỗi. Khi loại đó được
giữ riêng, chuyển toàn bộ ảnh của nó vào test trong manifest fold tương ứng.
Các loại khác giữ cách chia baseline. Normal split giữ nguyên qua các fold.

Đánh giá cả ba mô hình trên cùng test của fold. Tách kết quả good, lỗi đã học và lỗi
giữ riêng. Anomaly có thể dùng lại artifact đã fit nếu normal train và cấu hình
không đổi. U-Net phải train lại theo dữ liệu được phép của từng fold.

Nếu supervised vẫn bắt được lỗi giữ riêng hoặc anomaly bỏ sót, ghi nhận kết quả
và tìm ví dụ giải thích. Kết luận dựa trên số liệu đo được.

## 5. Config và các lệnh sẽ triển khai

Config YAML có các nhóm chung `experiment`, `data`, `model`, `training`,
`calibration`, `paths`. Các tham số riêng của thuật toán nằm trong nhóm `model`.

| Nhóm | Thông tin tối thiểu |
| --- | --- |
| `experiment` | Tên, model ID, category, protocol, fold, seed |
| `data` | Dataset root, manifest path, input height/width |
| `model` | Backbone hoặc encoder, tham số riêng |
| `training` | Device, batch size, số bước/epoch và cách chọn checkpoint phù hợp model |
| `calibration` | Cách chọn ngưỡng ảnh/pixel và các giá trị quantile |
| `paths` | Artifact root và output root |

Model ID được chốt là `patchcore`, `efficientad`, `unet_resnet18`. Ban đầu train
riêng theo category. Các đường dẫn tương đối trong config tính từ gốc repo.
Đọc đường dẫn tuyệt đối từ CLI khi chạy trong môi trường notebook.

Các lệnh dưới là giao diện đích. Mỗi PR tạo entry point phải cập nhật ví dụ chạy thật.

```powershell
python scripts/audit_dataset.py --dataset-root datasets/mvtec_ad
python scripts/create_splits.py --dataset-root datasets/mvtec_ad --seed 42
python scripts/train_anomaly.py --config configs/anomaly/patchcore_tile.yaml
python scripts/train_supervised.py --config configs/supervised/unet_resnet18_tile.yaml
python scripts/calibrate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python scripts/evaluate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python scripts/predict.py --artifact-dir <artifact_dir> --image <image_path>
python -m streamlit run app/streamlit_app.py
```

`train` lưu model và config thực tế. `calibrate` chỉ đọc normal val, lưu ngưỡng và
định danh manifest đã dùng. `evaluate` chỉ đọc test, sử dụng ngưỡng đã lưu.
`predict` load artifact và trả kết quả cho ảnh mới. Nếu thiếu config, artifact,
mask bắt buộc hoặc ngưỡng, báo lỗi kèm đường dẫn và cách xử lý.

## 6. Đầu ra inference dùng chung

Triệu tạo cấu trúc `Prediction` trong `inference/contracts.py`. Hùng dùng đúng cấu
trúc này cho U-Net và evaluator. Trước khi có model, dùng một prediction giả nhỏ
để kiểm tra việc nối evaluator, overlay và app.

| Trường | Quy ước |
| --- | --- |
| `image_score` | Số thực hữu hạn. Cao hơn nghĩa là đáng nghi hơn |
| `score_map` | Mảng float32 `[H, W]`, hữu hạn, cao hơn nghĩa là đáng nghi hơn |
| `image_threshold` | Ngưỡng ảnh đã chọn từ normal val |
| `pixel_threshold` | Ngưỡng pixel đã chọn từ normal val |
| `is_defective` | `image_score > image_threshold` |
| `pred_mask` | Mảng bool `[H, W]`, bằng `score_map > pixel_threshold` |
| `category`, `model_id`, `run_id` | Xác định mô hình đang dùng |

H/W là kích thước ảnh gốc trước preprocessing. Đưa score map về kích thước đó bằng
bilinear trước khi threshold. Threshold map đã resize ở cả calibration và inference.
Không tự xoay/crop ảnh đầu vào trong app. Nếu model cần crop, adapter phải ánh xạ
kết quả về đúng ảnh gốc và mô tả cách xử lý phần bị crop.

Anomaly dùng image score native của model. U-Net dùng sigmoid logits để tạo map
xác suất, image score ban đầu là giá trị lớn nhất của map trên ảnh gốc. Ghi quy tắc
đó trong metadata. Khi muốn đổi sang quy tắc gộp khác, chọn bằng validation rồi
chạy lại calibration và cập nhật phiên bản.

Score của các mô hình có thang khác nhau. App hiển thị tên score, giá trị và ngưỡng.
Không gắn nhãn tất cả score là phần trăm tin cậy. Không chuẩn hóa từng ảnh bằng
min/max trước khi tính metric hoặc PASS/DEFECT. Heatmap màu chỉ dùng để hiển thị.

Quyết định ở cấp ảnh và mask có thể khác nhau vì hai ngưỡng riêng. App hiển thị
đúng quyết định cấp ảnh, đồng thời giải thích mask là vùng vượt ngưỡng pixel.

## 7. Calibration và đánh giá

Mặc định dùng normal val của category để chọn ngưỡng cho cả ba mô hình.
Ngưỡng ảnh là quantile 0.95 của image score. Ngưỡng pixel là quantile 0.995 của các
pixel score trên normal val. Có thể tính pixel quantile theo từng chunk để tránh
hết RAM, nhưng phải giữ đúng cùng phép tính và ghi rõ cách tính.

Đây là ngưỡng ban đầu để tái lập. Tỷ lệ false positive trên test phải đo lại.
Pixel quantile không đại diện cho tỷ lệ ảnh báo nhầm. Nếu số normal val ít, ghi số
ảnh cụ thể để người đọc hiểu mức dao động của kết quả. Bất kỳ thay đổi calibration
nào đều cần được chốt bằng validation trước khi xem kết quả test.

Hùng viết một evaluator nhận score/prediction chuẩn hóa ở mục 6 và ground truth.
Không viết ba bộ tính metric riêng trong ba notebook.

| Nhóm | Kết quả bắt buộc |
| --- | --- |
| Cấp ảnh | AUROC, Average Precision, precision, recall, F1 và false-positive rate |
| Cấp pixel | AUROC, Average Precision, Dice và IoU |
| Unseen | Recall riêng lỗi đã biết và lỗi giữ riêng, kèm số ảnh từng nhóm |
| Tài nguyên | Latency, dung lượng artifact, peak RAM/VRAM nếu môi trường đo được |

Tính AUROC/AP cấp ảnh trên toàn bộ test của category/fold. Pixel AUROC/AP dùng
pixel toàn bộ test. Dice/IoU tính từng ảnh rồi lấy trung bình trên ảnh lỗi làm số
chính. Báo riêng Dice/IoU trên ảnh good, với quy tắc hai mask đều rỗng cho điểm 1,
chỉ một mask rỗng cho điểm 0. Metric không xác định vì thiếu lớp phải có giá trị
null và lý do, giữ số lượng mẫu đi kèm.

Ghi latency batch 1 sau warm-up, số lần đo, device và resolution. Đồng bộ CUDA
trước/sau đo khi dùng GPU. Tách model-only và thời gian gồm preprocessing/inference/
postprocessing. Không lấy thời gian tải weights lần đầu làm latency mỗi ảnh.

## 8. Artifact và thư mục kết quả

```text
artifacts/<protocol>/<model_id>/<category>/<run_id>/
outputs/<protocol>/<model_id>/<category>/<run_id>/
```

`run_id` dùng `<YYYYMMDD-HHMMSS>-seed<seed>-<fold>`, với thời gian Asia/Bangkok.
Ví dụ `20261110-143000-seed42-holdout-scratch`. Fold baseline là `baseline`.
Tên loại lỗi dùng đúng dataset. Dùng tên scratch chỉ khi category đó thực sự có.
Nếu run ID đã có, lệnh phải báo lỗi hoặc yêu cầu run ID mới để tránh ghi đè.

| File | Nơi lưu | Nội dung |
| --- | --- | --- |
| `model.ckpt` và file model bổ sung | artifacts | Checkpoint, memory bank và mọi tài nguyên cần để load offline |
| `metadata.json` | artifacts | Schema version, model/category, config, preprocessing, cách tính score, ngưỡng, manifest hash, package versions, commit SHA |
| `config.yaml` | outputs | Config thực tế sau khi áp dụng tham số CLI |
| `environment.txt` | outputs | Python, package versions, OS và thiết bị |
| `metrics.json` | outputs | Metric, số lượng mẫu, protocol/fold, ngưỡng và thông tin timing |
| `predictions.csv` | outputs | Một dòng mỗi ảnh, ID, nhãn, score, ngưỡng, quyết định và đường dẫn map/mask |
| `maps/<sample_key>.npz` | outputs | `score_map` float32 trên ảnh gốc, `sample_key` là SHA-256 của sample_id |
| `masks/<sample_key>.png` | outputs | Mask dự đoán 0/255 |
| `visualizations/` | outputs | Ảnh gốc, ground truth, map và overlay để so sánh |

Calibration thêm trạng thái `calibrated`, hai ngưỡng, phương pháp, validation
manifest hash vào metadata. Khi chưa calibrate, vẫn có thể xuất raw score cho kiểm
tra, nhưng entry point evaluate chính thức và app phải yêu cầu artifact calibrated.

Artifact có thể là thư mục chứa nhiều file. Triệu phải kiểm tra PatchCore load đủ
memory bank. Hùng phải kiểm tra U-Net load đủ kiến trúc và weights. Loader phát hiện
sai model/category/schema và báo lỗi rõ ràng. Ghi cả trạng thái working tree dirty
khi chạy để tránh hiểu nhầm commit SHA đủ mô tả code đã dùng.

Dataset, artifacts và outputs đang được ignore. Mỗi lần bàn giao có một chỉ mục
nhỏ tại `docs/results/` ghi link tải bộ artifact/output, SHA-256 gói tải, commit,
manifest hash và lệnh tái hiện. Chốt nơi lưu chung trước M2, cả hai truy cập được.
Thử load trên môi trường mới để phát hiện phụ thuộc weights nằm trong cache máy cũ.

## 9. Branch, commit, PR và review

Mỗi đầu việc trong bản phân công có một branch và một PR. Tạo branch từ main mới
nhất, dùng tên đã chốt. Commit chính và title PR dùng cùng tiêu đề đã ghi.
PR nên squash khi merge để giữ một commit rõ ràng cho mỗi đầu việc.

| Prefix | Khi dùng |
| --- | --- |
| `feat` | Thêm khả năng mới như reader, train, evaluate, app |
| `fix` | Sửa hành vi sai đã xác nhận |
| `refactor` | Sắp xếp code mà giữ nguyên hành vi |
| `docs` | Phân công, hợp đồng, hướng dẫn và báo cáo kết quả |
| `chore` | Môi trường, dependencies, công cụ và thiết lập repo |
| `test` | Thêm hoặc sửa kiểm tra tự động |

Dùng `chore`, viết số ít. Dạng title là `type(scope): mô tả bằng tiếng Anh`.
Tên branch không có tiền tố tên công cụ hay tên thành viên. Các fix phát sinh dùng
`fix/<van-de-cu-the>`, commit/PR dùng `fix(<scope>): <mo ta loi duoc sua>`.

Mỗi PR ghi mục tiêu, issue liên quan khi đã có, file chính, lệnh đã chạy, kết quả,
artifact nếu cần và thay đổi hợp đồng nếu có. Người còn lại review. Đầu việc chỉ
xong khi code đã tích hợp, hướng dẫn khớp code và người review chạy được ví dụ.
PR có training phải phân biệt kiểm tra pipeline ngắn và thực nghiệm đầy đủ.

## 10. Mốc triển khai và chuẩn bị GitHub

| Mốc | Việc cần xong | Điều kiện bàn giao |
| --- | --- | --- |
| M1 - Nền tảng chung | T01, T02, H01 | Reader, manifest và giao diện prediction thống nhất |
| M2 - Baseline tile | T03, H02, H03 | PatchCore/U-Net train, save, load, calibrate, evaluate được trên tile |
| M3 - Đủ mô hình và category | T04, T05, H04, H05 | PatchCore, EfficientAD, U-Net cho hai category và demo ảnh |
| M4 - Thực nghiệm | T06, T07, H06 | Đủ fold unseen và khảo sát lượng normal data |
| M5 - Bàn giao | T08, H07 | Kết quả, failure cases và tái hiện chéo |
| M6 - Mở rộng | T09 trước, H08 sau | M1-M5 đã hoàn thành |

Txx thuộc Triệu, Hxx thuộc Hùng. T02 bắt đầu trước train để các model cùng bám một
giao diện. H03 có thể phát triển bằng prediction giả khi chưa có model thật.
H02 cần manifest của T01, T03 cần cả manifest và hợp đồng prediction.

Khi hai người duyệt tài liệu, tạo một GitHub Project chung và milestone M1-M5.
Mỗi mã đầu việc thành một issue, một assignee chính và người còn lại review.
Board dùng Backlog, Ready, In progress, In review, Done. Label theo vùng shared,
anomaly, supervised, evaluation, demo và optional. Ghi dependency bằng issue link.
Mỗi người giữ tối đa một đầu việc chính ở In progress để review không bị tồn đọng.

Issue ghi việc cần làm, dependency, file dự kiến và checklist nghiệm thu. PR liên
kết issue bằng `Closes #<number>`. Optional để backlog riêng và chỉ kéo vào sau M5.
Chưa gán ngày vì hai người cần chốt thời gian rảnh và hạn bàn giao thực tế.
