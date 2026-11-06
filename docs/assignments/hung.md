# Phân công của Hùng

## 1. Vai trò và mục tiêu

Hùng phụ trách supervised segmentation, evaluator dùng chung và app Streamlit.
Mục tiêu là học từ ảnh good cùng các lỗi đã biết, dự đoán vùng lỗi và tạo một phép
đối chứng rõ ràng với hướng anomaly detection. Triệu review các phần Hùng làm.

Đọc [hợp đồng kỹ thuật chung](../development-contract.md) trước khi bắt đầu.
Các đường dẫn, schema, split, ngưỡng và cách lưu kết quả lấy từ tài liệu đó.
Đọc [phân công của Triệu](trieu.md) để tìm reader, manifests và mô hình anomaly.
Xem [workflow](../workflow.md) để biết đầu việc nào cần làm trước và khi nào nhận
artifact hoặc kết quả từ Triệu.

## 2. Mô hình và phạm vi

| Mức | Mô hình hoặc sản phẩm | Khi nào xong |
| --- | --- | --- |
| Bắt buộc | U-Net với ResNet-18 pretrained ImageNet | Train, lưu, load, calibrate và đánh giá cả hai category |
| Bắt buộc | Evaluator và calibration | Cả ba mô hình dùng cùng cách tính kết quả |
| Bắt buộc | Unseen defect | Train lại U-Net từng fold, so sánh trên test chung |
| Bắt buộc | Streamlit ảnh | Chọn category/model, hiển thị PASS/DEFECT, score, map và mask |
| Mở rộng | SegFormer-B0 | Sau M5 và sau khi nhóm đã ưu tiên thử Dinomaly |

U-Net dự đoán nhị phân normal/defect theo pixel. ResNet-18 là encoder bên trong
U-Net. Tên loại lỗi dùng để chia fold và phân tích kết quả. Chưa yêu cầu xây head
phân loại tên lỗi. Video thuộc phần mở rộng sau demo ảnh.

## 3. Bản đồ code Hùng cần biết

| Muốn tìm | Đường dẫn dự kiến |
| --- | --- |
| Reader và manifest do Triệu làm | Package `data/`, `splits/` |
| Train supervised | `scripts/train_supervised.py` |
| Model, loss và train loop | `src/industrial_defect_detection/models/supervised/` |
| Config U-Net | `configs/supervised/` |
| Notebook chạy cloud | `notebooks/supervised/` |
| Prediction và loader chung | `src/industrial_defect_detection/inference/` |
| Calibration và metric | `src/industrial_defect_detection/evaluation/` |
| Entry point calibration/evaluation | `scripts/calibrate.py`, `scripts/evaluate.py` |
| Overlay/heatmap | `src/industrial_defect_detection/visualization/` |
| Giao diện | `app/streamlit_app.py` |

Hùng cần load được artifact anomaly do Triệu bàn giao bằng loader chung. Khi cần
tìm cách Triệu fit model, xem `scripts/train_anomaly.py`, `configs/anomaly/`, rồi
đi vào package `models/anomaly/`.

## 4. Dữ liệu và protocol

Dùng reader và manifest của T01. Hùng review ghép ảnh/mask, xác nhận mask ảnh good
bằng 0 và các vùng lỗi khớp ảnh. Chia train/val/test trước khi augmentation.
Augmentation hình học áp dụng đồng bộ lên ảnh và mask, chỉ trên train.

- U-Net train bằng good và lỗi đã biết thuộc split train của custom protocol.
- Dùng val lỗi đã biết để chọn checkpoint theo validation Dice trên ảnh lỗi.
  Ghi cả validation loss và kết quả trên good để phát hiện báo nhầm.
- Dùng normal val chọn ngưỡng theo hợp đồng, giống hướng anomaly.
- Held-out defect chỉ xuất hiện trong test của fold đó.
- Giữ cùng test IDs với Triệu. Giữ lại tên defect_type để tổng hợp theo từng lỗi.

Nếu manifest/config mâu thuẫn với category hoặc fold, dừng và báo lỗi. Có thể bắt
đầu train sau khi T01 merge và prediction schema T02 đã chốt.

## 5. Các đầu việc và tên Git đã chốt

Mỗi hàng là một PR. Cột cuối dùng cho commit chính và title PR. Các commit phụ
trong lúc phát triển vẫn dùng prefix mô tả đúng việc, sau đó squash theo title này.

| Mã | Mốc | Branch | Commit chính và title PR |
| --- | --- | --- | --- |
| H01 | M1 | `feat/evaluation-contract-checks` | `feat(evaluation): validate shared predictions and ground truth` |
| H02 | M2 | `feat/unet-tile-baseline` | `feat(supervised): add reproducible U-Net tile baseline` |
| H03 | M2 | `feat/shared-calibration-metrics` | `feat(evaluation): add shared calibration and defect metrics` |
| H04 | M3 | `feat/unet-metal-nut` | `feat(supervised): support metal nut training and inference` |
| H05 | M3 | `feat/streamlit-inspection-demo` | `feat(app): add unified image inspection demo` |
| H06 | M4 | `feat/supervised-unseen-study` | `feat(supervised): add held-out defect training and comparison` |
| H07 | M5 | `docs/supervised-demo-handoff` | `docs(supervised): document results and demo handoff` |
| H08 | M6, optional | `feat/segformer-extension` | `feat(supervised): add SegFormer-B0 comparison` |

### H01 - Kiểm tra đầu vào evaluator

Làm cùng T02, bắt đầu khi cấu trúc Prediction đã được hai người chốt. Tạo kiểm tra
score map/ground truth đúng shape, nhãn đúng kiểu, số hữu hạn và category hợp lệ.
Tạo vài ví dụ giả có đáp án biết trước để kiểm tra nối dữ liệu và cách đưa map về
ảnh gốc. Các kiểm tra thật được giữ trong `tests/` khi bắt đầu có logic.

Hoàn thành khi cùng một cấu trúc prediction dùng được cho anomaly và U-Net,
dữ liệu sai có thông báo rõ, Triệu gọi được kiểm tra này từ adapter của mình.
H01 kiểm tra hợp đồng đầu vào. H03 bổ sung công thức metric và calibration.

### H02 - U-Net trên tile chạy xuyên suốt

Cần T01/T02 và H01. Tạo U-Net qua SMP, encoder ResNet-18 pretrained ImageNet,
một output channel logits. Bắt đầu với BCEWithLogitsLoss cộng Dice loss theo tỷ
lệ 1/1. Ghi loss và các tham số train trong config. Chọn resolution/batch size
qua smoke test theo VRAM, chốt bằng validation trước khi chạy đầy đủ.

Viết entry point train, notebook gọi entry point, lưu checkpoint tốt nhất theo
validation Dice trên ảnh lỗi. Làm rõ optimizer, learning rate, epoch, seed và
augmentation. Train ngắn để kiểm tra loss hữu hạn, ảnh/mask không lệch, rồi train đủ.
Không thêm sigmoid trước BCEWithLogitsLoss.

Thêm U-Net adapter vào loader chung, sigmoid lúc inference và đưa map về ảnh gốc.
Dùng image score max map theo hợp đồng. Khi H03 xong, calibrate và evaluate tile.
Hoàn thành khi Triệu load artifact trên môi trường mới và chạy lại được một kết quả.

### H03 - Calibration và metrics dùng chung

Cần H01, T02. Có thể viết phần lớn bằng prediction giả khi model thật chưa sẵn.
Viết ngưỡng ảnh/pixel từ normal val theo quantile đã chốt. Lưu phương pháp, giá
trị ngưỡng và hash manifest vào metadata, không lấy nhãn test để chọn ngưỡng.

Viết image metrics, pixel metrics, Dice/IoU và tách nhóm seen/unseen theo hợp đồng.
Ghi số ảnh/số pixel và lý do metric không xác định. Xuất `metrics.json`,
`predictions.csv`, map/mask và overlay cùng quy ước. Kiểm tra perfect prediction,
mask rỗng, score đảo chiều, thiếu một lớp và map sai shape bằng ví dụ biết đáp án.

Tạo `calibrate.py`, `evaluate.py` để nhận mọi model qua loader. Triệu review bằng
artifact PatchCore, Hùng review lại bằng U-Net. Hoàn thành khi cùng một lệnh đổi
artifact directory là đánh giá được cả hai, không cần sửa công thức theo model.

### H04 - U-Net trên metal_nut

Cần H02/H03. Thêm config và train riêng category metal_nut. Kiểm tra augmentation
không làm lệch mask và không tạo ảnh vô lý. Dùng cùng script, cách lưu và evaluator.
Ghi so sánh hai category, ảnh lỗi nhỏ và các vùng good dễ báo nhầm.

Hoàn thành khi cả hai artifact U-Net load được từ loader chung, có bảng custom
baseline và đủ metadata/link tải. Phần supervised dùng custom protocol của nhóm.

### H05 - Một app Streamlit cho cả ba mô hình

Cần H03 và artifact mẫu T03/H02. Có thể dựng UI trước khi các model còn lại xong.
Người dùng chọn category và artifact/model có sẵn, tải ảnh, bấm kiểm tra. Hiển thị
ảnh gốc, PASS/DEFECT, score kèm ngưỡng, heatmap và mask/overlay cùng kích thước.

App gọi loader và prediction chung. Cache model theo đường dẫn/định danh artifact,
tránh dùng nhầm model khi đổi category. Báo rõ artifact thiếu hoặc chưa calibrate.
Hiển thị riêng thang score của từng mô hình, giữ cách overlay thống nhất.
Ảnh mới do người dùng tải lên có thể không có ground truth, app vẫn hoạt động.

Hoàn thành khi chọn được PatchCore, EfficientAD và U-Net cho cả hai category sau
T05/H04. Triệu kiểm tra kết quả anomaly giữa app và CLI trên cùng ảnh. Ghi rõ
latency hiển thị đang đo phần nào. Kiểm tra file ảnh hỏng và đổi model liên tiếp.

### H06 - U-Net trên lỗi chưa thấy và bảng so sánh

Cần T01 và baseline U-Net ổn định. Train một model riêng cho từng fold giữ riêng
loại lỗi. Dữ liệu val không có loại đó. Dùng seed 42 trước, giữ config đã chốt qua
các fold. Nếu chạy nhiều seed, ghi từng seed và tổng hợp độ dao động rõ ràng.

Dùng evaluator chung, đối chiếu test IDs với kết quả T06. Viết bước tổng hợp từ
metrics/predictions đã lưu, không nhập tay số vào bảng. Báo known/unseen recall,
good false positives và chất lượng mask theo category/fold. Nếu một nhóm không
có mẫu, để null kèm số lượng thay vì điền 0.

Hoàn thành khi đủ fold, Triệu chạy lại được bước tổng hợp và truy từ một dòng
kết quả về ảnh, prediction, config và artifact tương ứng. Kết quả tốt/xấu đều được giữ.

### H07 - Bàn giao supervised và demo

Viết hướng dẫn train U-Net, tải artifact, calibrate, evaluate và mở app. Chọn ví dụ
good, lỗi đã học, lỗi giữ riêng, bỏ sót và báo nhầm. Ghi giới hạn dữ liệu và ảnh
ngoài phân phối như góc chụp/ánh sáng khác nếu đã quan sát được.

Đưa kết quả đã review vào `docs/results/`, có link tải và checksum. Hùng chạy lại
một artifact anomaly từ hướng dẫn Triệu. Triệu làm tương tự với U-Net. Hoàn thành
khi app và CLI cho cùng quyết định trên ảnh kiểm tra, hướng dẫn khớp code merge.

### H08 - SegFormer-B0 nếu đủ thời gian

Chỉ bắt đầu sau M5 và sau khi nhóm đã ưu tiên T09. Giữ binary segmentation,
manifest, output và evaluator như U-Net. Thêm model adapter/config thay vì tạo
evaluator khác. Chạy smoke test VRAM trước, ghi checkpoint pretrained đã dùng.
Hoàn thành khi có so sánh tái lập trên cùng test và chi phí chạy được ghi rõ.

## 6. Cách chạy và nơi nhận kết quả

Dùng các lệnh trong hợp đồng. Khi triển khai, mỗi PR bổ sung ví dụ thực tế và cách
gắn dataset trong Colab/Kaggle, không hardcode đường dẫn máy Hùng vào package.

```powershell
python scripts/train_supervised.py --config configs/supervised/unet_resnet18_tile.yaml
python scripts/calibrate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python scripts/evaluate.py --artifact-dir <artifact_dir> --dataset-root <dataset_root> --manifest <manifest_csv>
python -m streamlit run app/streamlit_app.py
```

Model nằm trong `artifacts/custom_comparison/unet_resnet18/<category>/<run_id>/`.
Config, số liệu và hình nằm trong `outputs/` theo cùng cấu trúc. Các thư mục này
đang được ignore, nên bàn giao cho Triệu bằng link gói tải và checksum trong chỉ
mục kết quả/PR. Artifact chứa đủ thông tin để load mà không cần notebook train.

## 7. Phụ thuộc và cách phối hợp

Hùng cần reader và manifest T01, prediction/metadata T02, artifact mẫu T03.
Triệu cần H03 để so sánh mô hình chính thức và H05 để kiểm tra demo. Trong lúc chờ
dữ liệu/model, dùng prediction giả để làm evaluator và overlay.

Hùng review việc split và ghép mask của Triệu trước khi train. Triệu review
calibration/metrics trước khi nhóm tin vào số liệu. Khi thay đổi giao diện chung,
cập nhật hợp đồng và ví dụ cho cả hai hướng trong cùng PR.

## 8. Checklist bàn giao của Hùng

- [ ] Triệu biết lệnh train U-Net, config nào được dùng và checkpoint lưu ở đâu.
- [ ] Artifact load được trên môi trường mới, có preprocessing và ngưỡng.
- [ ] Cả ba mô hình dùng chung calibration/evaluator và quy tắc metric.
- [ ] Có link tải và checksum, config, environment, commit và manifest hash.
- [ ] Có đủ hai category và từng fold unseen với cùng test IDs của Triệu.
- [ ] App dùng artifact đã calibrate, kết quả khớp CLI và score hiển thị đúng nghĩa.
- [ ] Có failure cases, giới hạn và kết quả tái hiện chéo.
- [ ] PR đã được review, các ví dụ lệnh và đường dẫn khớp code đã merge.
