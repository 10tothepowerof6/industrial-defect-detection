# Bàn giao phần ngày 20–21 và kế hoạch review

Phần bổ sung tái hiện U-Net metal_nut được làm ngày 22/11/2026 để khép lại bộ
baseline của ngày 20–21. Triệu sẽ tự trao đổi kiểm tra chéo với Hùng. Lượt này chỉ
commit local, không push, tạo PR, upload Release hoặc đánh dấu Hùng đã nghiệm thu.

## Những gì đã triển khai

| Phần | Kết quả và vai trò | Issue |
| --- | --- | --- |
| PatchCore tile | Fit 184 ảnh good, lưu đủ backbone/memory bank, load offline, raw prediction về size gốc | T03 #3 |
| PatchCore metal_nut | Cùng pipeline, config riêng, fit 176 ảnh good và lưu artifact riêng | Một phần T05 #5 |
| Giao diện chung | Loader PatchCore/U-Net, CLI raw/calibrated, dùng lại run helpers và overlay H03 | T02/T03, tích hợp H03 #11 |
| Calibration | Normal val 46/44 ảnh, quantile 0.95/0.995, chặn manifest khác hash | T03/T05, H03 #11 |
| Đánh giá | Khóa checkpoint/ngưỡng, test chung 67 ảnh tile và 61 ảnh metal_nut, metric/CSV/map/mask/overlay | T03/T05, một phần T06 #6 |
| Tái hiện supervised | Dùng model Release của Hùng, không train lại hoặc đổi ngưỡng để ép khớp | H02 #10, H04 #12 |
| Bằng chứng | 64 tests đạt ở commit tích hợp, lint, notebook, metric versioned, hình và checksum | Bàn giao cho review |

T01/T02 và H01 đã được nghiệm thu trước đó. Kết quả định lượng mới nhất ở
`docs/results/shared-evaluation-day21.md`; tên file giữ theo đợt công việc, phần
metal_nut bổ sung ghi rõ ngày 22. Các ZIP và generated runs ở outputs/artifacts,
không được đưa vào Git.

## Branch và commit đang giữ local

| Branch | Điểm dừng | Nội dung |
| --- | --- | --- |
| `feat/patchcore-tile-baseline` | `ea0b771` | `feat(anomaly): add reproducible PatchCore tile baseline` |
| `feat/anomaly-metal-nut` | `76dd4da` | `feat(anomaly): add PatchCore metal nut baseline` |
| `feat/patchcore-shared-evaluation` | HEAD hiện tại | Chứa cả hai branch trên, merge main H02/H03/H04 và phần đánh giá chung |

Các commit tích hợp đã có:

- `2e87465`: merge main đã có H02/H03/H04 vào branch local tích hợp.
- `6c5adcc`: `feat(inference): integrate PatchCore with shared calibration and evaluation`.
- `8c69eef`: `docs(results): record shared PatchCore evaluation and U-Net reproduction`.
- Commit bổ sung sau cùng ghi việc tái hiện U-Net metal_nut và bàn giao ngày 20–21.
  Dùng `git log --oneline origin/main..feat/patchcore-shared-evaluation` để lấy SHA.

Main local vẫn ở mốc cũ; origin/main đã được fetch tới mốc H04. Không push main
local lên remote. Ba branch feature là lịch sử local, không phải ba phiên bản model
độc lập; branch tích hợp đã chứa toàn bộ thay đổi cần review.

## Phương án PR đề xuất

Đề xuất **một PR tích hợp** từ `feat/patchcore-shared-evaluation` vào `main`.
Reviewer là `phi10hung`, tác giả/assignee là `10tothepowerof6`.

Title: `feat(anomaly): add PatchCore baselines with shared evaluation`.

Phương án này điều chỉnh quy ước mỗi đầu việc một PR trong phân công ban đầu.
Lý do: T03/T05 đã phát triển chồng branch, phần loader/calibration và bằng chứng
đánh giá hiện nằm ở branch tích hợp; một PR cho thấy luồng chạy cuối cùng đã được
kiểm tra. Giữ hai branch nền để xem lịch sử, không cần push cả hai. Không thực hiện
thay đổi quy ước GitHub nào trong lượt này; đây là kế hoạch để Triệu duyệt.

Nếu vẫn muốn mỗi issue một PR, có thể giữ ba PR nối tiếp, nhưng sau mỗi squash
merge phải dựng lại phần commit còn lại trên main mới để tránh diff lặp. Không
chỉ đổi base của các branch cũ sau squash rồi coi chúng là ba PR độc lập.

## Thứ tự push và mở PR sau khi Triệu cho phép

1. Fetch main mới nhất, so diff và xử lý conflict nếu có. Khi phải thay code, chạy
   lại tests và phần runtime chịu ảnh hưởng. Không dùng reset hard hoặc sửa lịch sử
   các branch đã chia sẻ.
2. Kiểm tra staged files. Các sửa nghiệm thu có sẵn ở `docs/day19-handoff.md`, bốn
   sidecar JSON và các file kế hoạch/slide chưa track không tự đưa vào PR này.
   Chúng cần commit riêng nếu Triệu muốn công bố trạng thái nghiệm thu đó.
3. Push đúng một branch: `git push -u origin feat/patchcore-shared-evaluation`.
4. Sau khi được phép, upload các gói calibrated/output và checksum đã chuẩn bị
   local lên GitHub Releases, bổ sung link tải vào chỉ mục bàn giao. PR cần link
   này để Hùng chạy lại được; người dùng tự trao đổi bước kiểm tra chéo với Hùng.
5. Mở PR head `feat/patchcore-shared-evaluation`, base `main`, title như trên,
   request reviewer `phi10hung`. Không dùng `Closes` cho các issue còn thiếu.
6. Hùng review code và bằng chứng; Triệu xác nhận kiểm tra chéo. Chỉ merge sau
   khi các điểm cần sửa đã giải quyết. Có thể squash theo title PR đã duyệt để
   main giữ một thay đổi chức năng rõ ràng.

## Nội dung PR dự thảo

```text
Adds good-only PatchCore baselines for tile and metal_nut with self-contained
artifacts, shared inference/calibration/evaluation, and reproducible custom-split
results. Reproduces the released U-Net baselines on the same test IDs without
retraining or changing their saved thresholds.

Validation: 64 tests passed at the integration commit; shared CLI exercised;
checksums and offline artifact loading verified; test IDs, image decisions and
numerical metric differences checked against the released references.

Includes known failures and the unchanged U-Net metal_nut v1 border-score issue.
PatchCore uses a resource-conscious ResNet-18/1% coreset configuration.

Refs #3, #5, #6, #11, #12
Artifacts and full outputs: <add reviewed Releases links before requesting review>
```

## Hùng cần tập trung review

- Fit chỉ good train; đúng seed/category/fold; test không chọn model hoặc ngưỡng.
- Checkpoint chứa đủ backbone/memory bank; load không cần pretrained cache.
- PatchCore và U-Net dùng chung loader, calibration, metric và cùng test IDs.
- Raw map đúng shape/finite/float32, threshold sau resize; score không min-max từng ảnh.
- Số liệu, false positives/negatives và giới hạn được trình bày đúng, không ép kết
  luận anomaly luôn thắng supervised.

## Còn ngoài phạm vi ngày 20–21

Kiểm tra chéo do hai người tự trao đổi. T03 còn nghiệm thu chéo/bàn giao công khai.
T05 còn EfficientAD, T06 còn các fold khác, T07 chưa làm. Lỗi score viền U-Net
metal_nut giữ ở baseline v1 và sửa bằng PR riêng; H05/demo và chuẩn bị slide là
các đầu việc tiếp theo. Không tự đóng các issue dài hạn từ bản bàn giao một phần.
