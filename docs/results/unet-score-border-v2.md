# U-Net v2: điểm ảnh bỏ viền (ngày 23/11/2026)

Sửa vấn đề đã ghi ở PR #19 và Release `unet-metal-nut-holdout-scratch-v1`: điểm ảnh U-Net
v1 là max của toàn bộ map, nhưng trên ảnh good max luôn rơi vào một pixel ở góc ảnh do
artifact padding của tích chập. v2 giữ nguyên checkpoint, preprocessing, map và mask;
**chỉ đổi điểm ảnh** thành max sau khi bỏ một dải viền `b` pixel ở kích thước ảnh gốc.

> **Sửa sau khi đã xem test.** Vấn đề được phát hiện khi phân tích kết quả test v1 của
> metal_nut. Độ rộng viền được chọn chỉ trên validation theo quy tắc ghi trước, nhưng v2
> vẫn là kết quả thăm dò sau test. Bảng v1 trong
> [shared-evaluation-day21.md](shared-evaluation-day21.md) giữ nguyên làm kết quả đầu.

## Chọn độ rộng viền trên validation

Quy tắc ghi trước khi chạy (cũng lưu trong [selection JSON](unet-score-border-selection.json)):
với mỗi `b` trong {0, 4, 8, 16, 32}, lấy quantile 0.95 điểm ảnh good của val (q95).
`b` hợp lệ khi q95 lệch không quá 10% so với `b` rộng nhất (viền không còn chi phối điểm)
và AUROC good/lỗi đã biết của val không thấp hơn `b = 0`. Chọn `b` nhỏ nhất hợp lệ ở cả
hai category, để mọi artifact U-Net dùng cùng một quy tắc.

| b | tile q95 good | tile AUROC | metal_nut q95 good | metal_nut AUROC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1849 | 1.0000 | 0.2560 | 1.0000 |
| 4 | 0.0728 | 1.0000 | 0.0641 | 1.0000 |
| 8 | 0.0327 | 1.0000 | 0.0641 | 1.0000 |
| 16 | 0.0302 | 1.0000 | 0.0641 | 1.0000 |
| 32 | 0.0236 | 1.0000 | 0.0641 | 1.0000 |

metal_nut đạt plateau ngay từ 4 px; tile giảm dần, chỉ `b = 32` thỏa ngưỡng 10%, nên
quy tắc chọn **b = 32**. Đây là giá trị lớn nhất đã thử. Kiểm tra rủi ro trên train/val:
không ảnh lỗi nào có mask nằm trọn trong dải 32 px (tile 0/50, metal_nut 0/54); 25 ảnh
tile và 3 ảnh metal_nut có mask chạm viền nhưng kéo dài vào trong. Điểm thấp nhất của ảnh
lỗi val không đổi giữa `b = 0` và `b = 32`.

## Kết quả test (custom_comparison, seed 42)

Cùng 67 test IDs tile và 61 test IDs metal_nut như bảng chung. Ngưỡng calibrate lại từ
normal val (quantile 0.95/0.995 theo config artifact). Ngưỡng pixel không đổi vì map không đổi.

| Model/category | Ngưỡng ảnh | Image AUROC | Image F1 | Good báo nhầm | Recall known | Recall held-out |
| --- | ---: | ---: | ---: | --- | --- | --- |
| U-Net tile v1 | 0.184943 | 1.0000 | 1.0000 | 0/33 | 17/17 | crack 17/17 |
| U-Net tile v2 (b=32) | 0.023596 | 1.0000 | 0.9855 | 1/33 | 17/17 | crack 17/17 |
| U-Net metal_nut v1 | 0.255985 | 0.8800 | 0.8000 | 12/22 | 16/16 | scratch 18/23 |
| U-Net metal_nut v2 (b=32) | 0.064074 | 0.9452 | 0.9333 | 1/22 | 16/16 | scratch 19/23 |

Pixel AUROC/AP và Dice giữ nguyên giữa v1 và v2 (tile 0.9704 / 0.9194 / 0.6282;
metal_nut 0.9307 / 0.8590 / 0.5778), đúng với thay đổi chỉ ở điểm ảnh.

## Đọc kết quả

- **metal_nut:** 11 ảnh good bị báo nhầm vì pixel góc (điểm v1 khoảng 0.257) về mức nền
  0.04–0.06 và PASS. Bắt thêm `scratch/001.png`.
- **`metal_nut/test/good/016.png` vẫn báo nhầm** với điểm 0.3111 giống hệt v1: phản hồi nằm
  trong ảnh, không phải viền. PatchCore cũng báo nhầm ảnh này trong bảng chung.
- **Bốn scratch vẫn bị sót** (`000`, `019`, `020`, `022`), điểm 0.042–0.058: U-Net không
  phản hồi với các vết xước này. Đây là giới hạn với lỗi chưa thấy, không phải lỗi gộp điểm.
- **tile:** điểm good sau khi bỏ viền rất thấp và sát nhau (0.014–0.033), ngưỡng quantile
  0.95 trên 46 ảnh val trở nên nhạy; `tile/test/good/019.png` (0.0331) vượt ngưỡng 0.0236.
  AUROC vẫn 1.0. Ghi nhận như một đánh đổi của v2, không chỉnh ngưỡng theo test.
- So với PatchCore metal_nut (2/22 good báo nhầm, scratch 21/23), U-Net v2 thu hẹp khoảng
  cách cấp ảnh nhưng vẫn bắt ít scratch hơn. Chỉ một fold mỗi category; không khái quát.

## Tái hiện

```powershell
$T1 = "artifacts\custom_comparison\unet_resnet18\tile\20261121-150943-seed42-holdout-crack"
$M1 = "artifacts\custom_comparison\unet_resnet18\metal_nut\20261121-162901-seed42-holdout-scratch"
python scripts/select_score_border.py --dataset-root datasets --artifact-dir $T1 --manifest splits/seed42/custom_comparison_tile_crack.csv --artifact-dir $M1 --manifest splits/seed42/custom_comparison_metal_nut_scratch.csv --output outputs\score-border\selection.json
python scripts/derive_unet_score_variant.py --artifact-dir $T1 --border 32 --selection outputs\score-border\selection.json
python scripts/derive_unet_score_variant.py --artifact-dir $M1 --border 32 --selection outputs\score-border\selection.json
python scripts/calibrate.py --artifact-dir "$T1-border32" --dataset-root datasets --manifest splits/seed42/custom_comparison_tile_crack.csv
python scripts/calibrate.py --artifact-dir "$M1-border32" --dataset-root datasets --manifest splits/seed42/custom_comparison_metal_nut_scratch.csv
python scripts/evaluate.py --artifact-dir "$T1-border32" --dataset-root datasets --manifest splits/seed42/custom_comparison_tile_crack.csv
python scripts/evaluate.py --artifact-dir "$M1-border32" --dataset-root datasets --manifest splits/seed42/custom_comparison_metal_nut_scratch.csv
```

Artifact v1 lấy từ hai Release U-Net v1. Artifact v2 có trong Release v2 tương ứng, kèm
checksum. Các lần chạy v2 được thực hiện trên commit `6f615e9` với thay đổi chưa commit
(metadata ghi `dirty=true`); code đó chính là nội dung PR này.

Metric đầy đủ: [tile v2](unet_resnet18-tile-custom-metrics-v2.json),
[metal_nut v2](unet_resnet18-metal_nut-custom-metrics-v2.json).
