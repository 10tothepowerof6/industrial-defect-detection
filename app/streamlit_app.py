"""Image inspection demo for PatchCore and U-Net on MVTec tile / metal_nut.

Run from the repository root:
    python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_defect_detection.demo import (  # noqa: E402
    ArtifactInfo,
    default_comparison,
    find_artifacts,
    ground_truth_mask,
    inspect,
    read_image,
    result_panels,
    sample_images,
)
from industrial_defect_detection.inference.loader import load_predictor  # noqa: E402
from industrial_defect_detection.models.supervised.scoring import ImageTooSmallError  # noqa: E402

ARTIFACT_ROOT = ROOT / "artifacts"
DATASET_ROOT = ROOT / "datasets"

TEXT = {
    "vi": {
        "title": "Kiểm tra lỗi sản phẩm công nghiệp",
        "subtitle": "MVTec AD · tile và metal_nut · PatchCore (anomaly) và U-Net (supervised)",
        "language": "Ngôn ngữ",
        "category": "Sản phẩm",
        "mode": "Chế độ",
        "mode_single": "Một mô hình",
        "mode_compare": "So sánh U-Net và PatchCore",
        "model": "Mô hình",
        "source": "Nguồn ảnh",
        "source_sample": "Ảnh mẫu trong dataset",
        "source_upload": "Tải ảnh lên",
        "defect_type": "Loại ảnh",
        "file": "Ảnh",
        "upload": "Chọn file ảnh (PNG/JPG)",
        "run": "Kiểm tra",
        "pass": "ĐẠT (PASS)",
        "defect": "CÓ LỖI (DEFECT)",
        "image_score": "Điểm ảnh",
        "image_threshold": "Ngưỡng ảnh",
        "pixel_threshold": "Ngưỡng pixel",
        "defect_pixels": "Pixel bị tô",
        "latency": "Thời gian",
        "panel_image": "Ảnh gốc",
        "panel_ground_truth": "Lỗi thật (xanh lá)",
        "panel_heatmap": "Heatmap điểm",
        "panel_mask": "Vùng mô hình tô (đỏ)",
        "explain": (
            "PASS/DEFECT dựa trên **ngưỡng ảnh**: DEFECT khi điểm ảnh > ngưỡng. Mask là các pixel "
            "vượt **ngưỡng pixel** nên có thể có vài điểm tô ngay cả khi ảnh PASS. Mỗi mô hình có "
            "thang điểm riêng, điểm không phải phần trăm tin cậy. Heatmap: giữa thang màu là ngưỡng "
            "pixel, màu chỉ để hiển thị."
        ),
        "details": "Chi tiết artifact",
        "no_artifacts": (
            "Chưa có mô hình đã calibrate cho sản phẩm này. Tải release và giải nén vào "
            "`artifacts/custom_comparison/<model>/<sản phẩm>/<run_id>/`, hoặc chạy "
            "`scripts/calibrate.py`."
        ),
        "uncalibrated": "Bỏ qua artifact chưa calibrate",
        "missing_compare": "Chọn ít nhất 2 mô hình để so sánh.",
        "no_dataset": "Không tìm thấy `datasets/{category}/test`; hãy tải ảnh lên.",
        "need_image": "Chọn hoặc tải một ảnh rồi bấm **Kiểm tra**.",
        "bad_image": "Không đọc được ảnh",
        "load_error": "Không load được mô hình",
        "predict_error": "Mô hình không xử lý được ảnh này",
        "image_too_small": (
            "Ảnh {width}×{height} px quá nhỏ cho mô hình này: điểm ảnh bỏ viền {border} px nên "
            "mỗi cạnh phải từ {minimum} px trở lên. Mô hình không tự đổi cách tính vì ngưỡng đã "
            "calibrate cho viền {border} px."
        ),
        "unet_metal_nut_border": (
            "U-Net metal_nut v1: điểm ảnh bị một pixel ở góc ảnh chi phối nên có thể báo nhầm "
            "ảnh tốt (12/22 trên test). Xem PR #19."
        ),
        "device": "Thiết bị",
        "held_out": "lỗi giữ riêng, chưa thấy khi train",
        "compare_models": "Mô hình so sánh (chọn từ 2)",
        "score_whole": "max toàn ảnh",
        "score_border": "bỏ viền {px} px",
    },
    "en": {
        "title": "Industrial product defect inspection",
        "subtitle": "MVTec AD · tile and metal_nut · PatchCore (anomaly) and U-Net (supervised)",
        "language": "Language",
        "category": "Product",
        "mode": "Mode",
        "mode_single": "Single model",
        "mode_compare": "Compare U-Net and PatchCore",
        "model": "Model",
        "source": "Image source",
        "source_sample": "Dataset sample",
        "source_upload": "Upload",
        "defect_type": "Image type",
        "file": "Image",
        "upload": "Choose an image file (PNG/JPG)",
        "run": "Inspect",
        "pass": "PASS",
        "defect": "DEFECT",
        "image_score": "Image score",
        "image_threshold": "Image threshold",
        "pixel_threshold": "Pixel threshold",
        "defect_pixels": "Flagged pixels",
        "latency": "Time",
        "panel_image": "Original",
        "panel_ground_truth": "Ground truth (green)",
        "panel_heatmap": "Score heatmap",
        "panel_mask": "Predicted mask (red)",
        "explain": (
            "PASS/DEFECT uses the **image threshold**: DEFECT when image score > threshold. The mask "
            "shows pixels above the **pixel threshold**, so a PASS image can still have a few "
            "flagged pixels. Each model has its own score scale; scores are not confidence "
            "percentages. Heatmap: mid-scale is the pixel threshold; colors are display only."
        ),
        "details": "Artifact details",
        "no_artifacts": (
            "No calibrated model for this product. Download a release and extract it into "
            "`artifacts/custom_comparison/<model>/<product>/<run_id>/`, or run "
            "`scripts/calibrate.py`."
        ),
        "uncalibrated": "Skipped uncalibrated artifacts",
        "missing_compare": "Pick at least 2 models to compare.",
        "no_dataset": "`datasets/{category}/test` not found; please upload an image.",
        "need_image": "Pick or upload an image, then press **Inspect**.",
        "bad_image": "Cannot read the image",
        "load_error": "Cannot load the model",
        "predict_error": "The model could not process this image",
        "image_too_small": (
            "Image {width}×{height} px is too small for this model: its image score ignores a "
            "{border}-pixel border, so each side must be at least {minimum} px. The score rule is "
            "not changed automatically because the threshold was calibrated for a {border}-pixel "
            "border."
        ),
        "unet_metal_nut_border": (
            "U-Net metal_nut v1: the image score is dominated by one corner pixel, so good parts "
            "may be flagged (12/22 on test). See PR #19."
        ),
        "device": "Device",
        "held_out": "held out, unseen in training",
        "compare_models": "Models to compare (pick 2 or more)",
        "score_whole": "whole-map max",
        "score_border": "{px} px border ignored",
    },
}

# Plain-language names shown in brackets after the dataset names.
NAMES = {
    "vi": {
        "tile": "gạch ốp",
        "metal_nut": "đai ốc kim loại",
        "good": "ảnh tốt",
        "crack": "vết nứt",
        "glue_strip": "vệt keo",
        "gray_stroke": "vệt xám",
        "oil": "vết dầu",
        "rough": "bề mặt sần",
        "bent": "cong vênh",
        "color": "sai màu",
        "flip": "lật ngược",
        "scratch": "vết xước",
    },
    "en": {
        "tile": "ceramic tile",
        "metal_nut": "metal nut",
        "good": "defect-free",
        "crack": "crack",
        "glue_strip": "glue strip",
        "gray_stroke": "gray stroke",
        "oil": "oil stain",
        "rough": "rough surface",
        "bent": "bent",
        "color": "discoloration",
        "flip": "flipped upside down",
        "scratch": "scratch",
    },
}


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def cached_predictor(artifact_dir: str, device: str):
    """One loaded model per artifact path and device, so switching products cannot reuse it.

    One warm-up prediction runs here, so the time shown per image excludes CUDA start-up.
    """
    predictor = load_predictor(Path(artifact_dir), device)
    predictor.predict(np.zeros((256, 256, 3), dtype=np.uint8))
    return predictor


def artifact_label(artifact: ArtifactInfo, t: dict, short: bool = False) -> str:
    """e.g. 'U-Net (ResNet-18) v2 · bỏ viền 8 px · holdout-crack · <run_id>'."""
    parts = [artifact.name + (f" {artifact.score_version}" if artifact.score_version else "")]
    if artifact.score_version:
        parts.append(
            t["score_border"].format(px=artifact.score_border)
            if artifact.score_border
            else t["score_whole"]
        )
    if not short:
        parts += [artifact.fold, artifact.run_id]
    return " · ".join(parts)


def show_result(info: ArtifactInfo, image, gt_mask, t: dict, device: str) -> None:
    try:
        predictor = cached_predictor(str(info.path), device)
    except Exception as error:  # noqa: BLE001 - show any loading problem to the user
        st.error(f"{t['load_error']}: {info.label} — {error}")
        return
    st.subheader(artifact_label(info, t, short=True))
    try:
        prediction, elapsed_ms = inspect(predictor, image)
    except ImageTooSmallError as error:
        st.error(
            t["image_too_small"].format(
                width=error.width,
                height=error.height,
                border=error.border,
                minimum=2 * error.border + 1,
            )
        )
        return
    except Exception as error:  # noqa: BLE001 - one failing model must not stop the page
        st.error(f"{t['predict_error']}: {error}")
        return
    if info.known_issue:
        st.warning(t[info.known_issue])
    if prediction.is_defective:
        st.error(f"### {t['defect']}")
    else:
        st.success(f"### {t['pass']}")
    columns = st.columns(5)
    columns[0].metric(t["image_score"], f"{prediction.image_score:.4f}")
    columns[1].metric(t["image_threshold"], f"{prediction.image_threshold:.4f}")
    columns[2].metric(t["pixel_threshold"], f"{prediction.pixel_threshold:.4f}")
    columns[3].metric(t["defect_pixels"], f"{int(prediction.pred_mask.sum()):,}")
    columns[4].metric(t["latency"], f"{elapsed_ms:.0f} ms")
    panels = result_panels(image, prediction, gt_mask)
    image_columns = st.columns(len(panels))
    for column, (name, picture) in zip(image_columns, panels.items(), strict=True):
        column.image(picture, caption=t[f"panel_{name}"], width="stretch")
    with st.expander(t["details"]):
        meta = predictor.metadata
        st.json(
            {
                "artifact_dir": str(info.path.relative_to(ROOT))
                if info.path.is_relative_to(ROOT)
                else str(info.path),
                "model_id": meta.get("model_id"),
                "category": meta.get("category"),
                "protocol": meta.get("protocol"),
                "fold": meta.get("fold"),
                "run_id": meta.get("run_id"),
                "image_score": meta.get("score", {}).get("image_score"),
                "derived_from": meta.get("derived_from", {}).get("run_id"),
                "device": device,
                "calibration": {
                    k: meta["calibration"].get(k)
                    for k in ("image_threshold", "pixel_threshold", "normal_val_images", "method")
                },
                "training_manifest": meta.get("training_manifest", {}).get("path"),
            }
        )


def main() -> None:
    st.set_page_config(page_title="Defect inspection", layout="wide")
    language = st.sidebar.radio(
        "Ngôn ngữ / Language",
        ["vi", "en"],
        format_func=lambda c: {"vi": "Tiếng Việt", "en": "English"}[c],
        horizontal=True,
    )
    t = TEXT[language]
    st.title(t["title"])
    st.caption(t["subtitle"])

    category = st.sidebar.radio(
        t["category"],
        ["tile", "metal_nut"],
        format_func=lambda c: f"{c} ({NAMES[language][c]})",
    )
    all_artifacts = [a for a in find_artifacts(ARTIFACT_ROOT) if a.category == category]
    artifacts = [a for a in all_artifacts if a.calibrated]
    skipped = len(all_artifacts) - len(artifacts)
    if skipped:
        st.sidebar.caption(f"{t['uncalibrated']}: {skipped}")
    if not artifacts:
        st.error(t["no_artifacts"])
        return

    def label(artifact: ArtifactInfo) -> str:
        return artifact_label(artifact, t)

    mode = st.sidebar.radio(t["mode"], ["single", "compare"], format_func=lambda m: t[f"mode_{m}"])
    if mode == "single":
        default = default_comparison(artifacts)[0]
        chosen = [
            st.sidebar.selectbox(
                t["model"], artifacts, index=artifacts.index(default), format_func=label
            )
        ]
    else:
        # Default: latest U-Net (v2 if available) against PatchCore; any 2+ may be picked.
        chosen = st.sidebar.multiselect(
            t["compare_models"],
            artifacts,
            default=default_comparison(artifacts),
            format_func=label,
            key=f"compare-{category}",
        )
        if len(chosen) < 2:
            st.warning(t["missing_compare"])
            return

    # Defect types the chosen model(s) never saw in training, from the fold "holdout-<defect>".
    held_out = {a.fold.removeprefix("holdout-") for a in chosen if a.fold.startswith("holdout-")}

    def defect_label(defect_type: str) -> str:
        label = f"{defect_type} ({NAMES[language].get(defect_type, defect_type)})"
        return f"{label} · {t['held_out']}" if defect_type in held_out else label

    source = st.sidebar.radio(
        t["source"], ["sample", "upload"], format_func=lambda s: t[f"source_{s}"]
    )
    image, gt_mask, image_name = None, None, ""
    if source == "sample":
        groups = sample_images(DATASET_ROOT, category)
        if not groups:
            st.warning(t["no_dataset"].format(category=category))
        else:
            defect_type = st.sidebar.selectbox(
                t["defect_type"], list(groups), format_func=defect_label
            )
            path = st.sidebar.selectbox(
                t["file"], groups[defect_type], format_func=lambda p: p.name
            )
            image_name = (
                f"{category} ({NAMES[language][category]}) / {defect_label(defect_type)} / "
                f"{path.name}"
            )
            try:
                image, gt_mask = read_image(path), ground_truth_mask(path)
            except ValueError as error:
                st.error(f"{t['bad_image']}: {error}")
    else:
        upload = st.sidebar.file_uploader(t["upload"], type=["png", "jpg", "jpeg", "bmp"])
        if upload is not None:
            image_name = upload.name
            try:
                image = read_image(upload)
            except ValueError as error:
                st.error(f"{t['bad_image']}: {error}")

    device = device_name()
    st.sidebar.caption(f"{t['device']}: {device}")
    if image is None or not st.sidebar.button(t["run"], type="primary", width="stretch"):
        st.info(t["need_image"])
        if image is not None:
            st.image(image, caption=image_name, width=320)
        return

    st.caption(f"{image_name} · {image.shape[1]}×{image.shape[0]}")
    for artifact in chosen:
        show_result(artifact, image, gt_mask, t, device)
    st.info(t["explain"])


if __name__ == "__main__":
    main()
