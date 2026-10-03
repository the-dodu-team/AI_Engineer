"""Recompute metrics from complete image predictions and compare all 20 models."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

from evaluate_coco_candidates import BY_NAME, GROUPS, write_csv
from evaluate_extended_candidates import CANDIDATES, CORRECTION_CANDIDATES, ROOT, completed_result, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path, expected: dict) -> dict:
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = {}
    for row in rows:
        key = str(Path(row["path"]).resolve()).casefold()
        if key in actual or key not in expected:
            raise ValueError(f"Duplicate/unexpected image in {path}: {key}")
        if row["true_label"] != expected[key]["true_label"]:
            raise ValueError(f"Different ground truth in {path}: {key}")
        if row.get("sha256", expected[key]["sha256"]) != expected[key]["sha256"]:
            raise ValueError(f"Different content hash in {path}: {key}")
        if row["prediction"] not in GROUPS:
            raise ValueError(f"Unknown prediction in {path}: {row['prediction']}")
        actual[key] = row
    if set(actual) != set(expected):
        raise ValueError(f"Incomplete dataset in {path}: {len(actual)}/{len(expected)}")
    return actual


def summarize(rows: dict) -> dict:
    counts = Counter((row["true_label"], row["prediction"]) for row in rows.values())
    positives = sum(row["true_label"] != "other" for row in rows.values())
    negatives = len(rows) - positives
    correct = sum(row["true_label"] == row["prediction"] for row in rows.values())
    fa = counts["other", "computer"] + counts["other", "book"]
    fr = counts["computer", "other"] + counts["book", "other"]
    return {"images": len(rows), "correct": correct, "accuracy": correct/len(rows),
            "false_accepts": fa, "negatives": negatives, "fpr": fa/negatives,
            "false_rejects": fr, "positives": positives, "fnr": fr/positives,
            "book_misses": counts["book", "other"], "computer_misses": counts["computer", "other"],
            "target_class_confusions": counts["book", "computer"]+counts["computer", "book"],
            "meets_both_5pct": fa/negatives <= .05 and fr/positives <= .05}


def paired_counts(rows: dict, reference: dict) -> dict:
    improved = worsened = 0
    for key, row in rows.items():
        correct = row["true_label"] == row["prediction"]
        old_correct = reference[key]["true_label"] == reference[key]["prediction"]
        improved += correct and not old_correct
        worsened += old_correct and not correct
    n = improved + worsened
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, worsened)+1)) / 2**n) if n else 1.0
    return {"fixed_vs_rtmdet_x": improved, "regressed_vs_rtmdet_x": worsened,
            "mcnemar_exact_p_unadjusted": p}


def plot_comparison(models: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"RTMDet reference": "#697586", "previous YOLO": "#5685b5",
              "corrected RT-DETR": "#8d6ab3", "new 5": "#dc873d"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
    names = [row["model"] for row in models]
    for axis, metric, title in zip(axes, ("accuracy", "fpr", "fnr"),
                                   ("Top-1 accuracy (%)", "False acceptance (%)", "False rejection (%)")):
        values = [row[metric]*100 for row in models]
        bars = axis.barh(names, values, color=[colors[row["cohort"]] for row in models])
        axis.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_xlim(0, 103 if metric == "accuracy" else max(6.5, max(values)*1.25))
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
        if metric != "accuracy":
            axis.axvline(5, color="#b42318", linestyle="--", linewidth=1)
    axes[0].invert_yaxis()
    figure.suptitle("372 identical images | gray: RTMDet, blue: YOLO, purple: corrected RT-DETR, orange: new 5\nDashed line: 5% error target", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", type=Path, default=ROOT / "outputs/coco_extended_detectors")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/coco_small_detectors")
    parser.add_argument("--corrected", type=Path, default=ROOT / "outputs/coco_corrected_rtdetr")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/rtmdet/model-size-20260927")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/coco-extended-detectors")
    args = parser.parse_args()
    manifest = read_json(args.extended / "run.json")
    expected = {str(Path(r["path"]).resolve()).casefold(): r for r in manifest["images"]}
    baseline_manifest = read_json(args.baseline / "run.json")
    from evaluate_extended_candidates import validate_manifest
    validate_manifest(manifest["images"], baseline_manifest)
    corrected_manifest = read_json(args.corrected / "run.json")
    validate_manifest(corrected_manifest["images"], baseline_manifest)
    if corrected_manifest["protocol"] != manifest["protocol"]:
        raise ValueError("Corrected RT-DETR run must use the same evaluation protocol")
    if manifest["protocol"]["threshold"] != .05:
        raise ValueError("Historical comparison requires the original threshold 0.05")
    models, all_rows, corrections = [], {}, []
    for name in baseline_manifest["selected_models"]:
        rows = read_predictions(args.baseline / name / "predictions.csv", expected)
        saved = read_json(args.baseline / name / "metrics.json")
        if saved["images"] != len(rows):
            raise ValueError(f"Invalid baseline count: {name}")
        models.append({"model": name, "cohort": "RTMDet reference", **summarize(rows),
                       "latency_ms": saved["runtime"]["latency_mean_ms"],
                       "params_m": "", "code_license": "Apache-2.0",
                       "commercial_status": "permissive_with_notices"})
        all_rows[name] = rows
    for name in BY_NAME:
        previous_rows = read_predictions(args.previous / name / "predictions.csv", expected)
        rows = previous_rows
        cohort = "previous YOLO"
        if name in CORRECTION_CANDIDATES:
            saved = completed_result(args.corrected, name, corrected_manifest)
            if saved is None:
                raise ValueError(f"Missing corrected RT-DETR result: {name}")
            rows = read_predictions(args.corrected / name / "predictions.csv", expected)
            cohort = "corrected RT-DETR"
            before, after = summarize(previous_rows), summarize(rows)
            corrections.append({"model": name, "old_correct": before["correct"], "corrected_correct": after["correct"],
                                "old_fpr": before["fpr"], "corrected_fpr": after["fpr"],
                                "old_fnr": before["fnr"], "corrected_fnr": after["fnr"]})
        models.append({"model": name, "cohort": cohort, **summarize(rows),
                       "latency_ms": statistics.mean(float(r["latency_ms"]) for r in rows.values()),
                       "params_m": BY_NAME[name].params_m,
                       "code_license": "AGPL-3.0" if name.startswith("yolo") else "Apache-2.0",
                       "commercial_status": "copyleft_or_separate_permission" if name.startswith("yolo") else "permissive_with_notices"})
        all_rows[name] = rows
    for name in CANDIDATES:
        saved = completed_result(args.extended, name, manifest)
        if saved is None:
            raise ValueError(f"Missing completed model: {name}")
        rows = read_predictions(args.extended / name / "predictions.csv", expected)
        models.append({"model": name, "cohort": "new 5", **summarize(rows),
                       "latency_ms": saved["mean_latency_ms"], "params_m": saved["actual_params_m"],
                       "code_license": saved["code_license"], "commercial_status": saved["commercial_status"]})
        all_rows[name] = rows
    reference = all_rows["rtmdet-x"]
    for row in models:
        row.update(paired_counts(all_rows[row["model"]], reference))
    models.sort(key=lambda r: (-r["accuracy"], r["fpr"], r["fnr"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "comparison.csv", models)
    write_csv(args.output / "corrections.csv", corrections)
    plot_comparison(models, args.output)
    write_json(args.output / "comparison.json", {"manifest_sha256": manifest["manifest_sha256"],
               "protocol": manifest["protocol"], "models": models,
               "licenses": {**CANDIDATES, **CORRECTION_CANDIDATES}, "corrections": corrections,
               "note": "Historical latency is not a controlled speed comparison. Paired p-values are exploratory and unadjusted for model selection/multiple comparisons."})
    per_source, disagreements = [], []
    for name, predictions in all_rows.items():
        sources = sorted({r["source_label"] for r in expected.values()})
        for source in sources:
            subset = [row for key, row in predictions.items() if expected[key]["source_label"] == source]
            per_source.append({"model": name, "source": source, "images": len(subset),
                               "correct": sum(row["true_label"] == row["prediction"] for row in subset),
                               **{group: sum(r["prediction"] == group for r in subset) for group in GROUPS}})
        if name in CANDIDATES:
            for key, row in predictions.items():
                if row["prediction"] != reference[key]["prediction"]:
                    disagreements.append({"model": name, "sha256": expected[key]["sha256"],
                                          "source": expected[key]["source_label"], "file": Path(row["path"]).name,
                                          "truth": row["true_label"], "rtmdet_x": reference[key]["prediction"],
                                          "candidate": row["prediction"]})
    write_csv(args.output / "by_source.csv", per_source)
    if disagreements:
        write_csv(args.output / "disagreements.csv", disagreements)
    lines = ["# 추가 사전학습 탐지 모델 5종 비교", "", "추가 5종 + 기존 후보 10종 + RTMDet 기준 5종", "",
             "## 평가 조건", "",
             "- 드라이브 sample의 고유 이미지 372장: computer 200, book 71, other 101. RTMDet 실행의 SHA-256·정답 집합과 일치함을 검증했습니다.",
             "- 원본 451개에서 완전 중복 79개를 제외했습니다. 추가 학습이나 평가 데이터에 맞춘 임계값 조정은 하지 않았습니다.",
             "- COCO laptop·tv(tvmonitor 포함) → computer, book → book, 나머지 → other. 그룹별 최고 점수, Top-1, 수락 임계값 0.05, 탐지 점수 하한 0.01입니다.",
             f"- 추가 모델은 CPU {manifest['protocol']['threads']} 스레드, 배치 1, 워밍업 {manifest['protocol']['warmup']}회, 이미지당 {manifest['protocol']['repeats']}회 측정했습니다.",
             "- 공개 가중치의 전처리를 유지했습니다: RF-DETR Large 704, Medium 576, 나머지 640. 런타임·입력 크기·반복 횟수가 다른 과거 지연 수치는 참고값입니다.",
             "- D-FINE/LW-DETR은 Transformers로 변환된 체크포인트를 사용했습니다. 저장소 리비전과 실제 가중치 해시는 실행 결과 metrics.json에 기록했습니다.", "",
             "## 기존 RT-DETR 결과 정정", "",
             "기존 평가 스크립트는 PekingU 체크포인트의 COCO TV 클래스 이름 `tvmonitor`를 `other`로 잘못 매핑했습니다. `computer`로 수정한 뒤 RT-DETR 4종을 같은 372장에 재실행했습니다. 아래 전체 결과에는 수정 결과를 사용했습니다. 수정 전 원본 파일은 보존했습니다.", "",
             "| 모델 | 수정 전 정답 | 수정 후 정답 | 수정 전 FPR | 수정 후 FPR | 수정 전 FNR | 수정 후 FNR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in corrections:
        lines.append(f"| {row['model']} | {row['old_correct']} | {row['corrected_correct']} | {row['old_fpr']:.2%} | {row['corrected_fpr']:.2%} | {row['old_fnr']:.2%} | {row['corrected_fnr']:.2%} |")
    lines += ["",
             "## 전체 결과", "",
             "정확도는 3개 라벨의 일치율입니다. FPR은 other를 대상으로 수락한 비율(분모 101), FNR은 computer·book을 other로 거절한 비율(분모 271)입니다. 대상 클래스 사이의 오분류는 정확도에 별도로 반영됩니다.", "",
             "| 모델 | 구분 | 정답/372 | 정확도 | FPR | FNR | 지연 ms¹ |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in models:
        lines.append(f"| {row['model']} | {row['cohort']} | {row['correct']} | {row['accuracy']:.2%} | {row['fpr']:.2%} ({row['false_accepts']}/101) | {row['fnr']:.2%} ({row['false_rejects']}/271) | {row['latency_ms']:.1f} |")
    lines += ["", "¹ 디스크 읽기·모델 로딩 제외. 과거 실행과의 속도 배수는 계산하지 않았습니다.", "", "![20개 모델의 정확도와 오류율](comparison.png)", "", "## 추가 모델과 RTMDet-x의 이미지별 대조", "",
              "| 추가 모델 | x의 오답을 수정 | x의 정답을 오답으로 변경 | book → other | computer → other | FPR·FNR 모두 ≤5% |", "|---|---:|---:|---:|---:|---|"]
    for row in models:
        if row["model"] in CANDIDATES:
            lines.append(f"| {row['model']} | {row['fixed_vs_rtmdet_x']} | {row['regressed_vs_rtmdet_x']} | {row['book_misses']} | {row['computer_misses']} | {'충족' if row['meets_both_5pct'] else '미충족'} |")
    lines += ["", "## 상업적 이용 조건", "",
              "아래는 확인한 공개 조건을 바탕으로 한 배포 검토 분류입니다. 가중치의 학습 데이터 조건과 코드 라이선스는 별도로 확인해야 합니다.", "",
              "| 모델 | 코드·모델 카드 | 상업 서비스 판단 |", "|---|---|---|",
              "| RF-DETR Large·Medium | Apache-2.0 | 공개 조건상 상업 이용 가능. 라이선스·저작권 고지 및 해당 NOTICE 유지. 사용한 두 모델은 Plus/PML 모델이 아님. |",
              "| D-FINE-L Objects365→COCO | Apache-2.0 + 가중치 조건 검토 | 제작사가 Objects365 가중치의 상업적 사용을 자동으로 허용된 것으로 보지 말라고 명시. 권리 확인 전 배포 후보 보류. |",
              "| LW-DETR Large | Apache-2.0 + 학습 데이터 조건 검토 | Objects365 사전학습을 사용하며 데이터 사이트는 학술 목적을 명시. 모델 가중치에 대한 상업 이용 허가 범위 확인 필요. |",
              "| YOLOv13-L | AGPL-3.0 | 상업 이용 자체는 가능하지만 배포·수정된 네트워크 서비스의 소스 제공 의무 등을 준수해야 함. 비공개 제품은 관련 권리자의 별도 허가 범위 확인 필요. |",
              "| RTMDet·RT-DETR 기존 기준 모델 | Apache-2.0 프로젝트 | 상업 이용에 유리한 허용적 코드 라이선스. 배포 시 사용 가중치와 고지 조건 확인. |",
              "| 기존 Ultralytics YOLO 후보 | AGPL-3.0 / 별도 Enterprise 조건 | 비공개 상업 제품은 별도 Enterprise 허가 범위 확인. 이 허가가 YOLOv13 외부 포크까지 포함한다고 가정하지 않음. |", "",
              "출처: [RF-DETR 모델별 라이선스](https://github.com/roboflow/rf-detr), [D-FINE 가중치 주의사항](https://github.com/Peterande/D-FINE#model-zoo), [LW-DETR 학습 경로](https://github.com/Atten4Vis/LW-DETR), [Objects365 이용 조건](https://www.objects365.org/download.html), [YOLOv13 LICENSE](https://github.com/iMoonLab/yolov13/blob/main/LICENSE), [MMDetection LICENSE](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE), [RT-DETR LICENSE](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE), [Ultralytics 라이선스](https://www.ultralytics.com/license).", "",
               "## 결론", "",
               "- 현재 조건에서 상업 배포 후보 1순위는 `rtdetrv2-r50`입니다. RTMDet-x보다 6장을 더 맞혔고(356 대 350), FNR은 7.75%에서 5.54%로 낮았으며 FPR은 같은 0.99%였습니다. 43.0M 파라미터로 RTMDet-x보다 작고 Apache-2.0 계열입니다.",
               "- 다만 목표 FNR 5%를 충족하려면 오탐 거절이 최대 13장이어야 하는데 15장으로 2장 초과했습니다. RTMDet-x와의 이미지별 정확도 차이도 exact McNemar p=0.238로, 이 372장만으로 우위를 확정할 수 없습니다.",
               "- 새 5개 중 정확도 1위인 D-FINE-L은 352장으로 RTMDet-x보다 2장 많았지만, Objects365 사전학습 가중치의 상업 이용 권리가 불명확해 제품 후보로 바로 채택하기 어렵습니다.",
               "- RF-DETR Large는 RTMDet-x와 동률이고 Medium은 1장 낮았습니다. YOLOv13-L은 동률이지만 AGPL-3.0 조건이 있으며, LW-DETR Large는 2장 낮고 Objects365 조건 확인이 필요합니다.", "",
               "## 해석 범위", "",
              "이 결과는 동일한 372장의 폴더 라벨 평가입니다. 모델 선택에 반복 사용한 데이터이므로 새로운 이미지에서의 우월성을 입증하지는 않습니다. 여러 대상이 함께 보이는 사진과 실제 서비스의 세션 판정은 별도 검증이 필요합니다.", "",
              "- [전체 집계 CSV](comparison.csv)", "- [기존 RT-DETR 정정 내역](corrections.csv)", "- [폴더별 결과](by_source.csv)", "- [RTMDet-x와 다른 예측 목록](disagreements.csv)",
              "- [설정·라이선스·집계 JSON](comparison.json)", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for row in models:
        print(f"{row['model']:20} correct={row['correct']}/372 FPR={row['fpr']:.2%} FNR={row['fnr']:.2%} {row['commercial_status']}")
    print(args.output / "README.md")


if __name__ == "__main__":
    main()
