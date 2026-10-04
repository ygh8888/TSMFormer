# Revision Log — Sensors-111383-2026

IEEE Sensors Journal 재투고 대응 작업 기록.
리뷰어 항목 번호(R1-1~R1-13, R2-1~R2-8)로 대응 관계를 표기한다.

- 원고: Temporal-Shift Transformer Backbones for Multi-Sensor Dynamic Hand Gesture Recognition
- 결정일: 2026-09-25 / 판정: Reject (revise and resubmit) = 실질 Major Revision
- 기준 커밋(논문 버전): `aaed89a1bec04e8906e2a9d31939454dc2b9ef0a`

---

## 0. 환경 재구축 (2026-09-29 ~ 09-30)

기존 대여 서버 만료로 신규 서버에 환경 재구축. 하드웨어·OS는 논문 당시와 동일.

| 항목 | 논문 당시 | 신규 서버 | 일치 |
|---|---|---|---|
| GPU | A100-SXM4-80GB MIG 1g.10gb | 동일 | O |
| Driver / CUDA | 535.261.03 / 12.2 | 동일 | O |
| OS | Ubuntu 20.04.3 LTS | 동일 | O |
| Python | 3.10.20 | 3.10.x (conda) | O |

확정된 패키지 버전 (`requirements-lock.txt`에 고정):

```
torch          2.5.1+cu121
torchvision    0.20.1+cu121
numpy          1.26.4      # imgaug가 numpy 2.x 미지원 → 1.x 고정 필수
opencv-python  4.10.0      # 5.0은 numpy>=2 요구 → 충돌
imgaug         0.4.0
pytorch_wavelets + PyWavelets 1.8.0
einops         0.8.2
tensorboardX   2.6.5
pandas         2.3.3
tqdm           4.70.1
torchinfo, fvcore (FLOPs 측정용)
```

추가 시스템 패키지: `libgl1`, `libglib2.0-0` (cv2 의존).

---

## 1. 저장소 결함 및 수정 (→ R1-9 재현성)

리뷰어 1이 요구한 "저장소 URL, 커밋 ID, 환경 명세, 재현 명령어"에 답하려면
아래를 모두 수정해야 한다. **현재 공개 커밋(`aaed89a`)은 실행 자체가 불가능하다.**

| # | 결함 | 영향 | 조치 | 상태 |
|---|---|---|---|---|
| S-1 | `models/backbones/{c3d,r3d,vgg}.py` 누락 | `temporal.py` import 실패 → 전체 실행 불가 | 로컬 백업(2026-05-18자)에서 복원 | 완료 |
| S-2 | `hyperparameters/Briareo/test_ir_tsm.json` 누락 | Briareo IR 평가 불가 (논문 최적 조합에 IR 포함) | depth config 기반 생성, 재현 검증됨 | 완료 |
| S-3 | `requirements.txt` 불완전 | imgaug·einops·pytorch-wavelets·PyWavelets·tensorboardX 누락, 버전 미고정 | 실제 의존성으로 교체 + 버전 고정 | 예정 |
| S-4 | `test.py` 선택적 import가 한 블록 | `torchstat` 미설치 시 `fvcore`·`torchinfo`까지 무력화 → FLOPs 측정 불가 | try/except 3개로 분리 | 완료 |
| S-5 | README에 실행법 없음 | `test.py` 직접 실행은 무동작 (진입점은 `main.py --phase test`) | 실행 명령 문서화 | 예정 |

올바른 실행 명령 (문서화 대상):

```bash
# 단일 모달리티 평가
python main.py --hypes hyperparameters/Briareo/test_rgb_tsm.json --phase test

# late fusion (모든 부분집합)
python cs.py --dataset Briareo \
  --csv_dir ./experiments/BL2_tsm/Briareo/csv \
  --results_dir ./experiments/BL2_tsm/Briareo/results --all
```

---

## 2. Phase 0-1 — 수식 (1)/(2) 판정 (→ R1-6)

**리뷰어 지적이 정확함.** 코드 추적 결과:

```
temporal.py:108    unimodal_logits = self.classifier(pooled)   # raw logit (softmax 없음)
train_perclass.py:226   logits = self.fusion(uni_logits)
perclass_fusion.py:16   z = torch.stack(unimodal_logits)       # raw logit을 그대로
                        w = F.softmax(self.W, dim=0)
                        return (w * z).sum(dim=0)              # logit의 가중합
```

- 식 (1) late fusion = **softmax 확률의 평균**
- 식 (2) W=0 = **logit의 균일 평균**
- softmax는 비선형이므로 두 값은 일반적으로 **같지 않음**
- 논문의 "exactly recovering the simple average of (1)"은 **성립하지 않는 주장**

차이 크기(시뮬레이션): 무작위 logit에서 예측 불일치 약 62%,
논문 수준의 단일모달 정확도를 가정하면 예측 불일치 약 5%, 정확도 차이 약 2pp.

### 대응 방향 (미확정 — CP-1에서 결정)

- **선택 A (권장)**: 구현을 확률 결합으로 수정 → "exactly recover" 주장이 실제로 참이 됨.
  Phase 1에서 어차피 재학습하므로 추가 비용 없음.
- 선택 B: 논문 서술을 구현에 맞게 정정. 재학습 불필요하나
  "late fusion을 하한으로 보장"이라는 논리가 약해짐.

---

## 3. Phase 0-2 — Briareo 수치 불일치 규명 (완료)

### 문제
보유 CSV가 논문 Table III과 불일치(Color 98.26 → 96.53 등, 최적 99.31 → 98.26).

### 원인
`experiments/BL2_tsm/Briareo/csv/`에 **BL1(GestFormer 베이스라인) 결과가 들어가 있었음.**
2026-06-02 대화 기록의 BL1/BL2 비교표와 대조해 확인:

| 모델 | Briareo | NVGesture |
|---|---|---|
| BL1 GestFormer | 98.26% | 85.06% |
| BL2 TSM (논문) | 99.31% | 89.63% |

보유 CSV가 산출한 값이 정확히 BL1과 일치 → 폴더 재정리 중 혼입된 것으로 판단.
데이터 손상이 아니라 **파일 혼선**.

### 해결
BL2 체크포인트(`best_tsm_briareo_*.pth`)로 재평가 → **논문 수치 완전 재현**.

| 모달리티 | 논문 | 재현 |
|---|---|---|
| Color | 98.26 | 98.26 |
| Depth | 96.18 | 96.18 |
| IR | 97.92 | 97.92 |
| Normal | 96.53 | 96.53 |
| Optical flow | 92.36 | 92.36 |
| **Color+Depth+IR** | **99.31** | **99.31** |
| 4모달 / 5모달 | 98.26 | 98.26 |

Table VI의 모든 부분집합 값도 일치. "부분집합이 전체보다 낫다"는 논문 주장 재확인.

기존 BL1 CSV는 `experiments/BL2_tsm/Briareo/csv_BL1_backup_20260518/`에 보존.

---

## 4. 확보된 자산 (재사용 가능)

| 자산 | 위치 | 용도 |
|---|---|---|
| BL2 체크포인트 (両 데이터셋) | `experiments/BL2_tsm/*/checkpoints/` | 재평가·재현 |
| LoRA 다중 시드 체크포인트 | `experiments/BL4_p0_multiseed/seed{1994,2024,777}/` | R1-1 기존 3시드 |
| per-class 산출물 | `experiments/BL4_perclass/` | R1-6 검증 |
| 학습 로그 | `experiments/BL4_p0_multiseed/*/train_run.log` | R1-3 분할 정보 추출 |
| 센서 고장 분석 | (계산 완료) | R1-7 조건부 이득 |

### 센서 고장 분석 결과 (R1-7 대응, 계산 완료)

NVGesture, late fusion 기준:

| 시나리오 | 잔존 모달리티 | 정확도 | Δ |
|---|---|---|---|
| 정상 | 5 | 89.63% | — |
| DUO 3D (스테레오 IR) 고장 | 4 | 89.63% | 0.00pp |
| DS325 (RGB-D) 고장 | 1 (IR만) | 62.45% | −27.18pp |

해석: IR 장치는 제거해도 무손실(논문 주장 보강). 단 DS325는 **단일 장애점** →
"센서를 줄이면 비용이 준다"에 "이중화가 사라진다"는 트레이드오프를 정직하게 병기할 근거.

---

## 5. 남은 작업

| Phase | 항목 | 대응 | 상태 |
|---|---|---|---|
| 0-3 | validation 분할 문서화 (subject-disjoint 여부, 샘플 수, 시드 간 고정) | R1-3, R1-4 | 다음 |
| 0-4 | 실험 매트릭스 확정 | R1-1 | 대기 |
| 1-1 | 전 fusion 변형 다중 시드 재학습 | R1-1, R1-2 | 대기 |
| 1-2 | per-class 궤적 다중 시드 + 불확실성 밴드 | R1-5 | 대기 |
| 1-3 | 교차 데이터셋 (라벨 체계 상이 → 백본 전이로 대체 검토) | R2-7 | 대기 |
| 2-1 | 상보성 정량화 (오차 중첩·불일치율·예측 상관) | R1-7 | 대기 |
| 3-* | 집필 (문헌 확장, 프레이밍, 주장 완화, 용어 일관화 등) | 다수 | 대기 |

### 주의: GPU 1슬라이스
신규 서버도 MIG 1g.10gb 단일 슬라이스 → **병렬 학습 불가**.
Phase 1 기간을 3~4주로 잡고, 변형별 시드 수 차등(핵심 변형 5시드 / 열세 변형 3시드) 검토.

---

## 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-09-29 | 신규 서버 환경 구축 시작, 구글 드라이브 체크포인트 복원 |
| 2026-09-30 | 데이터셋 복원, 의존성 확정, S-1·S-2·S-4 수정, Phase 0-2 완료 |

---

## 3.7 Phase 2 — 상보성 정량화 (완료) (→ R1-7)

`analyze_complementarity.py`로 저장된 CSV에서 계산 (GPU 불필요).

### 핵심 수치

| 지표 | NVGesture | Briareo |
|---|---|---|
| Q-statistic 범위 | 0.513 ~ 0.798 | **0.854 ~ 0.978** |
| 불일치율 범위 | 19.50 ~ 43.98% | 2.43 ~ 8.68% |
| Oracle 상한 | 96.89% | 99.31% |
| late fusion | 89.63% | 98.26% |
| **완벽한 선택기 가정 시 여지** | **7.26 pp** | **1.04 pp** |

### 조건부 이득 (한 모달리티 제거 시)

NVGesture: -IR **+0.00pp** (무손실), -Depth -2.70, -Opt.flow -2.70, -Normal -1.45, -Color -0.83
Briareo: 모든 단일 제거가 ±0.00pp (4모달 조합이 5모달과 동일 98.26%)

### 해석 — 논문 수정 필요

Briareo는 상보성이 거의 없음(Q>0.85, 여지 1.04pp)이 확인되어 기존 설명과 일치.
**그러나 NVGesture는 상보성이 상당히 존재**(Q 0.51~0.80, 여지 7.26pp).
 Discussion의 "both benchmarks ... highly redundant"는 NVGesture에 대해 과도한 일반화.

 수정 방향: 두 데이터셋이 **서로 다른 이유로** 같은 결론에 도달한다는 서술로 변경.
  - Briareo: 상보성 자체가 희박 (센서 수준 중복 — 같은 ToF 반사광의 위상/진폭)
  - NVGesture: 상보성은 있으나 **frozen feature + 소규모 데이터**로 도달 불가

 수정은 Table V 재실험 결과 확정 후 최종화할 것.

---

## 3.8 Phase 1 2단계 — fusion 다중 시드 (NVGesture, 15/21 완료)

: 901 train / 149 val (subject-disjoint) / 482 test, 재학습 백본(BL5_revision).
**late fusion 기준선 85.68%** (이 프로토콜 기준. 논문 본문 89.63%는 공식 프로토콜 값).

| 변형 | 시드 | 평균 ± 표준편차 | vs 기준선 | MSPE |
|---|---|---|---|---|
| Per-class weighting | 5 | **84.77 ± 0.11** | −0.91 pp | off |
| LoRA ensemble (P0) | 5 | **80.00 ± 1.16** | −5.68 pp | off |
| Bottleneck + stronger input | 5 | **73.90 ± 0.82** | −11.78 pp | off |
| Bottleneck (MBT+gate) | — | 미실행 (아래 참조) | — | **on** |
| Dense cross-attention | — | 미실행 (아래 참조) | — | **on** |

### 결과 해석 — 논문 주장이 강화됨

    프로토콜에서는 per-class가 late fusion과 **동일**(89.63 = 89.63), LoRA가 근소하게 아래(89.42)였으나,
apt update && apt install openssh-server -y 않은 validation에서는 **세 변형 모두 명확히 아래**이며 시드 변동(σ ≤ 1.16)보다 격차가 훨씬 크다.
 R1-2가 지적한 "주장이 증거보다 강하다"를 더 강한 증거로 충족.

**과적합의 직접 증거**: `mbt_bl2in`은 val 98.95~99.52%인데 test 72.61~74.69% (**격차 약 25 pp**).
per-class(약 8 pp), LoRA(약 10 pp)와 대비되며, "소규모 데이터에서 파라미터화된 fusion이 과적합한다"는
apt update && apt install openssh-server -y        주장에 직접 근거가 된다.

### mbt_gate / dense 미실행 사유 (S-10)

 config만 `use_mspe=True`(modality-specific positional encoding)로 설정되어 있어,
`use_mspe=False`로 재학습한 백본에서 `modality_embedding.weight` 키 부재로 로드 실패.

**중요 — 논문 각주 1 정정 필요**: 각주는 MSPE를 "탐색했으나 다루지 않는다"고 서술하나,
 **Table V의 dense(80.29%)와 MBT+gate(85.89%) 두 행이 MSPE를 켠 상태로 실험됨.**
 R1-12(아키텍처 세부 불충분)와 직결. 리비전에서 Table V에 MSPE 사용 여부를 명시하고
#
apt update && apt install openssh-server -y 정정할 것.

**실험 대응**: 두 변형은 기존 단일 실행 결과를 유지(C안). 근거는 효과 크기 —
dense −9.34 pp, MBT −3.74 pp(원 프로토콜)로, 접전 변형들에서 측정된 시드 변동(σ ≤ 1.16 pp)보다
 자릿수 이상 크다. 답변서에 반복 횟수 차등의 근거로 기술.
