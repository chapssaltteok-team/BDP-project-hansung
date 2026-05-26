"""
image_classifier.py
===================
ResNet-50 기반 KBO 뉴스 썸네일 감성 분류
담당: 분류모델 담당자

입력: data/images/*.jpg (224×224)
출력: outputs/resnet_sentiment/ 모델 가중치

아키텍처
────────
ImageNet pretrained ResNet-50
→ FC 마지막 레이어만 교체 (512 → 3 class)
→ 초반 레이어 Freeze, 후반 + FC Fine-tuning

실험
────
Exp-1  ResNet-50, FC만 학습 (feature extraction)
Exp-2  ResNet-50, layer4 + FC 학습 (fine-tuning)

추가 (v2)
─────────
- 체크포인트 저장: outputs/checkpoint_resnet_{label}_ep{N}.pth (에폭마다)
- Timer: results/time_log.json 소요 시간 기록
- Grad-CAM: 모델이 이미지 어느 부분을 보는지 시각화
  → results/figures/gradcam_{label}/ 저장
  → 발표용: 우리 예상(투구/타격 구도) vs 실제 활성화 영역 비교

실행: python src/models/image_classifier.py
"""
import os, sys, json, random, time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

PROC_DIR   = 'data/processed'
IMG_DIR    = 'data/images'
OUTPUT_DIR = 'outputs'
RESULT_DIR = 'results'
FIG_DIR    = os.path.join(RESULT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL2ID   = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


# ── 소요 시간 측정 ────────────────────────────────────────────────────────────
class Timer:
    """작업별 소요시간 기록 → results/time_log.json"""
    def __init__(self, log_path: str = 'results/time_log.json'):
        self.log_path = log_path
        self.records  = {}
        if os.path.exists(log_path):
            with open(log_path, encoding='utf-8') as f:
                self.records = json.load(f)

    def start(self, name: str):
        self.records[name] = {'start': time.time(), 'end': None,
                              'elapsed_sec': None, 'elapsed_min': None}

    def end(self, name: str):
        if name not in self.records:
            return
        elapsed = time.time() - self.records[name]['start']
        self.records[name]['end']         = time.time()
        self.records[name]['elapsed_sec'] = round(elapsed, 2)
        self.records[name]['elapsed_min'] = round(elapsed / 60, 2)
        print(f"  ⏱ [{name}] 소요시간: {elapsed / 60:.1f}분 ({elapsed:.1f}초)")
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 이미지 변환 ───────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Normalize 역변환 (Grad-CAM 시각화용)
DENORM = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_map: pd.DataFrame, transform):
        url2path    = dict(zip(image_map['url'], image_map['local_path']))
        self.paths  = [url2path.get(u, '') for u in df['url'].tolist()]
        self.labels = df['sentiment_str'].map(LABEL2ID).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (180, 180, 180))
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── 모델 구성 ─────────────────────────────────────────────────────────────────
def build_resnet(freeze_mode: str = 'fc_only') -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    if freeze_mode == 'layer4_fc':
        for param in model.layer4.parameters():
            param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_LABELS),
    )
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds_all.extend(out.argmax(-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    report = classification_report(
        labels_all, preds_all,
        target_names=list(LABEL2ID.keys()),
        output_dict=True, zero_division=0
    )
    return total_loss / len(loader), report


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def grad_cam_visualization(model: nn.Module,
                           img_tensor: torch.Tensor,
                           class_idx: int,
                           save_path: str):
    """
    Grad-CAM: ResNet이 이미지의 어느 부분을 보고 판단하는지 시각화

    활용
    ────
    - 우리 예상: 투구/타격 '구도'에서 활성화 높을 것
    - Grad-CAM 결과와 비교 → 모델-주제 적합성 근거로 발표에 활용
    - Plan A: 예상 일치 → 스포츠 썸네일 특성과 모델 특성 잘 맞음
    - Plan B: 예상 불일치 → 색감/배경 영향 → 추가 모델(CLIP 등) 제안
    """
    model.eval()
    features, grads = [], []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_full_backward_hook(backward_hook)

    output = model(img_tensor.unsqueeze(0).to(DEVICE))
    model.zero_grad()
    output[0, class_idx].backward()

    weights = grads[0].mean(dim=[2, 3], keepdim=True)
    cam     = (weights * features[0]).sum(dim=1).squeeze()
    cam     = torch.relu(cam)
    cam     = cam / (cam.max() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    # 시각화
    img_vis = DENORM(img_tensor).clamp(0, 1)
    cam_np  = cam.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(to_pil_image(img_vis))
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')
    axes[1].imshow(to_pil_image(img_vis), alpha=0.6)
    axes[1].imshow(cam_np, cmap='jet', alpha=0.4,
                   extent=[0, 224, 224, 0])
    axes[1].set_title(f'Grad-CAM (예측: {ID2LABEL[class_idx]})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Grad-CAM 저장: {save_path}")


def run_gradcam_samples(model: nn.Module,
                        test_df: pd.DataFrame,
                        image_map: pd.DataFrame,
                        label: str,
                        n_samples: int = 6):
    """
    클래스별 2장씩 Grad-CAM 시각화 저장
    → results/figures/gradcam_{label}/
    """
    save_dir = os.path.join(FIG_DIR, f'gradcam_{label}')
    os.makedirs(save_dir, exist_ok=True)

    url2path = dict(zip(image_map['url'], image_map['local_path']))
    done     = {0: 0, 1: 0, 2: 0}
    per_cls  = n_samples // NUM_LABELS

    for _, row in test_df.iterrows():
        lbl = LABEL2ID.get(row['sentiment_str'], -1)
        if lbl == -1 or done[lbl] >= per_cls:
            continue

        path = url2path.get(row['url'], '')
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue

        img_tensor = EVAL_TRANSFORM(img)
        save_path  = os.path.join(save_dir, f'{ID2LABEL[lbl]}_{done[lbl]}.png')
        grad_cam_visualization(model, img_tensor, lbl, save_path)
        done[lbl] += 1

        if all(v >= per_cls for v in done.values()):
            break

    print(f"  Grad-CAM 완료 → {save_dir}/")


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_resnet(
    freeze_mode : str   = 'fc_only',
    epochs      : int   = 10,
    batch_size  : int   = 32,
    lr          : float = 1e-3,
    patience    : int   = 5,
    tag         : str   = '',
) -> dict:

    set_seed(SEED)
    label = tag or freeze_mode
    timer = Timer()
    timer.start(f'resnet_{label}')

    print(f"\n{'='*55}")
    print(f"  ResNet-50 이미지 분류  [{label}]")
    print(f"  freeze_mode={freeze_mode}  |  device={DEVICE}")
    print(f"{'='*55}")

    # ── 데이터 로드 ─────────────────────────────────────
    train_df  = pd.read_csv(os.path.join(PROC_DIR, 'train.csv'), encoding='utf-8-sig')
    val_df    = pd.read_csv(os.path.join(PROC_DIR, 'val.csv'),   encoding='utf-8-sig')
    test_df   = pd.read_csv(os.path.join(PROC_DIR, 'test.csv'),  encoding='utf-8-sig')
    image_map = pd.read_csv(os.path.join(PROC_DIR, 'image_map.csv'), encoding='utf-8-sig')

    print(f"  데이터: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_loader = DataLoader(
        NewsImageDataset(train_df, image_map, TRAIN_TRANSFORM),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader  = DataLoader(NewsImageDataset(val_df,  image_map, EVAL_TRANSFORM), batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(NewsImageDataset(test_df, image_map, EVAL_TRANSFORM), batch_size=batch_size, num_workers=2)

    # ── 모델 ─────────────────────────────────────────────
    model    = build_resnet(freeze_mode).to(DEVICE)
    n_params = count_params(model)
    print(f"  학습 파라미터 수: {n_params:,}")

    label_counts = train_df['sentiment_str'].map(LABEL2ID).value_counts().sort_index()
    weights      = torch.tensor(
        [1.0 / c for c in label_counts.values], dtype=torch.float32).to(DEVICE)
    criterion    = nn.CrossEntropyLoss(weight=weights)

    fc_params    = list(model.fc.parameters())
    other_params = [p for p in model.parameters()
                    if p.requires_grad and not any(p is fp for fp in fc_params)]
    optimizer = torch.optim.Adam([
        {'params': fc_params,    'lr': lr},
        {'params': other_params, 'lr': lr * 0.1},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    # ── 학습 루프 ─────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_report = evaluate(model, val_loader, criterion)
        val_f1 = val_report['macro avg']['f1-score']
        scheduler.step(val_loss)
        print(f"  Epoch {epoch:2d} | train={train_loss/len(train_loader):.4f} "
              f"| val={val_loss:.4f} | F1={val_f1:.4f}")

        # ── 체크포인트 저장 (에폭마다) ───────────────────
        ckpt_path = os.path.join(OUTPUT_DIR, f'checkpoint_resnet_{label}_ep{epoch}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  체크포인트 저장: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early Stopping (epoch {epoch})")
                break

    # ── 테스트 평가 ───────────────────────────────────────
    model.load_state_dict(best_state)
    _, test_report = evaluate(model, test_loader, criterion)
    test_f1  = test_report['macro avg']['f1-score']
    test_acc = test_report['accuracy']
    print(f"\n  [Test] Accuracy={test_acc:.4f}  Macro-F1={test_f1:.4f}")

    # ── 추론 시간 측정 ────────────────────────────────────
    model.eval()
    sample = next(iter(test_loader))[0][:1].to(DEVICE)
    with torch.no_grad():
        for _ in range(10): model(sample)
        t0 = time.time()
        for _ in range(500): model(sample)
    infer_ms = (time.time() - t0) / 500 * 1000
    print(f"  추론 속도: {infer_ms:.4f} ms/sample")

    # ── Grad-CAM 시각화 ───────────────────────────────────
    print(f"\n  Grad-CAM 시각화 생성 중...")
    run_gradcam_samples(model, test_df, image_map, label, n_samples=6)

    # ── 최종 모델 저장 ────────────────────────────────────
    save_dir = os.path.join(OUTPUT_DIR, f'resnet_sentiment_{label}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, os.path.join(save_dir, 'model.pth'))
    print(f"  최종 모델 저장: {save_dir}/model.pth")

    timer.end(f'resnet_{label}')

    result = {
        'model'       : f'ResNet_{label}',
        'accuracy'    : round(test_acc, 4),
        'macro_f1'    : round(test_f1, 4),
        'inference_ms': round(infer_ms, 4),
        'n_params'    : n_params,
        'elapsed_min' : timer.records[f'resnet_{label}']['elapsed_min'],
        'report'      : test_report,
    }
    path = os.path.join(RESULT_DIR, 'scores.json')
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
    data.append(result)
    json.dump(data, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    return result


if __name__ == '__main__':
    # Exp-1: FC만 학습 (빠름)
    train_resnet(freeze_mode='fc_only', epochs=10, tag='Exp1')

    # Exp-2: layer4 + FC 학습 (정확도 높음)
    train_resnet(freeze_mode='layer4_fc', epochs=15, tag='Exp2')

    print('\nResNet 이미지 분류 완료')