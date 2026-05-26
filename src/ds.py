import torch, os, json
import pandas as pd
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from torchvision.transforms.functional import to_pil_image

DEVICE   = torch.device('cpu')
LABEL2ID = {'긍정': 0, '중립': 1, '부정': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_ENG = {0: 'positive', 1: 'neutral', 2: 'negative'}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
DENORM = transforms.Compose([
    transforms.Normalize([0.,0.,0.],[1/0.229,1/0.224,1/0.225]),
    transforms.Normalize([-0.485,-0.456,-0.406],[1.,1.,1.]),
])

# 모델 로드
def build_resnet():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512),
        nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 3),
    )
    state = torch.load('outputs/resnet_sentiment_Exp2/model.pth', map_location='cpu')
    m.load_state_dict(state)
    return m.eval()

model = build_resnet()

# Grad-CAM
def grad_cam(model, img_tensor, class_idx, save_path):
    features, grads = [], []
    h1 = model.layer4.register_forward_hook(lambda m,i,o: features.append(o.detach()))
    h2 = model.layer4.register_full_backward_hook(lambda m,gi,go: grads.append(go[0].detach()))
    
    out = model(img_tensor.unsqueeze(0))
    model.zero_grad()
    out[0, class_idx].backward()
    
    if not grads:
        h1.remove(); h2.remove(); return
    
    cam = (grads[0].mean(dim=[2,3], keepdim=True) * features[0]).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    h1.remove(); h2.remove()
    
    img_vis = DENORM(img_tensor).clamp(0,1)
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].imshow(to_pil_image(img_vis)); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(to_pil_image(img_vis), alpha=0.6)
    axes[1].imshow(cam.numpy(), cmap='jet', alpha=0.4, extent=[0,224,224,0])
    axes[1].set_title(f'Grad-CAM ({ID2LABEL[class_idx]})'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f'저장: {save_path}')

# 실행
test_df   = pd.read_csv('data/processed/test.csv', encoding='utf-8-sig')
image_map = pd.read_csv('data/processed/image_map.csv', encoding='utf-8-sig')

# 경로 정규화
url2path = {}
for _, row in image_map.iterrows():
    fname = os.path.basename(row['local_path'].replace('\\', '/'))
    url2path[row['url']] = os.path.join('data', 'images', fname)

save_dir = 'results/figures/gradcam_Exp2'
os.makedirs(save_dir, exist_ok=True)

done = {0:0, 1:0, 2:0}
per_cls = 2

for _, row in test_df.iterrows():
    lbl = LABEL2ID.get(row['sentiment_str'], -1)
    if lbl == -1 or done[lbl] >= per_cls: continue
    try:
        img = Image.open(url2path.get(row['url'],'')).convert('RGB')
    except: continue
    
    grad_cam(model, EVAL_TRANSFORM(img), lbl,
             os.path.join(save_dir, f'{LABEL_ENG[lbl]}_{done[lbl]}.png'))
    done[lbl] += 1
    if all(v >= per_cls for v in done.values()): break

print('완료!')