import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
DATA_DIR = r"D:\Chest X-Ray Pneumonia Classification Using Transformer\dataset"
MODELS_TO_TEST = ["vit", "swin"]

BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-4
PATIENCE = 4


# =========================
# DATASET
# =========================
class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []

        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for c in self.classes:
            folder = os.path.join(root, c)
            for f in os.listdir(folder):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(folder, f))
                    self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        y = self.labels[i]

        if self.transform:
            img = self.transform(img)

        return img, y


# =========================
# TRANSFORMS
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


# =========================
# SPLIT (FIXED)
# =========================
full = ChestXRayDataset(DATA_DIR)

n = len(full)
train_n = int(0.7 * n)
val_n = int(0.15 * n)
test_n = n - train_n - val_n

indices = list(range(n))
train_idx, val_idx, test_idx = random_split(indices, [train_n, val_n, test_n])


train_ds = Subset(ChestXRayDataset(DATA_DIR, train_tf), train_idx.indices)
val_ds   = Subset(ChestXRayDataset(DATA_DIR, val_tf), val_idx.indices)
test_ds  = Subset(ChestXRayDataset(DATA_DIR, val_tf), test_idx.indices)


# =========================
# CLASS BALANCE
# =========================
labels = [full.labels[i] for i in train_idx.indices]
counts = np.bincount(labels)
weights = 1. / (counts + 1e-8)

sample_w = [weights[l] for l in labels]
sampler = WeightedRandomSampler(sample_w, len(sample_w))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODELS
# =========================
def build_model(name):
    if name == "vit":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)

    elif name == "swin":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)

    return m


# =========================
# METRICS
# =========================
def acc(p,t): return np.mean(p==t)

def f1(p,t):
    f=[]
    for c in range(NUM_CLASSES):
        tp=np.sum((p==c)&(t==c))
        fp=np.sum((p==c)&(t!=c))
        fn=np.sum((p!=c)&(t==c))
        pr=tp/(tp+fp+1e-8)
        rc=tp/(tp+fn+1e-8)
        f.append(2*pr*rc/(pr+rc+1e-8))
    return np.mean(f)


def eval_model(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x,y in loader:
            x=x.to(DEVICE)
            out=model(x)
            preds.extend(torch.argmax(out,1).cpu().numpy())
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)


# =========================
# TRAIN
# =========================
def train_one(name):
    print(f"\n🔥 Training {name}")

    model = build_model(name).to(DEVICE)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best, patience = 0, 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total=0

        for x,y in tqdm(train_loader):
            x,y=x.to(DEVICE),y.to(DEVICE)

            opt.zero_grad()
            out=model(x)
            loss=loss_fn(out,y)
            loss.backward()
            opt.step()

            total+=loss.item()

        p,t = eval_model(model,val_loader)

        score=f1(p,t)
        a=acc(p,t)

        print(f"{name} | Epoch {epoch} | Acc {a:.3f} | F1 {score:.3f}")

        if score>best:
            best=score
            torch.save(model.state_dict(), f"{name}.pth")
            patience=0
        else:
            patience+=1

        if patience>=PATIENCE:
            break

    model.load_state_dict(torch.load(f"{name}.pth", weights_only=True))
    return model


# =========================
# MAIN
# =========================
if __name__=="__main__":

    models_dict={}

    for name in MODELS_TO_TEST:
        models_dict[name]=train_one(name)

    print("\n✅ Training Done")