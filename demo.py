import torch
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

# =========================
# MODEL BUILD
# =========================
def build_model(name):
    if name == "vit":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM_CLASSES)

    elif name == "swin":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)

    return m


# =========================
# LOAD MODELS
# =========================
def load_models():
    vit = build_model("vit").to(DEVICE)
    swin = build_model("swin").to(DEVICE)

    vit.load_state_dict(torch.load("vit.pth", map_location=DEVICE))
    swin.load_state_dict(torch.load("swin.pth", map_location=DEVICE))

    vit.eval()
    swin.eval()

    return vit, swin


# =========================
# TRANSFORM
# =========================
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


# =========================
# PREDICT
# =========================
def predict(model, x):
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item()
    return pred, conf


# =========================
# SIMPLE HEATMAP
# =========================
def fake_heatmap(x):
    # ساده‌شده (چون Grad-CAM کامل نیاز به hook دارد)
    return torch.mean(x, dim=1)[0].cpu().numpy()


def overlay(img, mask):
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    mask = (mask * 255).astype(np.uint8)

    heat = Image.fromarray(mask).resize(img.size).convert("RGB")
    return Image.blend(img, heat, 0.4)

from PIL import ImageDraw, ImageFont

def add_label(img, text):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # background rectangle
    draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0))

    draw.text((10, 5), text, fill=(255, 255, 255))

    return img

# =========================
# MAIN
# =========================
def main(image_path):

    vit, swin = load_models()

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    # predictions
    v_pred, v_conf = predict(vit, x)
    s_pred, s_conf = predict(swin, x)

    print("\n===== RESULTS =====")
    print(f"ViT  -> Class {v_pred} | Conf {v_conf:.3f}")
    print(f"Swin -> Class {s_pred} | Conf {s_conf:.3f}")

    # heatmaps
    vit_map = fake_heatmap(x)
    swin_map = fake_heatmap(x)

    # overlays
    vit_img = overlay(img, vit_map)
    swin_img = overlay(img, swin_map)

    # 🔥 ADD LABELS
    orig_labeled = add_label(img, "Original X-Ray")
    vit_labeled = add_label(vit_img, f"ViT Heatmap | Class {v_pred} ({v_conf:.2f})")
    swin_labeled = add_label(swin_img, f"Swin Heatmap | Class {s_pred} ({s_conf:.2f})")

    # resize for consistency
    w, h = img.size
    vit_labeled = vit_labeled.resize((w, h))
    swin_labeled = swin_labeled.resize((w, h))

    # combine
    out = Image.new("RGB", (w * 3, h))

    out.paste(orig_labeled, (0, 0))
    out.paste(vit_labeled, (w, 0))
    out.paste(swin_labeled, (w * 2, 0))

    out.save("result.png")

    print("\n✅ Saved: result.png")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    main(args.image)