import os
import shutil
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

def get_dataloaders(data_dir='tiny-imagenet/tiny-imagenet-200', batch_size=32, num_workers=2):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset non trovato in '{data_dir}'. "
            "Se sei in locale è normale: scarica il dataset solo su Google Colab."
        )

    annotations_path = os.path.join(data_dir, 'val', 'val_annotations.txt')
    val_images_path = os.path.join(data_dir, 'val', 'images')

    if os.path.exists(annotations_path) and os.path.exists(val_images_path):
        with open(annotations_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                fn, cls = parts[0], parts[1]
                target_dir = os.path.join(data_dir, 'val', cls)
                os.makedirs(target_dir, exist_ok=True)
                src = os.path.join(val_images_path, fn)
                dst = os.path.join(target_dir, fn)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
        shutil.rmtree(val_images_path)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_set = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader