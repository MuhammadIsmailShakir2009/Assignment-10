"""utils/preprocessing.py — Image preprocessing & augmentation."""

import numpy as np
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def prepare(
    classes: dict,
    img_size: int,
    val_split: float,
    augment: bool = True,
):
    """
    Preprocess images from class dict into train/val splits.

    Returns:
        X_train, X_val, y_train, y_val, class_names
    """
    class_names = list(classes.keys())
    n_classes   = len(class_names)

    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")

    X, y = [], []

    for label_idx, cls_name in enumerate(class_names):
        images = classes[cls_name]
        if len(images) < 2:
            raise ValueError(f'Class "{cls_name}" has fewer than 2 images.')

        for arr in images:
            # Resize & normalize
            pil = Image.fromarray(arr).resize((img_size, img_size))
            x   = np.array(pil, dtype=np.float32) / 255.0
            X.append(x)
            y.append(label_idx)

            # Augmentation
            if augment:
                aug_list = _augment(pil, img_size)
                for aug in aug_list:
                    X.append(aug)
                    y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # One-hot encode
    y_cat = to_categorical(y, num_classes=n_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=val_split,
        random_state=42,
        stratify=y,
    )

    # Store img_size for predict page
    st.session_state.img_size = img_size

    return X_train, X_val, y_train, y_val, class_names


def _augment(pil_img: Image.Image, img_size: int) -> list:
    """Generate augmented variants of one PIL image."""
    augmented = []

    # Horizontal flip
    flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    augmented.append(np.array(flipped, dtype=np.float32) / 255.0)

    # Brightness variations
    from PIL import ImageEnhance
    for factor in [0.75, 1.25]:
        bright = ImageEnhance.Brightness(pil_img).enhance(factor)
        augmented.append(np.array(bright.resize((img_size, img_size)), dtype=np.float32) / 255.0)

    # Slight rotation
    for angle in [-15, 15]:
        rotated = pil_img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
        augmented.append(np.array(rotated.resize((img_size, img_size)), dtype=np.float32) / 255.0)

    # Random crop + resize
    w, h = pil_img.size
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    cropped = pil_img.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    augmented.append(np.array(cropped.resize((img_size, img_size)), dtype=np.float32) / 255.0)

    return augmented
