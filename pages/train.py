"""
Page 2 — Model Training
Configure hyperparameters, build CNN, train with live metrics.
"""

import time
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from utils import session, model_builder, preprocessing


def render():
    st.markdown("# 🏋️ Model Training")
    st.markdown('<p class="mono">// configure & train your CNN</p>', unsafe_allow_html=True)
    st.divider()

    session.init()
    classes = st.session_state.classes

    # Guard
    if len(classes) < 2:
        st.warning("⚠️ Go to **📁 Dataset Builder** and create at least 2 classes with images first.")
        return

    min_imgs = min(len(v) for v in classes.values())
    if min_imgs < 2:
        st.warning("⚠️ Each class needs at least 2 images. Please add more images.")
        return

    total_imgs = sum(len(v) for v in classes.values())

    # ── Layout: Config | Live Metrics ────────────────────────────────────────
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("### ⚙️ Hyperparameters")

        epochs = st.slider("Epochs", 5, 100, 20, 5)
        lr = st.select_slider(
            "Learning Rate",
            options=[0.01, 0.003, 0.001, 0.0003, 0.0001],
            value=0.001,
            format_func=lambda x: f"{x:.4f}",
        )
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32)
        val_split = st.slider("Validation Split %", 10, 40, 20, 5) / 100
        img_size = st.selectbox("Input Image Size", [64, 96, 128], index=1)
        augment = st.checkbox("Data Augmentation", value=True, help="Flip, rotate, zoom during training")

        st.markdown("---")
        st.markdown("### 🏗️ CNN Architecture")
        st.code(f"""
Input: {img_size}×{img_size}×3
─────────────────────────
Conv2D(32, 3×3) → ReLU
MaxPooling2D(2×2)
Conv2D(64, 3×3) → ReLU
MaxPooling2D(2×2)
Conv2D(128, 3×3) → ReLU
MaxPooling2D(2×2)
─────────────────────────
Flatten
Dense(256) → ReLU
Dropout(0.5)
Dense({len(classes)}) → Softmax
─────────────────────────
Optimizer: Adam({lr})
Loss: Categorical CE
        """, language="text")

        train_btn = st.button("▶ START TRAINING", use_container_width=True)

    with right:
        st.markdown("### 📈 Live Training Progress")

        # Metric placeholders
        m1, m2, m3, m4 = st.columns(4)
        ep_metric     = m1.empty()
        acc_metric    = m2.empty()
        valacc_metric = m3.empty()
        loss_metric   = m4.empty()

        ep_metric.metric("Epoch", "—")
        acc_metric.metric("Train Acc", "—")
        valacc_metric.metric("Val Acc", "—")
        loss_metric.metric("Loss", "—")

        prog_bar  = st.progress(0, text="Waiting to train…")
        chart_ph  = st.empty()
        log_ph    = st.empty()

        # Show previous history if available
        if st.session_state.get("history"):
            _draw_curves(chart_ph, st.session_state.history, epochs)

    # ── TRAINING ─────────────────────────────────────────────────────────────
    if train_btn:
        with st.spinner("Preprocessing images…"):
            try:
                X_train, X_val, y_train, y_val, class_names = preprocessing.prepare(
                    classes, img_size, val_split, augment
                )
            except ValueError as e:
                st.error(str(e))
                return

        n_classes = len(class_names)
        st.session_state.class_names = class_names

        with st.spinner("Building model…"):
            mdl = model_builder.build(img_size, n_classes, lr)
            st.session_state.keras_model = mdl

        total_params = mdl.count_params()
        st.info(f"Model built — **{total_params:,} parameters** | Train: {len(X_train)} · Val: {len(X_val)} images")

        # Training loop
        history_acc, history_val_acc = [], []
        history_loss, history_val_loss = [], []
        logs = []

        for epoch in range(1, epochs + 1):
            # One epoch
            h = mdl.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=0,
            )

            acc     = h.history["accuracy"][0]
            val_acc = h.history["val_accuracy"][0]
            loss    = h.history["loss"][0]
            val_loss = h.history["val_loss"][0]

            history_acc.append(acc)
            history_val_acc.append(val_acc)
            history_loss.append(loss)
            history_val_loss.append(val_loss)

            # Update metrics
            ep_metric.metric("Epoch", f"{epoch}/{epochs}")
            acc_metric.metric("Train Acc", f"{acc*100:.1f}%")
            valacc_metric.metric("Val Acc", f"{val_acc*100:.1f}%")
            loss_metric.metric("Loss", f"{loss:.4f}")
            prog_bar.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")

            # Live log
            log_msg = f"[{epoch:>3}/{epochs}] acc={acc:.4f}  val_acc={val_acc:.4f}  loss={loss:.4f}  val_loss={val_loss:.4f}"
            logs.append(log_msg)
            log_ph.code("\n".join(logs[-12:]), language="text")

            # Live chart
            _draw_curves(
                chart_ph,
                dict(acc=history_acc, val_acc=history_val_acc,
                     loss=history_loss, val_loss=history_val_loss),
                epochs,
            )

        # Save results
        st.session_state.history = dict(
            acc=history_acc, val_acc=history_val_acc,
            loss=history_loss, val_loss=history_val_loss
        )
        st.session_state.X_val = X_val
        st.session_state.y_val = y_val
        st.session_state.model_trained = True
        st.session_state.total_params = total_params

        prog_bar.progress(1.0, text="✅ Training complete!")
        st.success(
            f"🎉 Training finished! Final Val Accuracy: **{history_val_acc[-1]*100:.1f}%** — "
            "Head to **📊 Evaluate** to see full metrics."
        )


# ── Helper: Draw dual-axis curves ────────────────────────────────────────────
def _draw_curves(placeholder, history, total_epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor("#111118")

    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#111118")
        ax.set_title(title, color="#e8e8f0", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", color="#6b6b80", fontsize=8)
        ax.set_ylabel(ylabel, color="#6b6b80", fontsize=8)
        ax.tick_params(colors="#6b6b80", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax.grid(color="#2a2a3a", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7.5, labelcolor="#e8e8f0",
                  facecolor="#1a1a24", edgecolor="#2a2a3a")

    ep = range(1, len(history["acc"]) + 1)

    # Accuracy
    ax1.plot(ep, history["acc"],     color="#7c3aed", linewidth=2, label="Train")
    ax1.plot(ep, history["val_acc"], color="#06b6d4", linewidth=2, label="Val", linestyle="--")
    ax1.set_ylim(0, 1.05)
    style_ax(ax1, "Accuracy", "Accuracy")

    # Loss
    ax2.plot(ep, history["loss"],     color="#f59e0b", linewidth=2, label="Train")
    ax2.plot(ep, history["val_loss"], color="#ef4444", linewidth=2, label="Val", linestyle="--")
    style_ax(ax2, "Loss", "Loss")

    plt.tight_layout()
    placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)
