"""
Page 4 — Predict
Upload or capture image, run inference, show confidence scores.
"""

import io
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from PIL import Image

from utils import session


COLORS = ["#7c3aed","#06b6d4","#f59e0b","#10b981","#ef4444","#8b5cf6","#ec4899","#14b8a6"]


def render():
    st.markdown("# 🔮 Real-Time Prediction")
    st.markdown('<p class="mono">// classify new images with your trained model</p>', unsafe_allow_html=True)
    st.divider()

    session.init()

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Train a model first (🏋️ Train Model) before running predictions.")
        return

    model  = st.session_state.keras_model
    cnames = st.session_state.class_names
    img_size = st.session_state.get("img_size", 96)

    left, right = st.columns([1, 1.2], gap="large")

    # ── Left: Upload ──────────────────────────────────────────────────────────
    with left:
        st.markdown("### 🖼️ Input Image")

        source = st.radio("Source", ["Upload File", "Camera Capture"], horizontal=True)

        img_array = None

        if source == "Upload File":
            uploaded = st.file_uploader(
                "Drop an image here",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded:
                img_pil = Image.open(uploaded).convert("RGB")
                img_array = np.array(img_pil)
                st.image(img_pil, caption="Uploaded Image", use_container_width=True)

        else:  # Camera
            cam_img = st.camera_input("Take a photo")
            if cam_img:
                img_pil = Image.open(cam_img).convert("RGB")
                img_array = np.array(img_pil)

        # Batch prediction — multiple files
        st.divider()
        st.markdown("#### 📦 Batch Predict")
        batch_files = st.file_uploader(
            "Upload multiple images for batch prediction",
            type=["jpg","jpeg","png","webp"],
            accept_multiple_files=True,
            key="batch_upload",
        )

    # ── Right: Results ────────────────────────────────────────────────────────
    with right:
        st.markdown("### 🎯 Prediction Results")

        if img_array is not None:
            # Preprocess
            resized = np.array(Image.fromarray(img_array).resize((img_size, img_size))) / 255.0
            tensor  = resized.reshape(1, img_size, img_size, 3).astype(np.float32)

            probs = model.predict(tensor, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_name = cnames[top_idx]

            # Top prediction banner
            color = COLORS[top_idx % len(COLORS)]
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}11);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 20px 24px;
                    margin-bottom: 16px;
                ">
                    <div style="font-size:0.72rem;color:#6b6b80;font-family:'Space Mono',monospace;margin-bottom:4px;">PREDICTION</div>
                    <div style="font-size:2rem;font-weight:800;color:{color};">{top_name}</div>
                    <div style="font-size:0.85rem;color:#e8e8f0;font-family:'Space Mono',monospace;">
                        Confidence: <strong style="color:{color};">{top_conf*100:.1f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence bar chart (seaborn-style via matplotlib)
            fig, ax = plt.subplots(figsize=(6, max(2.5, len(cnames) * 0.55)))
            fig.patch.set_facecolor("#111118")
            ax.set_facecolor("#111118")

            sorted_idx = np.argsort(probs)[::-1]
            sorted_names = [cnames[i] for i in sorted_idx]
            sorted_probs = [probs[i] * 100 for i in sorted_idx]
            bar_colors   = [COLORS[i % len(COLORS)] for i in sorted_idx]

            bars = ax.barh(sorted_names[::-1], sorted_probs[::-1],
                           color=bar_colors[::-1], height=0.55)
            ax.set_xlim(0, 115)
            ax.set_xlabel("Confidence %", color="#6b6b80", fontsize=8)
            ax.set_title("Class Probabilities", color="#e8e8f0", fontsize=10, fontweight="bold")
            ax.tick_params(colors="#e8e8f0", labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a3a")
            ax.grid(axis="x", color="#2a2a3a", linestyle="--", linewidth=0.5)
            for bar, pct in zip(bars, sorted_probs[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", va="center", color="#e8e8f0", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Preprocessing preview
            with st.expander("🔬 View Preprocessed Input"):
                col_a, col_b = st.columns(2)
                col_a.image(
                    img_array,
                    caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})",
                    use_container_width=True,
                )
                processed_vis = (resized * 255).astype(np.uint8)
                col_b.image(
                    processed_vis,
                    caption=f"Model input ({img_size}×{img_size})",
                    use_container_width=True,
                )

        else:
            st.info("⬅️ Upload or capture an image to see predictions here.")

    # ── Batch Results ─────────────────────────────────────────────────────────
    if batch_files:
        st.divider()
        st.markdown("### 📦 Batch Prediction Results")

        import pandas as pd
        rows = []
        batch_cols = st.columns(min(len(batch_files), 5))

        for i, f in enumerate(batch_files):
            img_pil = Image.open(f).convert("RGB")
            arr = np.array(img_pil.resize((img_size, img_size))) / 255.0
            tensor = arr.reshape(1, img_size, img_size, 3).astype(np.float32)
            probs = model.predict(tensor, verbose=0)[0]
            pred_idx  = int(np.argmax(probs))
            pred_name = cnames[pred_idx]
            confidence = float(probs[pred_idx]) * 100

            rows.append({"File": f.name, "Predicted": pred_name, "Confidence": f"{confidence:.1f}%"})

            col = batch_cols[i % len(batch_cols)]
            color = COLORS[pred_idx % len(COLORS)]
            col.image(img_pil, use_container_width=True)
            col.markdown(
                f'<div style="text-align:center;color:{color};font-weight:800;font-size:0.85rem;">'
                f'{pred_name}<br><span style="color:#6b6b80;font-size:0.7rem;">{confidence:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("#### Summary Table")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Distribution pie
        if len(rows) > 1:
            from collections import Counter
            counts = Counter(r["Predicted"] for r in rows)
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            fig_pie.patch.set_facecolor("#111118")
            ax_pie.pie(
                counts.values(),
                labels=counts.keys(),
                autopct="%1.0f%%",
                colors=[COLORS[cnames.index(k) % len(COLORS)] for k in counts],
                textprops={"color": "#e8e8f0", "fontsize": 9},
            )
            ax_pie.set_title("Batch Prediction Distribution", color="#e8e8f0", fontsize=10)
            st.pyplot(fig_pie, use_container_width=False)
            plt.close(fig_pie)
