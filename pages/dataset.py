"""
Page 1 — Dataset Builder
Create classes and upload training images.
"""

import io
import streamlit as st
from PIL import Image
import numpy as np
from utils import session


COLORS = [
    "#7c3aed", "#06b6d4", "#f59e0b", "#10b981",
    "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6",
]


def render():
    st.markdown("# 📁 Dataset Builder")
    st.markdown('<p class="mono">// create classes & upload training images</p>', unsafe_allow_html=True)
    st.divider()

    session.init()

    # ── Add new class ─────────────────────────────────────────────────────────
    with st.expander("➕ Add New Class", expanded=len(st.session_state.classes) == 0):
        col1, col2 = st.columns([3, 1])
        class_name = col1.text_input(
            "Class Name",
            placeholder="e.g. Cat, Dog, Hand Gesture A…",
            max_chars=30,
        )
        col2.write("")
        col2.write("")
        if col2.button("CREATE CLASS", use_container_width=True):
            name = class_name.strip()
            if not name:
                st.warning("Please enter a class name.")
            elif name.lower() in [k.lower() for k in st.session_state.classes]:
                st.error(f'Class "{name}" already exists.')
            else:
                st.session_state.classes[name] = []
                st.success(f'✅ Class "{name}" created!')
                st.rerun()

    st.divider()

    # ── Existing classes ──────────────────────────────────────────────────────
    classes = st.session_state.classes

    if not classes:
        st.info("No classes yet. Create at least **2 classes** to begin training.")
        return

    for idx, (cls_name, images) in enumerate(list(classes.items())):
        color = COLORS[idx % len(COLORS)]
        count = len(images)
        status = "🟢" if count >= 30 else "🟡" if count >= 5 else "🔴"

        with st.container():
            col_head1, col_head2, col_head3 = st.columns([5, 2, 1])
            col_head1.markdown(
                f'<span style="color:{color};font-size:1.1rem;font-weight:800;">● {cls_name}</span>',
                unsafe_allow_html=True,
            )
            col_head2.markdown(
                f'<span class="mono">{status} {count} images</span>',
                unsafe_allow_html=True,
            )
            if col_head3.button("✕", key=f"del_{cls_name}", help=f"Delete class {cls_name}"):
                del st.session_state.classes[cls_name]
                st.rerun()

        # Upload area
        uploaded = st.file_uploader(
            f"Upload images for **{cls_name}**",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"upload_{cls_name}",
            label_visibility="collapsed",
        )

        if uploaded:
            new_count = 0
            for f in uploaded:
                try:
                    img = Image.open(f).convert("RGB")
                    arr = np.array(img)
                    # Avoid duplicates by checking size/content hash
                    already = any(np.array_equal(arr, ex) for ex in images)
                    if not already:
                        images.append(arr)
                        new_count += 1
                except Exception as e:
                    st.warning(f"Could not load {f.name}: {e}")
            if new_count:
                st.session_state.classes[cls_name] = images
                st.success(f"Added {new_count} new image(s) to '{cls_name}'")
                st.rerun()

        # Thumbnail strip
        if images:
            show = images[:10]
            cols = st.columns(len(show))
            for col, arr in zip(cols, show):
                thumb = Image.fromarray(arr).resize((80, 80))
                col.image(thumb, use_container_width=False, width=80)
            if len(images) > 10:
                st.caption(f"… and {len(images)-10} more images")

        st.divider()

    # ── Dataset Summary ───────────────────────────────────────────────────────
    st.markdown("### 📊 Dataset Summary")
    total = sum(len(v) for v in classes.values())
    min_imgs = min(len(v) for v in classes.values()) if classes else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Classes", len(classes))
    c2.metric("Total Images", total)
    c3.metric("Min per Class", min_imgs)
    c4.metric("Recommended Min", "30")

    # Per-class bar chart
    if classes:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor("#111118")
        ax.set_facecolor("#111118")

        names = list(classes.keys())
        counts = [len(v) for v in classes.values()]
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

        bars = ax.barh(names, counts, color=bar_colors, height=0.5)
        ax.axvline(30, color="#6b6b80", linestyle="--", linewidth=1, label="30 img target")
        ax.set_xlabel("Image count", color="#6b6b80", fontsize=9)
        ax.tick_params(colors="#e8e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a3a")
        ax.xaxis.label.set_color("#6b6b80")
        ax.yaxis.label.set_color("#6b6b80")
        plt.setp(ax.get_xticklabels(), color="#6b6b80", fontsize=8)
        plt.setp(ax.get_yticklabels(), color="#e8e8f0", fontsize=9)

        for bar, c in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(c), va='center', color='#e8e8f0', fontsize=8)

        ax.legend(fontsize=8, labelcolor="#6b6b80", facecolor="#1a1a24", edgecolor="#2a2a3a")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Proceed button ────────────────────────────────────────────────────────
    st.divider()
    ready = len(classes) >= 2 and min_imgs >= 2
    if ready:
        st.success(f"✅ Dataset ready! {len(classes)} classes · {total} images total. Head to **🏋️ Train Model** to continue.")
    else:
        if len(classes) < 2:
            st.warning("⚠️ Add at least **2 classes** to enable training.")
        else:
            st.warning(f"⚠️ Each class needs at least **2 images** (recommended: 30+). Minimum class has {min_imgs}.")
