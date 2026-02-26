"""
Page 3 — Evaluate
Confusion matrix, classification report, per-class accuracy.
"""

import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
matplotlib.use("Agg")

from utils import session


def render():
    st.markdown("# 📊 Model Evaluation")
    st.markdown('<p class="mono">// confusion matrix & classification report</p>', unsafe_allow_html=True)
    st.divider()

    session.init()

    if not st.session_state.get("model_trained"):
        st.info("ℹ️ Train a model first (🏋️ Train Model) to see evaluation results.")
        return

    model  = st.session_state.keras_model
    X_val  = st.session_state.X_val
    y_val  = st.session_state.y_val
    cnames = st.session_state.class_names
    hist   = st.session_state.history
    params = st.session_state.get("total_params", 0)

    # ── Top metrics row ───────────────────────────────────────────────────────
    preds_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(preds_proba, axis=1)
    y_true = np.argmax(y_val, axis=1)

    final_acc  = hist["val_acc"][-1]
    final_loss = hist["val_loss"][-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val Accuracy",  f"{final_acc*100:.1f}%")
    c2.metric("Val Loss",      f"{final_loss:.4f}")
    c3.metric("Parameters",    f"{params:,}")
    c4.metric("Val Samples",   len(X_val))

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🟧 Confusion Matrix",
        "📋 Classification Report",
        "📈 Training Curves",
        "🎯 Per-Class Accuracy",
    ])

    # ── Tab 1: Confusion Matrix ──────────────────────────────────────────────
    with tab1:
        cm = confusion_matrix(y_true, y_pred)

        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            fig, ax = plt.subplots(figsize=(max(5, len(cnames)*1.1), max(4, len(cnames)*1.0)))
            fig.patch.set_facecolor("#111118")
            ax.set_facecolor("#111118")

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=sns.color_palette("rocket_r", as_cmap=True),
                xticklabels=cnames,
                yticklabels=cnames,
                ax=ax,
                linewidths=0.5,
                linecolor="#2a2a3a",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_xlabel("Predicted Label", color="#6b6b80", fontsize=9)
            ax.set_ylabel("True Label",      color="#6b6b80", fontsize=9)
            ax.set_title("Confusion Matrix", color="#e8e8f0", fontsize=11, fontweight="bold", pad=12)
            ax.tick_params(colors="#e8e8f0", labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            st.markdown("#### 🔍 Matrix Interpretation")
            st.markdown("""
**Diagonal cells** (correct predictions) should be the highest numbers.

**Off-diagonal** cells indicate misclassifications.

| Symbol | Meaning |
|--------|---------|
| **TP** | Correctly predicted |
| **FP** | Wrong prediction |
| **FN** | Missed prediction |
""")
            # Normalized version
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig2, ax2 = plt.subplots(figsize=(max(4, len(cnames)*1.1), max(3.5, len(cnames)*0.9)))
            fig2.patch.set_facecolor("#111118")
            ax2.set_facecolor("#111118")
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=cnames,
                yticklabels=cnames,
                ax=ax2,
                vmin=0, vmax=1,
                cbar_kws={"shrink": 0.8},
            )
            ax2.set_title("Normalized Confusion Matrix", color="#e8e8f0", fontsize=9, fontweight="bold")
            ax2.tick_params(colors="#e8e8f0", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    # ── Tab 2: Classification Report ─────────────────────────────────────────
    with tab2:
        report_str = classification_report(y_true, y_pred, target_names=cnames)
        report_dict = classification_report(y_true, y_pred, target_names=cnames, output_dict=True)

        st.code(report_str, language="text")

        # Visualize precision / recall / f1
        classes_only = {k: v for k, v in report_dict.items()
                        if k not in ["accuracy", "macro avg", "weighted avg"]}
        labels_  = list(classes_only.keys())
        prec_    = [classes_only[l]["precision"] for l in labels_]
        rec_     = [classes_only[l]["recall"]    for l in labels_]
        f1_      = [classes_only[l]["f1-score"]  for l in labels_]

        x = np.arange(len(labels_))
        width = 0.25

        fig3, ax3 = plt.subplots(figsize=(max(6, len(labels_)*1.4), 4))
        fig3.patch.set_facecolor("#111118")
        ax3.set_facecolor("#111118")

        bars1 = ax3.bar(x - width, prec_, width, label="Precision", color="#7c3aed", alpha=0.9)
        bars2 = ax3.bar(x,          rec_,  width, label="Recall",    color="#06b6d4", alpha=0.9)
        bars3 = ax3.bar(x + width, f1_,   width, label="F1-Score",  color="#10b981", alpha=0.9)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_, color="#e8e8f0", fontsize=9)
        ax3.set_ylim(0, 1.15)
        ax3.set_ylabel("Score", color="#6b6b80", fontsize=9)
        ax3.set_title("Per-Class Precision / Recall / F1", color="#e8e8f0", fontsize=10, fontweight="bold")
        ax3.tick_params(colors="#6b6b80", labelsize=8)
        ax3.legend(fontsize=8, labelcolor="#e8e8f0", facecolor="#1a1a24", edgecolor="#2a2a3a")
        for sp in ax3.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax3.grid(axis="y", color="#2a2a3a", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # ── Tab 3: Training Curves ────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)

        def curve_fig(title, train_data, val_data, train_label, val_label, colors, ylabel):
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            fig.patch.set_facecolor("#111118")
            ax.set_facecolor("#111118")
            ep = range(1, len(train_data) + 1)
            ax.plot(ep, train_data, color=colors[0], linewidth=2, label=train_label, marker="o", markersize=3)
            ax.plot(ep, val_data,   color=colors[1], linewidth=2, label=val_label,   marker="s", markersize=3, linestyle="--")
            ax.set_title(title, color="#e8e8f0", fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Epoch", color="#6b6b80", fontsize=8)
            ax.set_ylabel(ylabel, color="#6b6b80", fontsize=8)
            ax.tick_params(colors="#6b6b80", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2a2a3a")
            ax.grid(color="#2a2a3a", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=8, labelcolor="#e8e8f0", facecolor="#1a1a24", edgecolor="#2a2a3a")
            plt.tight_layout()
            return fig

        fig_acc = curve_fig(
            "Accuracy over Epochs",
            hist["acc"], hist["val_acc"],
            "Train Acc", "Val Acc",
            ["#7c3aed", "#06b6d4"],
            "Accuracy"
        )
        col1.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

        fig_loss = curve_fig(
            "Loss over Epochs",
            hist["loss"], hist["val_loss"],
            "Train Loss", "Val Loss",
            ["#f59e0b", "#ef4444"],
            "Loss"
        )
        col2.pyplot(fig_loss, use_container_width=True)
        plt.close(fig_loss)

        # Overfitting diagnostic
        final_gap = abs(hist["acc"][-1] - hist["val_acc"][-1])
        if final_gap > 0.20:
            st.warning(f"⚠️ Potential **overfitting** detected (gap = {final_gap:.1%}). Try adding more data or increasing dropout.")
        elif final_gap < 0.05:
            st.success(f"✅ Model generalizes well (gap = {final_gap:.1%}).")

    # ── Tab 4: Per-Class Accuracy ─────────────────────────────────────────────
    with tab4:
        per_class_correct = np.zeros(len(cnames))
        per_class_total   = np.zeros(len(cnames))

        for true, pred in zip(y_true, y_pred):
            per_class_total[true] += 1
            if true == pred:
                per_class_correct[true] += 1

        per_class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0)

        fig4, ax4 = plt.subplots(figsize=(7, max(3, len(cnames) * 0.7)))
        fig4.patch.set_facecolor("#111118")
        ax4.set_facecolor("#111118")

        COLORS = ["#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"]
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(cnames))]

        bars = ax4.barh(cnames, per_class_acc, color=bar_colors, height=0.5)
        ax4.set_xlim(0, 1.15)
        ax4.axvline(1.0, color="#2a2a3a", linewidth=1)
        ax4.set_xlabel("Accuracy", color="#6b6b80", fontsize=9)
        ax4.set_title("Per-Class Accuracy on Validation Set", color="#e8e8f0", fontsize=10, fontweight="bold")
        ax4.tick_params(colors="#e8e8f0", labelsize=9)
        for sp in ax4.spines.values():
            sp.set_edgecolor("#2a2a3a")

        for bar, acc_val in zip(bars, per_class_acc):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{acc_val:.1%}", va="center", color="#e8e8f0", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        # Table
        import pandas as pd
        df = pd.DataFrame({
            "Class":    cnames,
            "Correct":  per_class_correct.astype(int),
            "Total":    per_class_total.astype(int),
            "Accuracy": [f"{a:.1%}" for a in per_class_acc],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
