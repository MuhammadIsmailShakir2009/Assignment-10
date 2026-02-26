"""utils/session.py — Initialize session state."""
import streamlit as st


def init():
    defaults = {
        "classes": {},           # {name: [np.array, ...]}
        "class_names": [],
        "keras_model": None,
        "model_trained": False,
        "history": None,
        "X_val": None,
        "y_val": None,
        "total_params": 0,
        "img_size": 96,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
