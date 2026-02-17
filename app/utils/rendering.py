import streamlit as st

def show_limitations(data, level="warning"):
    limitations = data.get("limitations", [])
    for lim in limitations:
        if level == "error":
            st.error(lim)
        else:
            st.warning(lim)


def show_metadata(data):
    if data.get("meta", {}):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Metadata", expanded=False):
            st.json(data.get("meta", {}))