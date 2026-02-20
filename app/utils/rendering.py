import matplotlib.pyplot as plt
import streamlit as st

from utils.citations import CitationStyle


###Confidence profile

def render_confidence_profile(confidence): 
    st.subheader("Confidence Profile") 
    evidence = confidence["evidence"] 
    grounding = confidence["grounding"] 
    col1, col2 = st.columns(2, gap="large") 
    with col1: 
        st.metric("Evidence structure", f"{evidence['score']:.2f}", evidence['level']) 
        with st.expander("Why this score?"): 
            for bullet in evidence["explanation"]: 
                st.markdown(f"- {bullet}") 
                
    with col2: 
        st.metric("Grounding quality", f"{grounding['score']:.2f}", grounding['level']) 
        with st.expander("Why this score?"): 
            for bullet in grounding["explanation"]: 
                st.markdown(f"- {bullet}")


###Citations

def render_sentence_with_inline_citations(item, citation_style: CitationStyle):
    text = item["text"]
    citations = item.get("citations", [])

    if not citations:
        return f"- {text}"

    if citation_style == CitationStyle.NUMERIC:
        citation_str = ", ".join(citations)
        return f"- {text} [{citation_str}]"

    elif citation_style == CitationStyle.AUTHOR_YEAR:
        citation_str = "; ".join(citations)
        return f"- {text} [{citation_str}]"

    else:
        return f"- {text}"
    

###Limitations

def show_limitations(data, level="warning"):
    limitations = data.get("limitations", [])
    for lim in limitations:
        if level == "error":
            st.error(lim)
        else:
            st.warning(lim)


###Metadata

def show_metadata(data):
    if data.get("meta", {}):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Metadata", expanded=False):
            st.json(data.get("meta", {}))