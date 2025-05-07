import streamlit as st

st.set_page_config(page_title="Streamlit Test", layout="wide")

# Add custom CSS
st.markdown("""
<style>
.test-box {
    background-color: lightblue;
    padding: 20px;
    border-radius: 10px;
    font-family: 'Arial', sans-serif;
    color: navy;
}
</style>
""", unsafe_allow_html=True)

st.title("Streamlit Test App")
st.markdown("<div class='test-box'>This box should be light blue with navy text.</div>", unsafe_allow_html=True)

# Test the theme colors
st.write("Testing primary color")
st.button("Test Button")

# Test built-in elements
st.slider("Test Slider", 0, 100, 50)
st.selectbox("Test Dropdown", ["Option 1", "Option 2", "Option 3"])