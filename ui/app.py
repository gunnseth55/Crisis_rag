#pip install streamlit
#streamlit run ui/app.py

import streamlit as st
st.title("CrisisRag")
st.write("Your Crisis Support")
string=st.text_input("ask your concern, i'm there to help")

st.text_area("comments")
age = st.slider("Age", 0, 100, 25)
option = st.selectbox("Choose", ["Python", "Java", "C++"])

st.button("click me")



