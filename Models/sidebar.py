import streamlit as st
st.sidebar.markdown("This is the sidebar content")
st.sidebar.title("ICBP certification KT session")
st.sidebar.button("click")
st.sidebar.radio("Pick your team",["TSP","S4F","EY","CU"])
 
container = st.container()
container.write("This is written inside the container")
 
st.write("This is written outside the container") 