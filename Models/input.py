import streamlit as st
 
st.checkbox('Yes')
st.button('Click Me')
st.radio('Pick your gender', ['Male', 'Female'])
st.selectbox('choose a program Name', ['TSP', 'Unnati', 'IBM'])
st.multiselect('Choose a supervisor Name', ['Bharti', 'vidhi', 'shabaz'])
st.select_slider('ICBP certification', ['DONE', 'no', 'Still progressing'])
st.slider('Pick a number', 0, 100)
 
 
st.number_input('Pick a number between (1-10)', 0, 10)
st.text_input('Email address')
st.date_input('Traveling date')
st.time_input('Office time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')
 