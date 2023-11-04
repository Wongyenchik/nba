import streamlit as st
from PIL import Image

st.set_page_config(
    page_icon= "🏀",
    page_title= "NBA Analysis"
)

# Load your image
image = Image.open('Untitled design.png')
st.image(image, use_column_width=True, output_format='auto')
# Specify the file path of the audio file on your laptop
audio_file = open('NBA-on-TNT-Original-Theme-Music-_TubeRipper.com_-_AudioTrimmer.com_.ogg', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/ogg')

# st.image(image)
st.markdown("<h1 style='text-align: center;'>NBA Analysis 🏀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>This dashboard provides insightful visualizations and statistics, showcasing player performance, and NBA league trend. </p>", unsafe_allow_html=True)
st.markdown("<h2></h2>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; '>Topics</h2>", unsafe_allow_html=True)

# Point form using HTML unordered list
st.markdown("""
<ul >
  <li style="font-size: 20px;">Will number of international players influence the 3PA?</li>
  <li style="font-size: 20px;">Is height matters in NBA?</li>
  <li style="font-size: 20px;">Is it easier to get foul nowadays in NBA?</li>
  <li style="font-size: 20px;">Prediction on top 3 points per game player in NBA</li>
  <li style="font-size: 20px;">Does draft pick matters?</li>
  <li style="font-size: 20px;">Can NBA player improves their points per game by making more FGA?</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("<h2></h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Our Team</h2>", unsafe_allow_html=True)

profImage = Image.open('DSC00496.jpg')
profImage1 = Image.open('Marco.png')
profImage2 = Image.open('Hezron.png')

# Split the screen into two columns
col1, col2, col3 = st.columns(3)
with col1:
    st.image(profImage, use_column_width=True, output_format='auto')
    st.markdown("<p style='text-align: center; font-size: 20px;'>Wong Yen Chik</p>", unsafe_allow_html=True)

with col2:
    st.image(profImage1, use_column_width=True, output_format='auto')
    st.markdown("<p style='text-align: center; font-size: 20px;'>Marco Setiawan</p>", unsafe_allow_html=True)

with col3:
    st.image(profImage2, use_column_width=True, output_format='auto')
    st.markdown("<p style='text-align: center; font-size: 20px;'>Ling Yang En</p>", unsafe_allow_html=True)    

# # ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
