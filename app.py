import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# The Open AI API Key
os.environ['OPENAI_API_KEY'] = apikey

# Setup the app UI using StreamLit
st.title('🦜️🔗 LangChain for Alireza')
output_type = st.selectbox('What Kind of Output you are looking for?', ("Image", "Text"))
subject = st.text_input('Subject')
style = st.text_input('Style')
output_title = '<p style="font-family:fantasy; color:rgb(135,62,35); font-size: 50px;">Output</p>'
st.markdown(output_title, unsafe_allow_html=True)


text_template = PromptTemplate(
    input_variables=['style', 'subject'],
    template="Write me a poem with maximum of 20 words in the style of {style}, about {subject}"
)

image_template = PromptTemplate(
    input_variables=['style', 'subject'],
    template="Generate a detailed prompt to generate an image based on the following description:"
             "the style of the image is {style}, and the subject of the image is {subject}"
)

llm = OpenAI(temperature=0.7)
if output_type == "Text":
    chain = LLMChain(llm=llm, prompt=text_template)

    # Show the LLM response
    if subject and style:
        response = chain.run({'style': style,
                              'subject': subject})
        st.write(response)
else:
    chain = LLMChain(llm=llm, prompt=image_template)
    image_url = DallEAPIWrapper().run(chain.run({'style': style,
                                                 'subject': subject}))
    st.image(image_url, width=400)
