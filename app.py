import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# The Open AI API Key
os.environ['OPENAI_API_KEY'] = apikey

# Setup the app UI using StreamLit
st.title('🦜️🔗 LangChain for Alireza')
subject = st.text_input('Subject')
style = st.text_input('Poet')


text_template = PromptTemplate(
    input_variables=['style', 'subject'],
    template="Write me a poem with maximum of 20 words in the style of {style}, about {subject}"
)

image_template = PromptTemplate(
    input_variables=['style', 'subject'],
    template="Create an image in the style of {style}, about {subject}"
)

# LLM
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=text_template)

# Show the LLM response
if subject and style:
    response = chain.run({'style': style,
                          'subject': subject})
    st.write(response)
