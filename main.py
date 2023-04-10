import streamlit as st
from io import StringIO
from langchain import PromptTemplate
from langchain.llms import OpenAI

st.set_page_config(page_title="LLM Translation", page_icon=":earth_americas:", layout="wide")
st.header("LLM Translation")

col1, col2 = st.columns(2)

with col1:
    model = st.radio(
        "Select your LLM model",
        ('text-davinci-00', 'gpt-3.5-turbo', 'gpt-4'))

with col2:
    openai_key = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")

    languages = st.multiselect(
        'Select languages',
        ['Assamese', 'Bengali', 'Hindi', 'Gujarati', 'Kannada', 'Malayalam', 'Marathi', 'Oriya', 'Punjabi', 'Tamil', 'Telugu'],
        ['Hindi'])

few_shot_placeholder = """
    English: Hi, my name is sumanth. I'm a PhD student at IIT Madras.
    Hindi: हाय, मेरा नाम सुमंत है। मैं आईआईटी मद्रास में पीएचडी का छात्र हूं।

    English: I'm from India. I'm a big fan of Indian food.
    Hindi: मैं भारत से हूं। मैं भारतीय भोजन का एक बड़ा प्रशंसक हूं।
"""

few_shot = st.text_area(label="Help with some few-shot examples",  placeholder=few_shot_placeholder, key="few_shot")

dcol1, dcol2 = st.columns(2)

with dcol1:
    st.subheader("Traslation for a single sentence")
    input_sent = st.text_input(label="Input Sentence ",  placeholder="", key="input_sent")

with dcol2:
    st.subheader("Upload a file to translate")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"), newline='\n')
        string_data = stringio.read()
        lines = string_data.splitlines()
        lines = [line.strip() for line in lines]

        st.write("Check if the file is correct")
        st.write(lines[:4])

        input_sent = lines[0]

template_with_few_shot = """
    Here are some examples of translations:
    {few_shot}
    Using these examples, can you help translate from English to {languages}.
    English: {input_sent}
"""
prompt_few_shot = PromptTemplate(
    input_variables=["few_shot", "languages", "input_sent"],
    template=template_with_few_shot,
)

template_zero_shot = """
    Can you help translate from English to {languages}.
    English: {input_sent}
"""
prompt_zero_shot = PromptTemplate(
    input_variables=["languages", "input_sent"],
    template=template_zero_shot,
)

if few_shot:
    prompt = prompt_few_shot.format(few_shot=few_shot, languages=", ".join(languages), input_sent=input_sent)
else:
    prompt = prompt_zero_shot.format(languages=", ".join(languages), input_sent=input_sent)

def translate():
    try:
        llm = OpenAI(
            model_name=model,
            temperature=0,
            openai_api_key=openai_key,
        )
    except:
        st.write("Please enter a valid OpenAI API Key")
        return None
    
    output = llm(prompt)
    st.subheader("Translation")
    st.write(output)
    return output

if st.button("Fire Away!"):
    translate()

st.subheader("Final Prompt")
st.write(prompt)

