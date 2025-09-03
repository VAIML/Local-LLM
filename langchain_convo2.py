import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import SimpleMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_ollama import OllamaLLM


llm_model = OllamaLLM(model="mistral:7b-instruct-v0.3-q4_0", temperature=0.7, max_tokens=1000)

template = """ 
You are a children's book author, writing a story for kids aged 12-15.
Your story should be engaging, imaginative, and suitable for young readers. Please come up with a creative story lullaby  based on 
the location {location} and the main character {main_character} in language {language}


Story:
"""
prompt = PromptTemplate(
    input_variables=["location", "main_character", "language"],
    template=template,
    verbose=True
    
)

chain = LLMChain(
    llm=llm_model,
    prompt=prompt,
    verbose=True 
    )


print(chain.run(
    location="Mumbai",
    main_character="Aaditya",
    language="German"
))