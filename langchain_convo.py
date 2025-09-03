import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import SimpleMemory
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm_model = OllamaLLM(model="mistral:7b-instruct-v0.3-q4_0", temperature=0.7, max_tokens=2000)
llm1 = OllamaLLM(model="llama3.2:3b-instruct-q4_0", temperature=0.98, max_tokens=2000)
template = """ 
You are a children's book author, writing a story for kids aged 12-15.
Your story should be engaging, imaginative, and suitable for young readers. Please come up with a creative story lullaby based on 
the location {location} and the main character {main_character}

Story:
"""
prompt = PromptTemplate(
    input_variables=["location", "main_character"],
    template=template
)

# Replace LLMChain with RunnableSequence
chain_story = prompt | llm_model

# Use invoke instead of run
story_response = chain_story.invoke({
    "location": "Mumbai",
    "main_character": "Aaditya"
})
print(story_response)

#====Sequential Chain=============

template_update = """ 
Translate the {story} into {language}. Make sure the language is simple and fun

Translation:
"""
prompt_translate = PromptTemplate(
    input_variables=["story", "language"],
    template=template_update
)

# Replace LLMChain with RunnableSequence
chain_translate = prompt_translate | llm1

# Custom RunnableSequence for SequentialChain
from langchain_core.runnables import RunnableLambda

def extract_story(inputs):
    return {"story": inputs["story"], "language": inputs["language"]}

overall_chain = RunnableSequence(
    chain_story | (lambda x: {"story": x, **x}) | RunnableLambda(extract_story) | chain_translate
)

response = overall_chain.invoke({
    "location": "Mumbai",
    "main_character": "Aaditya",
    "language": "Hindi"
})
print(llm_model)
print(f"English Story: {response['story']}\n\n")
print(f"Translation: {response}\n\n")