from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import FileChatMessageHistory, ConversationSummaryMemory, ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chains import LLMChain
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

chat_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
    convert_system_message_to_human=True
)
# convert_system_message_to_human = True to enable summarizing since it is human like
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat_model
    # chat_memory=FileChatMessageHistory("messages.json")
)

# return_messages=True means returning not just strings but with its wrapper
# FileChatMessageHistory doesn't work well w/ the
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)
chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    memory=memory,
    verbose=True
)
while 1:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
