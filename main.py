import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Initialize the LLM model
GROQ_LLM = ChatGroq(
    model='llama3-70b-8192'
)

# Define the prompt template for categorizing text
prompt = PromptTemplate(
    template="""system
    You are a Text Categorizer Agent. You are a master at understanding what my friend is saying and are able to categorize it in a useful way. The texts might be in English or Roman Urdu.
    For context, 'milo' is my nickname.

    user
    Conduct a comprehensive analysis of the text provided and categorize it into 'compliment' or 'off_topic':
        compliment - used when my friend is complimenting me on something. This can be them calling me cute, smart, hot, etc. Supportive or encouraging texts will not be regarded as a compliment.
        off_topic - used when it doesn't relate to the compliment description.

    Output a single category only from the types ('compliment', 'off_topic').
    Example:
    'compliment'

    Output a single category only. Do not output anything else.

    TEXT CONTENT:\n\n {text} \n\n
    
    assistant
    """,
    input_variables=["text"],
)

# Create the text categorizer using the prompt template and the LLM
text_category_generator = prompt | GROQ_LLM | StrOutputParser()

# Define the GraphState class
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        text: initial text
        text_category: text category
        return_text: return text
    """
    text: str
    text_category: str
    return_text: str

# Define the nodes for the state graph
def categorize_text(state):
    initial_text = state['text']
    text_category = text_category_generator.invoke({"text": initial_text})
    state['text_category'] = text_category
    return state

def reply_to_text(state):
    print("---replying to text---")
    text_cat = state["text_category"]
    if text_cat == 'compliment':
        state['return_text'] = "nu youuuu!!"
    else:
        state['return_text'] = "noted."
    return state

def state_printer(state):
    """Print the state"""
    print("---STATE PRINTER---")
    print(f"Initial text: {state['text']}")
    print(f"Text Category: {state['text_category']}")
    print(f"Return Text: {state.get('return_text', '')}")
    return state

# Create the workflow using StateGraph
workflow = StateGraph(GraphState)

workflow.add_node("categorize_text", categorize_text)  # categorize text
workflow.add_node("reply_to_text", reply_to_text)  # reply to text
workflow.add_node("state_printer", state_printer)  # print state

workflow.set_entry_point("categorize_text")

# Define the conditions for edges
def categorize_condition(state):
    text_cat = state['text_category']
    if text_cat == 'compliment':
        return "reply_to_text"
    else:
        return "reply_to_text"
# Add conditional edges
workflow.add_edge("categorize_text", 'reply_to_text')
workflow.add_edge("reply_to_text", "state_printer")
workflow.add_edge("state_printer", END)

# Compile the workflow
app = workflow.compile()

# Test the workflow
input_text = "tum aik kutte ho"
initial_state = {"text": input_text}
out = app.invoke(initial_state)

# Print the output
print(out)
