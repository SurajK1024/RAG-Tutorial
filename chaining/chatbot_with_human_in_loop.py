# from langchain_ollama.chat_models import ChatOllama
# from langgraph.graph import StateGraph, START, END
# from langgraph.types import Command
# from langgraph.checkpoint.memory import InMemorySaver

# from typing import TypedDict

# class ChatbotState(TypedDict):
#     message: str = "Hello, I need a help."
#     history: list[str]
#     user: str
#     context: str

# llm = ChatOllama(model="qwen3:1.7b")

# def get_human_feedback(state: ChatbotState) -> str:
#     feedback = input("Please provide your response:")
#     state["message"] = feedback
#     return state

# def decide_node(state: ChatbotState) -> str:
#     if state["message"] == "quit":
#         return "END"
#     return "talk_to_llm"

# def talk_to_llm(state: ChatbotState) -> str:
#     response = llm.invoke(state["message"])
#     print(response)
#     return response

# checkpointer = InMemorySaver()

# graph = StateGraph(ChatbotState)

# graph.add_node("human_feedback", get_human_feedback, )
# graph.add_node("talk_to_llm", talk_to_llm)

# graph.add_edge(START, "talk_to_llm")
# graph.add_edge("talk_to_llm", "human_feedback")
# graph.add_conditional_edges(
#     "human_feedback",
#     decide_node,
#     {
#         "END" : END,
#         "talk_to_llm": "talk_to_llm"
#     }
# )
# workflow = graph.compile(interrupt_before=["human_feedback"], checkpointer=checkpointer)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class State(TypedDict):
    message: str

def get_human(state):
    msg = interrupt("What is your message?")
    return {"message": msg}

def decide(state):
    if state["message"].lower() == "quit":
        return END
    return "bot"

def bot(state):
    print(f"Bot: You said '{state['message']}'")
    return state

graph = StateGraph(State)

graph.add_node("get_human", get_human)
graph.add_node("bot", bot)

graph.add_edge(START, "get_human")
graph.add_conditional_edges(
    "get_human",
    decide,
    {
        "bot": "bot",
        END: END,
    },
)
graph.add_edge("bot", "get_human")

workflow = graph.compile()

for event in workflow.stream({"message":"Hello!"}, config={"configurable": {"thread_id":1}}, stream_mode="values"):
    print(f"Event: {event}")
    interrupt = event.get("__interrupt__", [])

    if interrupt:
        print(f"Interruptted, {interrupt[0].value}")

        decision = input().strip().lower()
        if decision == "quit":
            workflow.send_interrupt_response(END)
        else:
            workflow.send_interrupt_response(decision)