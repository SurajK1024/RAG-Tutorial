from typing import TypedDict, Literal
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage

from pydantic import BaseModel

from dotenv import load_dotenv
import os

llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0.8
    )

QuestionSystemPrompt = SystemMessage(content="""You are personality question generator. Your task is to ask the user with questions to determine their personality type. This personality test is to make the users experience more personalized when thy use the travel agent. You can ask any question which will help the travel agent to understand and create itenrary for the user. You can ask about their preferences, interests, and travel habits. You can also ask about their past travel experiences and what they liked or disliked about them. The more information you gather, the better the travel agent will be able to create a personalized itinerary for the user. Please ask one question at a time and wait for the user's response before asking the next question.""")

SummarizeSystemPrompt = SystemMessage(content="""Your task is to understand the questions answering provided and summarize the personality of the user.""")

class ResponseState(TypedDict):
    AIQuestion: str
    HumanAnswer: str
    ChatData: list[BaseMessage] = []
    PersonalitySummary: str
    shouldQuestion: bool = True

class NextQuestionSchema(BaseModel):
    nextQuestion: str

class ShouldContinueSchema(BaseModel):
    shouldContinue: bool

def getHumanResponse(state: ResponseState) -> ResponseState:
    print("#"*100)
    print(f"Bot 🤖: {state['AIQuestion']}")
    humanResponse = input("Your reply 🙋‍♂️:")
    while not humanResponse:
        print("Please provide a response.")
        humanResponse = input("Your reply:")

    humanResponse = HumanMessage(content=humanResponse)
    state["HumanAnswer"] = humanResponse.content
    state["ChatData"].append(humanResponse)

    return state

def getAIResponse(state: ResponseState) -> ResponseState:
    ai_response: NextQuestionSchema = llm.with_structured_output(NextQuestionSchema).invoke([QuestionSystemPrompt] + state["ChatData"])

    ai_response = AIMessage(content=ai_response.nextQuestion)
    state["AIQuestion"] = ai_response.content
    state["ChatData"].append(ai_response)
    return state

def summarizePersonality(state: ResponseState) -> ResponseState:
    summary = llm.invoke(
        [SummarizeSystemPrompt] + state["ChatData"]
    )
    return {"PersonalitySummary": summary.content}

def shouldContinueSurvey(state: ResponseState) -> str:

    response: ShouldContinueSchema = llm.with_structured_output(ShouldContinueSchema).invoke(
        ["Do you have enough data for summarizing the personality of person?"] + state["ChatData"]
    )

    if response.shouldContinue:
        return "askQuestion"
    return "summarize"

def Agent():
    workflow = StateGraph(ResponseState)

    workflow.add_node("HumanResponseNode", getHumanResponse)
    workflow.add_node("AIResponseNode", getAIResponse)
    workflow.add_node("SummarizeNode", summarizePersonality)

    workflow.add_edge(START, "AIResponseNode")
    workflow.add_edge("AIResponseNode", "HumanResponseNode")
    workflow.add_conditional_edges(
        "HumanResponseNode",
        shouldContinueSurvey,
        {
            "askQuestion": "AIResponseNode",
            "summarize": "SummarizeNode"
        }
    )
    workflow.add_edge("SummarizeNode", END)
    return workflow.compile()

if __name__ == "__main__":
    inital_state: ResponseState = {
        "AIQuestion": "",
        "HumanAnswer": "",
        "ChatData": [],
        "PersonalitySummary": "",
        "shouldQuestion": True
    }

    personalityAgent = Agent()

    final_response = personalityAgent.invoke(inital_state)

    print(final_response)