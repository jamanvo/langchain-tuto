import operator
from typing import Annotated, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

model = "gpt-4.1-nano"


# Schemas
class Target(BaseModel):
    name: str = Field(..., description="대상자 이름")
    phone_number: str = Field(..., description="대상자 전화번호")
    message: str | None = Field(None, description="문자 메시지 내용")


class StateModel(BaseModel):
    messages: Annotated[list, operator.add]
    tool_name: Literal["call_phone", "send_message"] | None = None
    next_step: str | None = None
    target: Target | None = None


class CallPhoneModel(BaseModel):
    name: str = Field(..., description="이름")
    phone_number: str = Field(..., description="전화번호")


class SendMessageModel(BaseModel):
    name: str = Field(..., description="이름")
    phone_number: str = Field(..., description="전화번호")
    message: str = Field(..., description="보낼 메시지 내용")


class AgentResult(BaseModel):
    response: str = Field(..., description="에이전트 실행 결과")
    next_step: str = Field(..., description="다음 단계 목적지")
    name: str = Field(..., description="대상자 이름")
    phone_number: str = Field(..., description="대상자 전화번호")
    message: str | None = Field(None, description="문자 메시지 내용")


class DefaultResponse(BaseModel):
    response: str = Field(..., description="에이전트 실행 결과")


# Tools
@tool(description="전화를 겁니다", args_schema=CallPhoneModel)
def call_phone(name: str, phone_number: str) -> str:
    return f"{name}에게 전화를 겁니다\n" f"전화번호: {phone_number}"


@tool(description="문자를 보냅니다", args_schema=SendMessageModel)
def send_message(name: str, phone_number: str, message: str) -> str:
    return f"{name}에게 문자를 보냅니다...\n" f"전화번호: {phone_number}\n" f"메시지 내용: {message}"


# Runnables
def main_runnable():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    1. 전화를 걸거나 문자를 보내야 한다면 'call_phone' 또는 'send_message' 를 tool_name에 반환하고, 필요한 데이터를 설정해.
                    2. 문자를 보내려는 경우에는 요청 내용에서 보낼 문자 내용을 찾아서 message에 설정해.
                    3. 이름을 찾을 수 없거나 번확 없는 등 처리할 수 없는 요청이라면 next_step을 'error'로 설정해.
                    4. 처리 후 결과는 response에 종합해서 기록해.

                    네가 처리할 수 있는 연락처 정보:
                    name | phone_number
                    차태식 | 01011112222
                    오대수 | 12300991123
                    
                    # 문자 내용 예시
                    - 만석이한테 어디냐고 물어봐 -> message=어디냐
                    - 동수한테 저녁먹자고 문자 보내줘 -> message=저녁먹자
                """,
            ),
            ("human", "사용자의 요청입니다.\n{request}"),
        ]
    )
    agent = ChatOpenAI(model=model).with_structured_output(AgentResult)

    return prompt | agent


def error_runnable():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            너는 전달받은 에러를 처리하는 에이전트야.
            요청을 처리할 수 없는 이유를 최대한 알기 쉽게 설명해줘.
            
            실행 결과는 반드시 다음의 JSON 형식이 맞추도록 해.
            {{
                'response': 처리 결과
            }}
        """,
            ),
            ("ai", "오류 내용입니다. {input}"),
        ]
    )

    agent = ChatOpenAI(model=model).with_structured_output(DefaultResponse)

    return prompt | agent


def end_runnable():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        너는 작업한 내용을 유저에게 전달하는 에이전트야.
        간결하게 실행 결과를 안내하되 사람이 보기 쉽도록 정보를 재구성해줘.
        
        실행 결과는 반드시 다음의 JSON 형식이 맞추도록 해.
        {{
            'response': 처리 결과
        }}
    """,
            ),
            ("ai", "실행 결과입니다. {input}"),
        ]
    )

    agent = ChatOpenAI(model=model).with_structured_output(DefaultResponse)

    return prompt | agent


def main_node(state: StateModel) -> dict:
    print("run main node")
    main_result: AgentResult = main_runnable().invoke({"request": state.messages[-1]})

    return {
        "messages": [("ai", main_result.response)],
        "next_step": main_result.next_step,
        "target": {
            "name": main_result.name,
            "phone_number": main_result.phone_number,
            "message": main_result.message,
        },
    }


def error_node(state: StateModel) -> dict:
    print("run error node")
    _result: DefaultResponse = error_runnable().invoke({"input": state.messages[-1]})

    return {"messages": [("ai", _result.response)], "next_step": "end"}


def end_node(state: StateModel) -> dict:
    print("run end node")
    _result: DefaultResponse = end_runnable().invoke({"input": state.messages[-1]})

    return {"messages": [("ai", _result.response)]}


def call_phone_tool(state: StateModel) -> dict:
    print("run call_phone_tool node")
    _result = call_phone.invoke({"name": state.target.name, "phone_number": state.target.phone_number})

    return {"messages": [("ai", _result)]}


def send_message_tool(state: StateModel) -> dict:
    print("run send_message_tool node")

    _result = send_message.invoke(
        {"name": state.target.name, "phone_number": state.target.phone_number, "message": state.target.message}
    )

    return {"messages": [("ai", _result)]}


def condition_edge(state: StateModel) -> str:
    if state.tool_name:
        return "call_phone"

    return state.next_step


def main(message: str) -> tuple:
    workflow = StateGraph(StateModel)

    # 노드 등록
    workflow.add_node("main", main_node)
    workflow.add_node("error", error_node)
    workflow.add_node("end", end_node)
    workflow.add_node("call_phone", call_phone_tool)
    workflow.add_node("send_message", send_message_tool)

    # 엣지 저장
    workflow.add_conditional_edges(
        "main",
        condition_edge,
        {
            "end": "end",
            "error": "error",
            "call_phone": "call_phone",
            "send_message": "send_message",
        },
    )
    workflow.add_edge("call_phone", "end")
    workflow.add_edge("send_message", "end")

    # 시작점 지정
    workflow.set_entry_point("main")

    graph_app = workflow.compile()

    response = graph_app.invoke({"messages": [("user", message)]})

    return response["messages"][-1]


if __name__ == "__main__":
    result = main("태식이한테 어디냐고 문자보내.")
    print(result)
