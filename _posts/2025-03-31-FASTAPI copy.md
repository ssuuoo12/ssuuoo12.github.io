---
layout: post
title: FASTAPI(챗봇기반)
date: 2025-03-31 23:18 +0800
last_modified_at: 2025-03-31 23:2023:20 +0800
tags: [chatbot, python, FastAPI]
toc:  true
---
**FASTAPI** 챗봇 기반 예제
{: .message }

- 1. main.py : 
{% highlight text %}
- 설명: FastAPI 애플리케이션의 진입점(entry point). 애플리케이션을 초기화하고 라우터를 포함하며, 기본 엔드포인트(/)를 제공
- 주요 역할: 서버 실행(Uvicorn), 라우터 연결, API 상태 확인용 루트 엔드포인트 제공.
- 내용: FastAPI 객체 생성, chat_router 포함, 기본 GET / 엔드포인트.

- 쉽게 말하면: 이 파일은 프로그램의 "시작 버튼" 같은 거예요.
- 하는 일: FastAPI 앱을 만들고, 서버를 실행하며, 다른 파일(라우터)을 연결해요. 또 기본 페이지(/)에 접속하면 "잘 돼요!" 같은 간단한 확인 메시지를 보여줘요.
{% endhighlight %}

2. app/routers/chat_router.py
{% highlight text %}
- 설명: 챗봇 관련 API 엔드포인트를 정의하는 라우터 파일
- 주요 역할: /chat/message POST 엔드포인트를 통해 사용자 메시지를 받아 OllamaService로 전달하고 응답을 반환
- 내용: APIRouter를 사용해 /chat 접두사 설정, ChatRequest 스키마로 요청 처리, 응답 반환
- 쉽게 말하면: 챗봇과 대화하는 "창구"를 만드는 파일이에요.
- 하는 일: 사용자가 "/chat/message"로 메시지를 보내면 그걸 받아서 챗봇 서비스로 넘기고, 답을 돌려줘요.
{% endhighlight %}


3. app/services/ollama_service.py
{% highlight text %}
- 설명: Ollama API와 통신하는 서비스 로직을 담당
- 주요 역할: Gemma2 모델에 사용자 입력(prompt)을 전달하고 응답을 받아오는 역할
- 내용: OllamaService 클래스 정의, 환경 변수로 API URL 설정, HTTP 요청으로 Ollama 호출 및 에러 처리
- 쉽게 말하면: 챗봇의 "두뇌"와 연결해주는 중간 다리예요.
- 하는 일: 사용자가 보낸 메시지를 Gemma2라는 AI 모델에 전달하고, 그 답을 받아오는 역할을 해요. 문제가 생기면 에러도 처리해줘요.

{% endhighlight %}
4. app/schemas/chat_schema.py
{% highlight text %}
- 설명: API 요청 및 응답 데이터의 구조를 정의하는 Pydantic 모델 파일
- 주요 역할: 요청(ChatRequest)과 응답(ChatResponse) 데이터의 형식 검증
- 내용: BaseModel을 상속받아 message 필드(요청)와 response 필드(응답) 정의
- 쉽게 말하면: 메시지와 답의 "형식"을 정해주는 규칙 책이에요.
- 하는 일: 사용자가 보낸 메시지와 챗봇이 줄 답이 어떤 모양이어야 하는지 미리 정의해서 데이터가 엉키지 않게 해요.
{% endhighlight %}

## 전체 흐름
{% highlight text %}
- main.py가 앱을 시작하고 →
- chat_router.py가 사용자의 메시지를 받고 →
- ollama_service.py가 AI한테 물어본 뒤 →
- chat_schema.py로 깔끔하게 정리된 답을 돌려줘요.
- python main.py 실행 ==> http://0.0.0.0:8000 메시지 확인(모델이 잘 실행되는지 확인)
{% endhighlight %}


## main.py

{% highlight js %}
// 사전설치 : pip install fastapi uvicorn pandas numpy scikit-learn tensorflow 
// 사전설치 : pip install sqlalchemy pymysql python-dotenv pydantic requests
// uvicorn: FastAPI 앱을 실행시키는 서버
import uvicorn

// FastAPI 클래스 및 모듈 가져오기
from fastapi import FastAPI

// app하위의 routers폴더에서 불러오기
from app.routers import chat_router2

// FastAPI 애플리케이션 인스턴스 생성
// title과 description은 docs 페이지에 표시됨
app = FastAPI(
    title="Chatbot API", description="Hugging Face 기반 요약 챗봇 서비스"
)

// /chat/message 같은 라우트를 app에 등록
// chat_router.router는 APIRouter 인스턴스임
app.include_router(chat_router2.router)

// 루트 경로를 (/)로 정의 예) localhost:8000/
@app.get("/")
async def root():
    return {"message": "Chatbot API Running"}

// 이 파일이 메인으로 실행될 경우 서버 실행
// reload=True는 코드 변경 시 자동으로 서버를 재시작해줌 (개발용)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
{% endhighlight %}

## chat_router2.py
{% highlight js %}
// app/routers/chat_router.py
from fastapi import APIRouter, HTTPException
// APIRouter: FastAPI에서 라우팅을 모듈 단위로 관리할 수 있게 해주기 
// 즉, main.py에 모든 경로를 정의하지 않고, 따로 라우팅 파일(chat_router.py)로 분리할 수 있게 도와줌
// HTTPException: 예외 발생 시 HTTP 상태 코드와 메시지를 클라이언트에 보낼 수 있도록 해준다

from pydantic import BaseModel
// FastAPI는 데이터 검증을 위해 Pydantic이라는 라이브러리를 사용
// BaseModel을 상속받은 클래스를 만들면 요청(body)로 들어온 데이터를 자동으로 검증하고 처리해줌

from app.services.Hugface_service import HuggingFaceSummaryService
// app\services\Hugface_service.py의 HuggingFaceSummaryService를 임포트하기


router = APIRouter(prefix="/chat", tags=["Chatbot"])
// APIRouter() 인스턴스를 생성해서 이 파일에서 사용할 라우터를 만들기
// prefix="/chat" → 이 파일 안에 정의되는 모든 경로는 기본적으로 /chat으로 시작하기
// tags=["Chatbot"] → 자동 문서화(/docs)에서 이 라우터에 붙는 태그

class ChatRequest(BaseModel):
    message: str
// 클라이언트가 보낼 JSON은 반드시 다음처럼 생겨야 함


@router.post("/message")
async def get_chat_response(request: ChatRequest):
    # 이 부분은 POST /chat/message 요청이 들어올 때 실행되는 API 핸들러
    # 요청 본문은 ChatRequest로 받고, 그 안에 들어있는 message를 처리함
    try:
        hugface_service = HuggingFaceSummaryService() # 인스턴스 생성,이 인스턴스는 요약 모델(T5 등)을 로드하고, 입력을 요약하는 역할
        response = hugface_service.generate_summary(request.message) 
        # request.message → 유저가 보낸 텍스트
        # generate_summary() → 이 텍스트를 요약해서 결과 문자열을 반환
        return {"response": response} # json형식으로 반환, 예) {"response": "요약된 문장"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    // 만약 모델이 없거나, 에러가 나면 500 서버 에러로 예외를 클라이언트에게 반환
    // 출력 예시) {"detail": "모델을 로드할 수 없습니다."}
{% endhighlight %}

## Hugface_service.py
{% highlight js %}
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HuggingFaceSummaryService:
    def __init__(self, model_name="lcw99/t5-base-korean-text-summary"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # 토크나이저와 모델을 불러오기
        self.prefix = "summarize: "
        self.max_input_length = 512
        self.max_output_length = 100
        # 입력 문장은 512 토큰까지 자르고,
        # 출력은 최대 100토큰까지 생성하게 제한

    def generate_summary(self, text: str):
        prompt = self.prefix + text
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        # return_tensors="pt" → PyTorch 텐서로 반환
        # truncation=True → 길면 자르고
        outputs = self.model.generate(**inputs, num_beams=4, do_sample=True, min_length=10, max_length=self.max_output_length)
        # **inputs : 토크나이저로부터 얻은 입력 텐서 (input_ids 등)
        # num_beams=4 → Beam search 사용 (더 나은 결과 생성)
        # do_sample=True → 약간의 랜덤성 부여 (다양한 문장 가능)
        # min_length=10 → 최소 생성 길이 제한
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # generate()는 여러 문장을 생성할 수 있기 때문에 결과가 리스트로 나와요.
        # [0]은 그 중 첫 번째 결과를 출력
        return summary
{% endhighlight %}

## chat_schema.py
{% highlight js %}
from pydantic import BaseModel
// BaseModel을 상속한 클래스를 만들면, 그것이 하나의 데이터 스키마

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
{% endhighlight %}


