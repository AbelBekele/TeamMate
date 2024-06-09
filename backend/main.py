from fastapi import FastAPI, File, UploadFile, APIRouter, Query, HTTPException
from typing import List, Dict
import socketio
import uvicorn
from utils.config import settings
from utils.weaviate_client import query_weaviate
from utils.agent_setup import agent_executor, process_uploaded_file, update_retriever
from pydantic import BaseModel
import fitz  # PyMuPDF

class InputModel(BaseModel):
    input: str

# FastAPI application
app = FastAPI()

# SocketIO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
# Wrap with ASGI application
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Dictionary to store session data
sessions: Dict[str, List[Dict[str, str]]] = {}

@sio.on("connect")
async def connect(sid, env):
    print(f"New Client Connected: {sid}")

@sio.on("disconnect")
async def disconnect(sid):
    print(f"Client Disconnected: {sid}")

@sio.on("connectionInit")
async def handle_connection_init(sid):
    await sio.emit("connectionAck", room=sid)

@sio.on("sessionInit")
async def handle_session_init(sid, data):
    print(f"===> Session {sid} initialized")
    session_id = data.get("sessionId")
    if session_id not in sessions:
        sessions[session_id] = []
    print(f"**** Session {session_id} initialized for {sid} session data: {sessions[session_id]}")
    await sio.emit("sessionInit", {"sessionId": session_id, "chatHistory": sessions[session_id]}, room=sid)

# Handle incoming chat messages
@sio.on("textMessage")
async def handle_chat_message(sid, data):
    print(f"Message from {sid}: {data}")
    session_id = data.get("sessionId")
    if session_id:
        if session_id not in sessions:
            raise Exception(f"Session {session_id} not found")
        received_message = {
            "id": data.get("id"),
            "message": data.get("message"),
            "isUserMessage": True,
            "timestamp": data.get("timestamp"),
        }
        sessions[session_id].append(received_message)

        # Invoke the agent and get the response
        try:
            result = agent_executor.invoke({"input": data.get("message")})
            output_result = result.get("output")

            response_message = {
                "id": data.get("id") + "_response",
                "textResponse": output_result,
                "isUserMessage": False,
                "timestamp": data.get("timestamp"),
                "isComplete": True,
            }
        except Exception as e:
            response_message = {
                "id": data.get("id") + "_response",
                "textResponse": str(e),
                "isUserMessage": False,
                "timestamp": data.get("timestamp"),
                "isComplete": True,
            }

        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)

        print(f"Message from {sid} in session {session_id}: {data.get('message')}")

    else:
        print(f"No session ID provided by {sid}")

api_router = APIRouter()

# Endpoint to query Weaviate
@app.get("/query")
def query_weaviate_endpoint(concepts: List[str] = Query(...)):
    return query_weaviate(concepts)

# Endpoint to execute agent
@app.post("/execute_agent")
async def execute_agent(input_data: InputModel):
    try:
        result = agent_executor.invoke({"input": input_data.input})
        output_result = result.get("output")
        return {"output": output_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for PDF file upload
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the PDF content
        pdf_content = await file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        # Process the text extracted from the PDF
        update_retriever(text)
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run("main:socket_app", host=settings.HOST, port=settings.PORT, lifespan="on", reload=True)
