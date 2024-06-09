import uvicorn
from utils.config import client
from utils.document_processing import process_document
from utils.chat_setup import create_review_chain
from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import socketio
import os

app = FastAPI()

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

sessions = {}
review_chain = None

class ChatMessage(BaseModel):
    sessionId: str
    id: str
    message: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global review_chain
    # Initialize review_chain with a dummy docsearch
    dummy_docsearch = ...  # Replace with an appropriate docsearch object if available
    review_chain = create_review_chain(dummy_docsearch)

@app.get("/")
def read_root():
    return {"Hello": "World"}

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
    session_id = data.get("sessionId")
    if session_id not in sessions:
        sessions[session_id] = []
    await sio.emit("sessionInit", {"sessionId": session_id, "chatHistory": sessions[session_id]}, room=sid)

@sio.on("textMessage")
async def handle_chat_message(sid, data):
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
        
        # Process the message using the review_chain
        question = data.get("message")
        context = ""  # Replace with actual context if available
        response = review_chain({"context": context, "question": question})

        response_message = {
            "id": data.get("id") + "_response",
            "textResponse": response,  # Get the final output from review_chain
            "isUserMessage": False,
            "timestamp": data.get("timestamp"),
            "isComplete": True,
        }
        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)
    else:
        print(f"No session ID provided by {sid}")

@app.get("/query")
def query_weaviate(concepts: List[str] = Query(...)):
    class_name = "all_nov_jobs"
    results = client.query.get(
        class_name, ["title", "place", "description"]
    ).with_near_text(
        {"concepts": concepts}
    ).with_additional(
        ["distance", "id"]
    ).with_limit(1).do()
    return results

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    docsearch = process_document(contents, client)
    review_chain = create_review_chain(docsearch)
    return {"filename": file.filename, "status": "processed"}

@app.post("/chat/")
async def chat_endpoint(chat_message: ChatMessage):
    session_id = chat_message.sessionId
    if session_id not in sessions:
        sessions[session_id] = []

    received_message = {
        "id": chat_message.id,
        "message": chat_message.message,
        "isUserMessage": True,
        "timestamp": chat_message.timestamp,
    }
    sessions[session_id].append(received_message)
    
    # Process the message using the review_chain
    question = chat_message.message
    context = ""  # Replace with actual context if available
    response = review_chain({"context": context, "question": question})

    response_message = {
        "id": chat_message.id + "_response",
        "textResponse": response,  # Get the final output from review_chain
        "isUserMessage": False,
        "timestamp": chat_message.timestamp,
        "isComplete": True,
    }
    sessions[session_id].append(response_message)
    
    return response_message

if __name__ == "__main__":
    uvicorn.run("main:socket_app", host="192.168.137.236", port=6789, lifespan="on", reload=True)
