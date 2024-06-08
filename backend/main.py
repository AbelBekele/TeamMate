from fastapi import FastAPI, Query
from typing import Dict, List
import socketio
import uvicorn
import weaviate
import os
import pandas as pd
import numpy as np

# FastAPI application
app = FastAPI()

# SocketIO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
# Wrap with ASGI application
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Dictionary to store session data
sessions: Dict[str, List[Dict[str, str]]] = {}

# Weaviate client setup
api_key = os.environ["OPENAI_API_KEY"]
weaviate_url = os.environ["WEAVIATE_URL"]

client = weaviate.Client(
    url=weaviate_url,
    additional_headers={
        "X-OpenAI-API-Key": api_key
    }
)

def jprint(json_in):
    import json
    print(json.dumps(json_in, indent=2))


# Print {"Hello": "World"} on localhost:7777
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
        response_message = {
            "id": data.get("id") + "_response",
            "textResponse": data.get("message"),
            "isUserMessage": False,
            "timestamp": data.get("timestamp"),
            "isComplete": True,
        }
        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)

        print(f"Message from {sid} in session {session_id}: {data.get('message')}")

    else:
        print(f"No session ID provided by {sid}")

# Endpoint to query Weaviate
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

if __name__ == "__main__":
    uvicorn.run("main:socket_app", host="192.168.137.236", port=6789, lifespan="on", reload=True)
