import uvicorn
import socketio
import weaviate
from fastapi import FastAPI
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Fast API application
app = FastAPI()
# Socket io (sio) create a Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
# wrap with ASGI application
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

# Dictionary to store session data
sessions: Dict[str, List[Dict[str, str]]] = {}

# Weaviate Client Initialization
api_key = os.environ["OPENAI_API_KEY"]
weaviate_url = os.environ["WEAVIATE_URL"]

client = weaviate.WeaviateClient(
    url=weaviate_url,
    additional_headers={
        "X-OpenAI-API-Key": api_key
    }
)
# client = weaviate.connect_to_custom(
#     http_host="192.168.137.236",
#     http_port="8080",
#     http_secure=False,
#     grpc_host="192.168.137.236",
#     grpc_port="50051",
#     grpc_secure=False,
#     headers={
#         "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")  # Or any other inference API keys
#     }
# )

# Print {"Hello":"World"} on localhost:7777
@app.get("/")
def read_root():
    return {"Hello": "World"}


@sio.on("connect")
async def connect(sid, env):
    print("New Client Connected to This id :" + " " + str(sid))


@sio.on("disconnect")
async def disconnect(sid):
    print("Client Disconnected: " + " " + str(sid))


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

        # Perform Weaviate search
        query_result = client.query.get(
            "all_nov_jobs", ["title", "place", "description"]
        ).with_near_text(
            {"concepts": [data.get("message")]}
        ).with_additional(
            ["distance", "id"]
        ).with_limit(5).do()

        # Prepare the response message with search results
        results = query_result['data']['Get']['all_nov_jobs']
        response_text = "Top 5 search results:\n"
        for result in results:
            response_text += f"Title: {result['title']}\nPlace: {result['place']}\nDescription: {result['description']}\n\n"

        response_message = {
            "id": data.get("id") + "_response",
            "textResponse": response_text,
            "isUserMessage": False,
            "timestamp": data.get("timestamp"),
            "isComplete": True,
        }
        await sio.emit("textResponse", response_message, room=sid)
        sessions[session_id].append(response_message)

        print(f"Message from {sid} in session {session_id}: {data.get('message')}")

    else:
        print(f"No session ID provided by {sid}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6789, lifespan="on", reload=True)
