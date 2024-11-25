from fastapi import FastAPI, HTTPException
from fastapi import status
from fastapi.responses import JSONResponse
from models import ConversationRequest
from utils import gen_response

BASE_ENDPOINT = 'chatbot/api/v1'

app = FastAPI()

@app.post(f"/response")
async def generate_response(request: ConversationRequest):
    if len(request.messages) < 1:
        raise HTTPException(status_code=400, detail={"status_code": 500, "error": "The 'messages' list must contain at least one item."})
    
    try:
        assistant_message = gen_response(request.messages)
        return {
            "status_code": 200,
            "assistant_message": assistant_message
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail={"status_code": 500, "error": str(e)})