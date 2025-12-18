from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os, uuid, time, math

app = FastAPI(title="Smart City API - Starter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKETS = []

class TicketCreate(BaseModel):
    title: str
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    tags: Optional[List[str]] = []

@app.get('/')
def read_root():
    return {"message": "Smart City API - running"}

@app.post('/api/v1/tickets')
async def create_ticket(ticket: TicketCreate):
    ticket_id = str(uuid.uuid4())
    entry = ticket.dict()
    entry.update({"id": ticket_id, "status": "open"})
    TICKETS.append(entry)
    return {"ok": True, "ticket": entry}

@app.get('/api/v1/tickets')
def list_tickets():
    return {"tickets": TICKETS}

@app.post('/api/v1/cv/analyze')
async def analyze_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(status_code=400, detail='Invalid image type')
    out_dir = 'uploaded_images'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uuid.uuid4()}{ext}")
    with open(out_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    result = {"detections": [{"label": "pothole", "confidence": 0.87, "bbox": [50, 30, 200, 150]}]}
    return {"ok": True, "file": out_path, "result": result}

@app.post('/api/v1/chat/query')
async def chat_query(payload: dict):
    message = payload.get('message', '').lower() if isinstance(payload, dict) else ''
    if 'report' in message or 'pothole' in message:
        return {"reply": "To report a pothole, please upload a photo via /api/v1/cv/analyze or create a ticket via /api/v1/tickets"}
    if 'traffic' in message:
        return {"reply": "Traffic forecast: moderate for next hour. (demo)"}
    return {"reply": "Sorry, I am a demo assistant. Try asking about traffic or reporting issues."}

@app.get('/api/v1/models/traffic/predict')
def predict_traffic(horizon: int = 60):
    now = int(time.time())
    preds = []
    for i in range(0, horizon, 10):
        preds.append({"ts": now + i*60, "congestion": max(0.0, 0.5 + 0.4*math.sin(i/10.0))})
    return {"horizon_min": horizon, "predictions": preds}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
