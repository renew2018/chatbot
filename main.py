# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import logging
from uuid import uuid4

# Import functions (ensure these exist in their files)
from extract_pdf import extract_structured_data
from embed_store import embed_and_store

app = FastAPI(title="Renew Chart Bot API", version="1.0")

# Create logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Endpoint to upload and process PDF
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        os.makedirs("data", exist_ok=True)
        file_id = uuid4().hex
        filepath = f"data/{file_id}_{file.filename}"

        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Received file: {filepath}")

        # Extract structured content from PDF
        parsed_data = extract_structured_data(filepath)
        if not parsed_data:
            raise ValueError("PDF extraction returned empty data.")

        # Embed and store in vector DB
        count = embed_and_store(parsed_data)

        return JSONResponse(
            content={
                "message": f"Successfully embedded {count} entries.",
                "filename": file.filename,
                "id": file_id,
                "status": "success"
            },
            status_code=200
        )

    except Exception as e:
        logger.exception("Failed to process the PDF.")
        raise HTTPException(status_code=500, detail=str(e))

