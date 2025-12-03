from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import uuid
import os

app = FastAPI()

# Dummy function – replace with your actual PDF generator function
def generate_pdf_from_files(file_paths, output_path):
    with open(output_path, "wb") as f:
        f.write(b"%PDF-FAKE-FILE%")  # Placeholder PDF bytes


@app.post("/generate-pdf")
async def generate_pdf(files: list[UploadFile] = File(...)):
    temp_dir = f"temp_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    saved_files = []
    for f in files:
        path = os.path.join(temp_dir, f.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append(path)

    output_pdf = os.path.join(temp_dir, "generated.pdf")
    generate_pdf_from_files(saved_files, output_pdf)

    return FileResponse(
        output_pdf,
        media_type="application/pdf",
        filename="report.pdf",
    )


@app.get("/")
async def root():
    return {"message": "PDF Generator API running"}
