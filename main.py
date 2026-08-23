from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove
import cv2
import numpy as np
import os
import tempfile
from pdf2docx import Converter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "PLP AI Engine is running!"}

# 1. AI កាត់ Background (ប្រើប្រាស់ AI Model u2net ផ្ទាល់ រក្សារូបមនុស្សស្អាត 100%)
@app.post("/api/remove-bg")
async def remove_background_api(file: UploadFile = File(...)):
    contents = await file.read()
    output = remove(contents)
    return Response(content=output, media_type="image/png")

# 2. បង្កើនភាពច្បាស់ (រក្សាពណ៌ធម្មជាតិ និងដំឡើង Sharpness ស្រាលៗ)
@app.post("/api/enhance-doc")
async def enhance_doc_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ប្រើ Unsharp Masking ដើម្បីបង្កើនភាពច្បាស់ដោយមិនបំផ្លាញពណ៌
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    enhanced = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
    _, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# 3. បំប្លែង PDF ទៅ Word
@app.post("/api/pdf-to-word")
async def pdf_to_word_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_content = await file.read()
        tmp_pdf.write(pdf_content)
        tmp_pdf_path = tmp_pdf.name

    tmp_docx_path = tmp_pdf_path.replace(".pdf", ".docx")

    try:
        cv = Converter(tmp_pdf_path)
        cv.convert(tmp_docx_path, start=0, end=None)
        cv.close()

        with open(tmp_docx_path, "rb") as docx_file:
            docx_content = docx_file.read()

        return Response(
            content=docx_content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    finally:
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
        if os.path.exists(tmp_docx_path):
            os.remove(tmp_docx_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
