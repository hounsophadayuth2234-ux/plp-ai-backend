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

# Health Check Endpoint សម្រាប់ Render
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "PLP AI Engine is running!"}

# ១. API កាត់ Background
@app.post("/api/remove-bg")
async def remove_background_api(file: UploadFile = File(...)):
    contents = await file.read()
    output = remove(contents)
    return Response(content=output, media_type="image/png")

# ២. API បង្កើនភាពច្បាស់ (រក្សាពណ៌ដើម និង Sharpen)
@app.post("/api/enhance-doc")
async def enhance_doc_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # ដំឡើង Contrast លើ LAB Color Space ដើម្បីរក្សាពណ៌ដើម
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # បន្ថែម Sharpening Filter ឱ្យអក្សរ និងរូបភាពច្បាស់
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    _, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# ៣. API បំប្លែង PDF ទៅ Word
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

# 🟢 បន្ថែមផ្នែកខាងក្រោមនេះដើម្បី Binding Port របស់ Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
