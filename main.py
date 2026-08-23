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
async def enhance_doc_api(file: UploadFile = File(...), mode: str = "photo"):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if mode == "doc":
        # សម្រាប់សន្លឹកកិច្ចការ/តារាងអក្សរ៖ បង្កើន Contrast អក្សរ និងសម្អាតផ្ទៃស
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ដកស្រមោលចេញដោយប្រើ Morphological filter
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        enhanced = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    else:
        # សម្រាប់រូបថតមនុស្ស/Poster៖ រក្សាពណ៌ដើម ដំឡើង Contrast និង Sharpness ត្រឹមត្រូវ
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # បន្ថែម Sharpness ល្មមមិនឱ្យបែករូប
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    _, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
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
