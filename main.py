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

# ១. API កាត់ Background
@app.post("/api/remove-bg")
async def remove_background_api(file: UploadFile = File(...)):
    contents = await file.read()
    output = remove(contents)
    return Response(content=output, media_type="image/png")

# ២. API បង្កើនភាពច្បាស់ (បំបែក 2 Mode ដាច់ពីគ្នា)
@app.post("/api/enhance-doc")
async def enhance_doc_api(file: UploadFile = File(...), mode: str = "photo"):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Response(status_code=400, content="Invalid image")

    if mode == "doc":
        # 🟢 MODE ឯកសារ/សន្លឹកកិច្ចការ៖ លុបស្រមោលក្រដាស ប៉ុន្តែរក្សាពណ៌រូបគំនូរ (ផ្លែឈើ/រូបភាព) ឱ្យនៅស្អាត 100%
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # គណនាដកស្រមោលចេញពី Lightness channel ដោយមិនប្រើ Adaptive Threshold
        dilated = cv2.dilate(l, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(l, bg)
        l_norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # ផ្គុំ Color Channel (a, b) ចូលវិញដើម្បីរក្សាពណ៌ដើម
        enhanced_lab = cv2.merge((l_norm, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # ដំឡើងភាពច្បាស់លើអក្សរ និងខ្សែបន្ទាត់
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1.2, gaussian, -0.2, 0)

    else:
        # 🟢 MODE រូបថតមនុស្ស/វត្ថុ/Poster៖ បន្ថែម Sharpness HD រក្សាពណ៌ធម្មជាតិ និងមិនខ្មៅខ្លោច
        smooth = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
        
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
