import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Rafakost Biometric API")

API_SECRET = os.getenv("API_SECRET", "")


def check_api_secret(x_api_secret: Optional[str]) -> None:
    if API_SECRET and x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API secret")


def save_upload_file(upload_file: UploadFile, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)

    extension = os.path.splitext(upload_file.filename or "")[1].lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        extension = ".jpg"

    file_path = os.path.join(folder, f"{uuid.uuid4()}{extension}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


@app.get("/")
def root():
    return {
        "service": "Rafakost Biometric API",
        "status": "online",
        "endpoints": [
            "/api/verify"
        ]
    }


@app.post("/api/verify")
async def verify_identity(
    ktp_photo: UploadFile = File(...),
    selfie_photo: UploadFile = File(...),
    x_api_secret: Optional[str] = Header(default=None),
):
    check_api_secret(x_api_secret)

    temp_dir = "/tmp/rafakost-biometric"
    ktp_path = None
    selfie_path = None

    try:
        ktp_path = save_upload_file(ktp_photo, temp_dir)
        selfie_path = save_upload_file(selfie_photo, temp_dir)

        from deepface import DeepFace

        face_result = DeepFace.verify(
            img1_path=ktp_path,
            img2_path=selfie_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True,
        )

        verified = bool(face_result.get("verified", False))
        distance = face_result.get("distance")
        threshold = face_result.get("threshold")

        gender_result = DeepFace.analyze(
            img_path=selfie_path,
            actions=["gender"],
            detector_backend="opencv",
            enforce_detection=True,
        )

        gender_data = gender_result[0] if isinstance(gender_result, list) else gender_result

        dominant_gender = gender_data.get("dominant_gender")
        gender_scores = gender_data.get("gender", {})

        normalized_gender = None
        gender_confidence = None

        if dominant_gender:
            if dominant_gender.lower() == "woman":
                normalized_gender = "female"
            elif dominant_gender.lower() == "man":
                normalized_gender = "male"
            else:
                normalized_gender = dominant_gender.lower()

            gender_confidence = gender_scores.get(dominant_gender)

        if verified:
            status = "approved"
            message = "Wajah KTP dan selfie cocok."
        else:
            status = "rejected"
            message = "Wajah KTP dan selfie tidak cocok."

        return JSONResponse({
            "success": True,
            "status": status,
            "message": message,
            "face_match": {
                "verified": verified,
                "distance": distance,
                "threshold": threshold,
                "raw": face_result,
            },
            "gender": {
                "detected": normalized_gender,
                "confidence": gender_confidence,
                "raw": gender_scores,
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "status": "manual_review",
                "message": "Verifikasi otomatis gagal, perlu review admin.",
                "error": str(e),
            }
        )

    finally:
        for path in [ktp_path, selfie_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass