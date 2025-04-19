from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from processors.ContourCornersDetector import ContourCornersDetector
import io
from starlette.responses import Response

contour_page_extractor = ContourCornersDetector()

app = FastAPI()


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        processed_img = contour_page_extractor(img)

        _, buffer = cv2.imencode('.jpg', processed_img)
        bytes_io = io.BytesIO(buffer)

        return StreamingResponse(
            bytes_io,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)