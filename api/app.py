from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import traceback
import base64
from AnalyseNowCast import visualize_result


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/event')
def event_query(request: Request, idx_id: str = ""):
  
  file_name = f"image_{int(datetime.now().timestamp())}.png"
  save_path = "/tmp/export/"
  file_path = os.path.join(save_path, file_name)
  try:
    fig,ax = plt.subplots(13,2,figsize=(5,20))
    fig.delaxes(ax[12][1])
    visualize_result([gan_model],x_test,y_test,int(idx_id),ax,labels=['cGAN+MAE'],save_path=file_path)
    with open(file_path, "rb") as file:
        image_bytes: bytes = base64.b64encode(file.read())
    return {"data": image_bytes}
  except Exception as e:
    message = traceback.format_exc()
    print(message)
    return "An internal error occurred"
