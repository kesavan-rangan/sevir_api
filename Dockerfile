WORKDIR /api_server
COPY . .
RUN pip install -r requirements.txt
RUN uvicorn api.app:app --reload
EXPOSE 8000