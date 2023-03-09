FROM python:3.7

EXPOSE 8080

# upgrade pip
RUN pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy over backend relevant files
COPY . /app
WORKDIR /app

# run
ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]