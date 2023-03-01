FROM python:3.7

EXPOSE 8080

# upgrade pip
RUN pip install -U pip

COPY ../requirements.txt app/requirements.txt
RUN pip install -r app/requirements

# copy over app-relevant files
COPY . /app
WORKDIR /app

# run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]