FROM python:3.7

EXPOSE 8501

# upgrade pip
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# copy over app-relevant files
COPY . /app
WORKDIR /app

# run
ENTRYPOINT ["streamlit", "run", "app.py"]