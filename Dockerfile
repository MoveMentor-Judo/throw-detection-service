FROM python:3.12-slim
LABEL authors="williamstone"

WORKDIR /app
RUN pip install poetry==1.8.3

COPY pyproject.toml poetry.lock* /app/
# Have Poetry not create a virtual environment inside the container
RUN poetry config virtualenvs.create false

RUN poetry install --no-dev --no-interaction --no-ansi
COPY . /app/

EXPOSE 5000

CMD ["python", "app.py"]