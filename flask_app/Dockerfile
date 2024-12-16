FROM python:3.10-slim

LABEL maintainer "suhan_web"

RUN pip install --upgrade pip
RUN mkdir -p /web

ENV APP_PATH /web

COPY requirements.txt $APP_PATH/
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt

COPY app.py $APP_PATH/
COPY templates/ $APP_PATH/templates/
COPY static/ $APP_PATH/static/

EXPOSE 5001

CMD ["python", "/web/app.py"]
