FROM python:3.9-slim-buster

COPY ./ /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696","web_service:app"]