FROM nvcr.io/nvidia/pytorch:20.10-py3

WORKDIR /aravind/

ADD . .

RUN pip install -r requirements.txt

