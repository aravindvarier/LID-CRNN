FROM nvcr.io/nvidia/pytorch:20.10-py3

ADD requirements.txt /aravind/

WORKDIR /aravind/

RUN pip install -r requirements.txt

