FROM nvcr.io/nvidia/pytorch:20.10-py3

ADD req.txt /aravind/

WORKDIR /aravind/

RUN pip install -r req.txt

