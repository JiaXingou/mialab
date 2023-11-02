FROM python:3.7.12
ADD . /contact
WORKDIR /contact
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN git clone https://github.com/JiaXingou/mialab.git && \
    cd mialab 
