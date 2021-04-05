FROM python:3.7

WORKDIR /data/www

COPY requirements.txt /data/www

RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt

COPY sources.list /etc/apt

RUN apt-get update

RUN apt-get install -y \
    build-essential \
    tk-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline6-dev \
    libdb5.3-dev \
    libgdbm-dev \
    libsqlite3-dev \
    libssl-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    zlib1g-dev \
    libffi-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libqtgui4 \
    libqt4-test

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir tmp
RUN mkdir /root/.keras
RUN mkdir /root/.keras/models

COPY vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models
COPY vgg16_weights_tf_dim_ordering_tf_kernels.h5 /root/.keras/models
COPY resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth

COPY . /data/www

EXPOSE 5500 5501 5502 5503

CMD ["sh", "/data/www/run.sh"]
