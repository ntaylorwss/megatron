FROM tensorflow/tensorflow:latest-py3

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

RUN apt-get update && apt-get install -y graphviz python3-tk

RUN useradd -ms /bin/bash megatron
RUN mkdir /etc/jupyter /home/megatron/.jupyter
ADD https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/jupyter_notebook_config.py /etc/jupyter/
RUN echo "c.NotebookApp.token = u''" > /home/megatron/.jupyter/jupyter_notebook_config.py && \
    chown megatron:megatron /home/megatron/.jupyter/jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py
USER megatron

CMD jupyter notebook
RUN mkdir /home/megatron/work
WORKDIR /home/megatron/work

ENV PATH "/home/megatron/.local/bin:${PATH}"

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt

COPY megatron/ /usr/local/lib/python3.6/dist-packages/megatron
