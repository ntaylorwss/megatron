FROM ntaylor22/tensorflow-cpu

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

RUN useradd -ms /bin/bash piper
RUN mkdir /home/piper/.jupyter && \
    echo "c.NotebookApp.token = u''" > /home/piper/.jupyter/jupyter_notebook_config.py
RUN chown piper:piper /etc/jupyter/jupyter_notebook_config.py
USER piper

CMD jupyter notebook
RUN mkdir /home/piper/work
WORKDIR /home/piper/work

COPY megatron/ /usr/local/lib/python3.6/dist-packages/megatron
COPY tests/ /tmp
#RUN python -m pytest /tmp
