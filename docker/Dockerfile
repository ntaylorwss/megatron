FROM ntaylor22/tensorflow-cpu

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

RUN useradd -ms /bin/bash megatron
RUN mkdir /home/megatron/.jupyter && \
    echo "c.NotebookApp.token = u''" > /home/megatron/.jupyter/jupyter_notebook_config.py
RUN chown megatron:megatron /etc/jupyter/jupyter_notebook_config.py
USER megatron

CMD jupyter notebook
RUN mkdir /home/megatron/work
WORKDIR /home/megatron/work

COPY megatron/ /usr/local/lib/python3.6/dist-packages/megatron
