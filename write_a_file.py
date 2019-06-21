from datetime import datetime
import os
import logging
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python_operator import PythonOperator
import numpy as np
import datetime


log = logging.getLogger(__name__)

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'retries': 0
}

dag = DAG('test_dag',
          description='Test DAG',
          catchup=False,
          schedule_interval='1 * * * *',
          default_args=default_args,
          start_date=datetime(2018, 8, 8))

path = '/home/aamir/airflow/local/lib/python2.7/site-packages/airflow'

if os.path.isdir(path):
    os.chdir(path)
else:
    os.mkdir(path)
    os.chdir(path)


def write_some_file():
    try:
        with save("/home/aamir/" '%s.csv' %currentDT_min "wt") as fout:
            fout.write('test1\n')
    except Exception as e:
        log.error(e)
        raise AirflowException(e)


write_file_task = PythonOperator(
    task_id='write_some_file',
    python_callable=write_some_file,
    dag=dag
)
