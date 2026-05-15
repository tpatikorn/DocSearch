import datetime
import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()


def get_db_connection():
    # Supports both DATABASE_URL or individual components
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = psycopg.connect(db_url, row_factory=dict_row)
    else:
        conn = psycopg.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_SERVER"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_DB"),
            row_factory=dict_row
        )
    return conn


def execute_query(query, params=None, fetch=True):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            conn.commit()


def execute_batch(query, params_list):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for params in params_list:
                cur.execute(query, params)
            conn.commit()


def debug_print(*args, **kwargs):
    print(f"[{datetime.datetime.now()}]", *args, **kwargs)


def error_log(process, message):
    process = str(process)
    message = str(message)
    debug_print(process, message)
    execute_query("""
                  INSERT INTO error_logs (process, message)
                  VALUES (%s, %s)""",
                  (process, message), fetch=False)


def init_db():
    with open('schema.sql', 'r') as f:
        schema = f.read()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema)
            conn.commit()
