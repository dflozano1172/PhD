import pandas as pd
from pathlib import Path
import sqlite3
from sqlite3 import Connection
import streamlit as st

URI_SQLITE_DB = r"C:\Users\Lenovo\Documents\GitHub\PhD\MockData\test1.db"

def main():
    st.title("My Super Calculator")
    st.markdown("Enter data in database from sidebar, then run the **mighty** calculator")
    
    conn = get_connection(URI_SQLITE_DB)
    init_db(conn)

    build_sidebar(conn)
    display_data(conn)
    run_calculator(conn)
    
    
def sql_fetch_table(con):
    cursorObj = con.cursor()
    cursorObj.execute('SELECT name from sqlite_master where type= "table"')
    print(cursorObj.fetchall())



def init_db(conn: Connection):
    conn.execute(
        """CREATE TABLE IF NOT EXISTS test1
            (
                INPUT1 INT,
                INPUT2 INT
            );"""
    )
    conn.commit()
#cursor = conn.execute('select * from test1')
#names = list(map(lambda x: x[0], cursor.description))

def build_sidebar(conn: Connection):
    st.sidebar.header("Configuration")
    input1 = st.sidebar.slider("Input 1", 0, 100,50)
    input2 = st.sidebar.slider("Input 2", 0, 100,50)
    if st.sidebar.button("Save to database"):
        conn.execute(f"INSERT INTO test1 (INPUT1, INPUT2) VALUES ({input1}, {input2})")
        conn.commit()


def display_data(conn: Connection):
    if st.checkbox("Display data in sqlite databse"):
        st.dataframe(get_data(conn))


def run_calculator(conn: Connection):
    if st.button("Run calculator"):
        st.info("Run your function")
        df = get_data(conn)
        st.write(df.sum())


def get_data(conn: Connection):
    df = pd.read_sql("SELECT * FROM test1", con=conn)
    return df


@st.cache(hash_funcs={Connection: id})
def get_connection(path: str):
    """Put the connection in cache to reuse if path does not change between Streamlit reruns.
    NB : https://stackoverflow.com/questions/48218065/programmingerror-sqlite-objects-created-in-a-thread-can-only-be-used-in-that-sa
    """
    return sqlite3.connect(path, check_same_thread=False)


if __name__ == "__main__":
    main()