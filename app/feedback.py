import logging
import sqlite3
import os
import time

logger = logging.getLogger(__name__)


def get_db_file() -> str:
    """
    Returns either the database location that was provided
    via the FEEDBACK_DB environment variable, or returns
    a default location.
    """
    db_file = os.getenv("FEEDBACK_DB")
    if not db_file:
        db_file = "/tmp/feedback.db"
        logger.warning(
            "No feedback database file specified, storing feedback in default location: '%s'",
            db_file,
        )
    return db_file


def create_connection() -> sqlite3.Connection | None:
    """
    Creates connection with the Sqlite database.
    """

    db_file = get_db_file()
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        logger.error(e)
    return conn


def create_table(conn: sqlite3.Connection):
    """
    Creates the feedback table if it does not exist.
    """
    try:
        sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            type text NOT NULL,
                                            score text NOT NULL,
                                            text text,
                                            timestamp text NOT NULL
                                        ); """
        c = conn.cursor()
        c.execute(sql_create_feedback_table)
    except sqlite3.Error as e:
        logger.error(e)


def store_feedback(feedback: dict) -> None:
    """
    Stores feedback from the provided feedback dict, which
    comes from the streamlit-feedback component.

    If the database does not exist, it will be created.
    """
    logger.info("Got feedback: %s", str(feedback))
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        sql = """ INSERT INTO feedback(type,score,text,timestamp)
                    VALUES(?,?,?,?) """
        cur = conn.cursor()
        logger.debug("Storing feedback in database.")
        cur.execute(
            sql,
            (
                feedback["type"],
                feedback["score"],
                feedback.get("text", ""),
                time.ctime(),
            ),
        )
        conn.commit()
        logger.debug("Feedback stored in database.")
    else:
        logger.error("Error! cannot create the database connection.")
