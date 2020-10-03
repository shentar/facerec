import sqlite3
import queue


def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class SQLiteUtil(object):
    __queue_conn = queue.Queue(maxsize=1)
    __path = None

    def __init__(self, path):
        self.__path = path
        print('path:', self.__path)
        self.__create_conn()

    def __create_conn(self):
        conn = sqlite3.connect(self.__path, check_same_thread=False)
        self.__queue_conn.put(conn)

    def __close(self, cursor, conn):
        if cursor is not None:
            cursor.close()
        if conn is not None:
            cursor.close()
            self.__create_conn()

    def execute_query(self, sql, params):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        try:
            if not params is None:
                records = cursor.execute(sql, params).fetchall()
            else:
                records = cursor.execute(sql).fetchall()
            field = [i[0] for i in cursor.description]
            value = [dict(zip(field, i)) for i in records]
        finally:
            self.__close(cursor, conn)
        return value

    def execute(self, sql, params):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        try:
            if not params is None:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            conn.commit()
        except Exception:
            conn.rollback()
        finally:
            self.__close(cursor, conn)

    def executescript(self, sql):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        try:
            cursor.executescript(sql)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.__close(cursor, conn)
