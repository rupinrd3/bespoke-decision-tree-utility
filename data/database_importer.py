# utility/data/database_importer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Importer Module for Bespoke Utility
Handles importing data from various database systems.

"""


import logging
import sqlite3
import os
from typing import Dict, Any, Optional, List
import pandas as pd

try:
    import pyodbc
except ImportError:
    pyodbc = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import mysql.connector
except ImportError:
    mysql = None
else:
    mysql = mysql.connector

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None

logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass

class DatabaseQueryError(Exception):
    """Custom exception for database query errors."""
    pass

class DatabaseImporter:
    """
    Class responsible for importing data from various database sources.
    This class will need to be significantly expanded with methods for each
    supported database type and proper error handling.
    """

    SUPPORTED_DB_TYPES = ["sqlite", "postgresql", "mysql", "sqlserver", "oracle"] # Example

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_params = {} # To be populated by UI or config
        self.connection = None
        self.cursor = None
        logger.info("DatabaseImporter initialized.")

    def connect(self, db_type: str, connection_details: Dict[str, Any]) -> bool:
        """
        Establishes a connection to the specified database.

        Args:
            db_type: The type of database (e.g., 'postgresql', 'mysql', 'sqlite').
            connection_details: A dictionary containing connection parameters
                                (host, port, user, password, database_name, etc.).

        Returns:
            True if connection is successful, False otherwise.

        Raises:
            DatabaseConnectionError: If connection fails.
            NotImplementedError: If db_type is not supported.
        """
        self.disconnect()  # Ensure any existing connection is closed
        self.connection_params = connection_details
        db_type = db_type.lower()

        logger.info(f"Attempting to connect to {db_type} database...")

        try:
            if db_type == "sqlite":
                db_path = connection_details.get('database_path')
                if not db_path:
                    raise DatabaseConnectionError("SQLite database path not provided.")
                if not os.path.exists(db_path):
                    raise DatabaseConnectionError(f"SQLite database file not found: {db_path}")
                self.connection = sqlite3.connect(db_path)
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to SQLite database: {db_path}")

            elif db_type == "postgresql":
                if psycopg2 is None:
                    raise NotImplementedError("PostgreSQL connector requires 'psycopg2' library. Install with: pip install psycopg2-binary")
                
                self.connection = psycopg2.connect(
                    host=connection_details.get('host'),
                    port=connection_details.get('port', 5432),
                    user=connection_details.get('user'),
                    password=connection_details.get('password'),
                    dbname=connection_details.get('database_name')
                )
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to PostgreSQL: {connection_details.get('host')}/{connection_details.get('database_name')}")

            elif db_type == "mysql":
                if mysql is None:
                    raise NotImplementedError("MySQL connector requires 'mysql-connector-python' library. Install with: pip install mysql-connector-python")
                
                self.connection = mysql.connect(
                    host=connection_details.get('host'),
                    port=connection_details.get('port', 3306),
                    user=connection_details.get('user'),
                    password=connection_details.get('password'),
                    database=connection_details.get('database_name')
                )
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to MySQL: {connection_details.get('host')}/{connection_details.get('database_name')}")

            elif db_type == "sqlserver":
                if pyodbc is None:
                    raise NotImplementedError("SQL Server connector requires 'pyodbc' library. Install with: pip install pyodbc")
                
                driver = connection_details.get('driver', '{ODBC Driver 17 for SQL Server}')
                server = connection_details.get('host')
                database = connection_details.get('database_name')
                username = connection_details.get('user')
                password = connection_details.get('password')
                port = connection_details.get('port', 1433)
                
                if connection_details.get('trusted_connection', False):
                    conn_str = f"DRIVER={driver};SERVER={server},{port};DATABASE={database};Trusted_Connection=yes;"
                else:
                    conn_str = f"DRIVER={driver};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};"
                
                self.connection = pyodbc.connect(conn_str)
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to SQL Server: {server}/{database}")

            elif db_type == "oracle":
                if cx_Oracle is None:
                    raise NotImplementedError("Oracle connector requires 'cx_Oracle' library. Install with: pip install cx_Oracle")
                
                host = connection_details.get('host')
                port = connection_details.get('port', 1521)
                service_name = connection_details.get('service_name')
                sid = connection_details.get('sid')
                username = connection_details.get('user')
                password = connection_details.get('password')
                
                if service_name:
                    dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
                elif sid:
                    dsn = cx_Oracle.makedsn(host, port, sid=sid)
                else:
                    raise DatabaseConnectionError("Oracle connection requires either 'service_name' or 'sid'")
                
                self.connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
                self.cursor = self.connection.cursor()
                logger.info(f"Successfully connected to Oracle: {host}/{service_name or sid}")
            
            else:
                raise NotImplementedError(f"Database type '{db_type}' is not supported.")
            
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {db_type} database: {e}", exc_info=True)
            self.connection = None
            self.cursor = None
            raise DatabaseConnectionError(f"Connection to {db_type} failed: {e}")


    def disconnect(self):
        """Closes the database connection."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                logger.warning(f"Error closing cursor: {e}")
            finally:
                self.cursor = None
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.connection = None
        self.connection_params = {}

    def list_tables_or_views(self, schema: Optional[str] = None) -> Optional[List[str]]:
        """
        Lists tables and views available in the connected database.
        Specific implementation depends on the database type.
        
        Args:
            schema: Schema name (optional, for databases that support schemas)
        """
        if not self.connection or not self.cursor:
            logger.error("Not connected to any database.")
            raise DatabaseConnectionError("Not connected to database.")

        items = []
        db_type = self.get_connected_db_type()
        logger.info(f"Listing tables/views for {db_type} database.")

        try:
            if db_type == "sqlite":
                self.cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name;")
                items = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "postgresql":
                if schema:
                    query = """SELECT table_name FROM information_schema.tables 
                              WHERE table_schema = %s 
                              UNION 
                              SELECT table_name FROM information_schema.views 
                              WHERE table_schema = %s 
                              ORDER BY table_name"""
                    self.cursor.execute(query, (schema, schema))
                else:
                    query = """SELECT table_name FROM information_schema.tables 
                              WHERE table_schema = 'public' 
                              UNION 
                              SELECT table_name FROM information_schema.views 
                              WHERE table_schema = 'public' 
                              ORDER BY table_name"""
                    self.cursor.execute(query)
                items = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "mysql":
                if schema:
                    self.cursor.execute(f"USE {schema}")
                self.cursor.execute("SHOW FULL TABLES WHERE Table_type IN ('BASE TABLE', 'VIEW')")
                items = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "sqlserver":
                if schema:
                    query = """SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                              WHERE TABLE_SCHEMA = ? AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                              ORDER BY TABLE_NAME"""
                    self.cursor.execute(query, schema)
                else:
                    query = """SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
                              WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                              ORDER BY TABLE_NAME"""
                    self.cursor.execute(query)
                items = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "oracle":
                if schema:
                    query = """SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = :schema 
                              UNION 
                              SELECT VIEW_NAME FROM ALL_VIEWS WHERE OWNER = :schema 
                              ORDER BY TABLE_NAME"""
                    self.cursor.execute(query, schema=schema.upper())
                else:
                    query = """SELECT TABLE_NAME FROM USER_TABLES 
                              UNION 
                              SELECT VIEW_NAME FROM USER_VIEWS 
                              ORDER BY TABLE_NAME"""
                    self.cursor.execute(query)
                items = [row[0] for row in self.cursor.fetchall()]
            
            else:
                raise NotImplementedError(f"Listing tables for '{db_type}' is not implemented.")
            
            return items
        except Exception as e:
            logger.error(f"Error listing tables/views: {e}", exc_info=True)
            raise DatabaseQueryError(f"Could not list tables/views: {e}")


    def import_data_from_query(self, query: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Imports data from the connected database using a SQL query.

        Args:
            query: The SQL query to execute.
            chunk_size: If provided, read data in chunks (useful for large datasets).

        Returns:
            A pandas DataFrame containing the query results.

        Raises:
            DatabaseConnectionError: If not connected to a database.
            DatabaseQueryError: If the query fails.
        """
        if not self.connection:
            logger.error("Not connected to any database.")
            raise DatabaseConnectionError("Not connected to database to execute query.")

        logger.info(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        try:
            if chunk_size and chunk_size > 0:
                chunks = []
                for chunk_df in pd.read_sql_query(query, self.connection, chunksize=chunk_size):
                    chunks.append(chunk_df)
                if not chunks: return pd.DataFrame()
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql_query(query, self.connection)
            
            logger.info(f"Query successful. Fetched {len(df)} rows and {len(df.columns)} columns.")
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}", exc_info=True)
            raise DatabaseQueryError(f"Query execution failed: {e}")

    def import_table(self, table_name: str, schema: Optional[str] = None, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Imports an entire table from the connected database.

        Args:
            table_name: The name of the table to import.
            schema: The schema of the table (optional, for databases that use schemas).
            chunk_size: If provided, read data in chunks.

        Returns:
            A pandas DataFrame containing the table data.
        """
        query = f"SELECT * FROM {f'{schema}.{table_name}' if schema else table_name}"
        return self.import_data_from_query(query, chunk_size)

    def get_connected_db_type(self) -> Optional[str]:
        """Helper to infer current DB type."""
        if self.connection:
            conn_type = str(type(self.connection)).lower()
            if "sqlite" in conn_type: 
                return "sqlite"
            elif "psycopg2" in conn_type or "postgresql" in conn_type: 
                return "postgresql"
            elif "mysql" in conn_type: 
                return "mysql"
            elif "pyodbc" in conn_type: 
                return "sqlserver"
            elif "cx_oracle" in conn_type or "oracle" in conn_type: 
                return "oracle"
        return None
    
    def list_schemas(self) -> Optional[List[str]]:
        """
        Lists available schemas in the connected database.
        Only applicable for databases that support schemas.
        """
        if not self.connection or not self.cursor:
            logger.error("Not connected to any database.")
            raise DatabaseConnectionError("Not connected to database.")

        db_type = self.get_connected_db_type()
        schemas = []

        try:
            if db_type == "postgresql":
                self.cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast') ORDER BY schema_name")
                schemas = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "mysql":
                self.cursor.execute("SHOW DATABASES")
                schemas = [row[0] for row in self.cursor.fetchall()]
                schemas = [s for s in schemas if s not in ('information_schema', 'performance_schema', 'mysql', 'sys')]
            
            elif db_type == "sqlserver":
                self.cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME NOT IN ('information_schema', 'sys') ORDER BY SCHEMA_NAME")
                schemas = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "oracle":
                self.cursor.execute("SELECT USERNAME FROM ALL_USERS WHERE USERNAME NOT IN ('SYS', 'SYSTEM', 'ANONYMOUS', 'APEX_040000', 'APEX_PUBLIC_USER', 'APPQOSSYS', 'CTXSYS', 'DBSNMP', 'DIP', 'FLOWS_FILES', 'HR', 'MDSYS', 'ORACLE_OCM', 'XDB', 'XS$NULL') ORDER BY USERNAME")
                schemas = [row[0] for row in self.cursor.fetchall()]
            
            elif db_type == "sqlite":
                return []
            
            else:
                raise NotImplementedError(f"Schema listing for '{db_type}' is not implemented.")
            
            return schemas
        except Exception as e:
            logger.error(f"Error listing schemas: {e}", exc_info=True)
            raise DatabaseQueryError(f"Could not list schemas: {e}")
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a table including column names, types, and row count.
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            
        Returns:
            Dictionary containing table information
        """
        if not self.connection or not self.cursor:
            raise DatabaseConnectionError("Not connected to database.")
        
        db_type = self.get_connected_db_type()
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        try:
            columns = []
            if db_type == "sqlite":
                self.cursor.execute(f"PRAGMA table_info({table_name})")
                for row in self.cursor.fetchall():
                    columns.append({
                        'name': row[1],
                        'type': row[2],
                        'nullable': not row[3],
                        'primary_key': bool(row[5])
                    })
            
            elif db_type == "postgresql":
                query = """SELECT column_name, data_type, is_nullable, 
                                  CASE WHEN column_name IN (
                                      SELECT column_name FROM information_schema.key_column_usage 
                                      WHERE table_name = %s AND table_schema = %s
                                  ) THEN true ELSE false END as is_primary_key
                           FROM information_schema.columns 
                           WHERE table_name = %s AND table_schema = %s
                           ORDER BY ordinal_position"""
                schema_name = schema or 'public'
                self.cursor.execute(query, (table_name, schema_name, table_name, schema_name))
                for row in self.cursor.fetchall():
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2] == 'YES',
                        'primary_key': row[3]
                    })
            
            elif db_type in ["mysql", "sqlserver", "oracle"]:
                test_query = f"SELECT * FROM {full_table_name} LIMIT 1"
                if db_type == "oracle":
                    test_query = f"SELECT * FROM {full_table_name} WHERE ROWNUM <= 1"
                elif db_type == "sqlserver":
                    test_query = f"SELECT TOP 1 * FROM {full_table_name}"
                
                df_sample = pd.read_sql_query(test_query, self.connection)
                for col in df_sample.columns:
                    columns.append({
                        'name': col,
                        'type': str(df_sample[col].dtype),
                        'nullable': True,  # Default assumption
                        'primary_key': False  # Would need more complex query to determine
                    })
            
            count_query = f"SELECT COUNT(*) FROM {full_table_name}"
            self.cursor.execute(count_query)
            row_count = self.cursor.fetchone()[0]
            
            return {
                'table_name': table_name,
                'schema': schema,
                'columns': columns,
                'row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}", exc_info=True)
            raise DatabaseQueryError(f"Could not get table info: {e}")
    
    def test_connection(self) -> bool:
        """
        Test if the current connection is still valid.
        
        Returns:
            True if connection is valid, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            db_type = self.get_connected_db_type()
            if db_type == "sqlite":
                self.cursor.execute("SELECT 1")
            elif db_type == "postgresql":
                self.cursor.execute("SELECT 1")
            elif db_type == "mysql":
                self.cursor.execute("SELECT 1")
            elif db_type == "sqlserver":
                self.cursor.execute("SELECT 1")
            elif db_type == "oracle":
                self.cursor.execute("SELECT 1 FROM DUAL")
            else:
                return False
            
            self.cursor.fetchone()
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.disconnect()
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    db_file = "temp_test_db.sqlite"
    conn_test = sqlite3.connect(db_file)
    cursor_test = conn_test.cursor()
    cursor_test.execute("DROP TABLE IF EXISTS test_table")
    cursor_test.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, value REAL)")
    cursor_test.executemany("INSERT INTO test_table (name, value) VALUES (?, ?)",
                       [('alpha', 1.23), ('beta', 4.56), ('gamma', 7.89)])
    conn_test.commit()
    conn_test.close()
    
    importer = DatabaseImporter(config={})
    try:
        sqlite_details = {'database_path': db_file}
        if importer.connect("sqlite", sqlite_details):
            print("Connected to SQLite.")
            tables = importer.list_tables_or_views()
            print("Tables/Views:", tables)
            if tables and "test_table" in tables:
                df_table = importer.import_table("test_table")
                print("\nImported test_table:\n", df_table)

                df_query = importer.import_data_from_query("SELECT name, value FROM test_table WHERE value > 2.0")
                print("\nImported from query (value > 2.0):\n", df_query)
            importer.disconnect()
    except Exception as e:
        print(f"An error occurred during SQLite test: {e}")
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"\nCleaned up {db_file}")

