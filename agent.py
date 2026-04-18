import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import oracledb
from sqlalchemy import (create_engine, text)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient-basic-windows.x64-19.20.0.0.0dbru\instantclient_19_20")

DB_USER = os.getenv("DB_LAKE_USER", "")
DB_PASSWORD = os.getenv("DB_LAKE_PASSWORD", "")
DB_HOST = os.getenv("DB_LAKE_HOST", "MTL-ORACLE-SVR")
DB_PORT = os.getenv("DB_LAKE_PORT", "1521")
DB_SERVICE = os.getenv("DB_LAKE_SERVICE", "mtle")

engine = create_engine(f"oracle+oracledb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?service_name={DB_SERVICE}")

### [HELPER] - Mapping Prefix table for LAKES's table
def mapping_prefix_table(department: str):
    """[HELPER] Function for generating table's prefix base on DEPARTMENT input."""
    prefix_table = ""
    
    if (department.upper() == "MT600") or (department.upper() == "MT700"):
        prefix_table = f"{department.upper()}A"
    else: prefix_table = department.upper()
    
    return prefix_table

### [HELPER] - Get Process-list for target lotno
def get_process_list(lotno:str, department: str):
    """[HELPER] Function for retrieving process lists base on LOTNO, DEPARTMENT input."""
    
    prefix_table = mapping_prefix_table(department)
    process_sql = f"""
        SELECT DISTINCT 
            NOC0008 AS process_code 
        FROM {prefix_table}_PTH0001
        WHERE NOC0027 = '{lotno}'
        ORDER BY NOC0008
    """
    
    with engine.connect() as connection:
        result = pd.read_sql_query(text(process_sql), connection)
    
    return result["process_code"].to_list()
    
### [HELPER] - Get previous lotno for target lotno
def get_previous_lotno(lotno: str, department: str):
    """[HELPER] Function for retrieving previous lotno base on LOTNO, DEPARTMENT input"""
    
    prefix_table = mapping_prefix_table(department)
    previous_sql = f"""
        SELECT
            process_code, 
            current_lotno, 
            previous_lotno
        FROM (
            SELECT
                NOC0008 AS process_code,
                NOC0027 AS current_lotno,
                LAG(NOC0027) OVER (
                    PARTITION BY NOC0008
                    ORDER BY HID0013_1, HID0004_1, hid0006, tmc0004
                ) AS previous_lotno,
                HID0013_1 AS current_op_date,
                LAG(HID0013_1) OVER (
                    PARTITION BY NOC0008
                    ORDER BY HID0013_1, HID0004_1, hid0006, tmc0004
                ) AS previous_op_date,
                ROW_NUMBER() OVER (PARTITION BY NOC0008 ORDER BY HID0013_1 DESC, HID0004_1 DESC, hid0006 DESC, tmc0004 DESC) as rn
            FROM (
                SELECT
                    noc0027,
                    noc0008,
                    hid0013_1, 
                    HID0004_1,
                    HID0006,
                    TMC0004
                FROM {prefix_table}_PTH0001
                WHERE TRUNC(HID0013_1) >= TRUNC(SYSDATE) - 100
            ) 
        )
        WHERE current_lotno = '{lotno}'
    """
    
    with engine.connect() as connection:
        result = pd.read_sql_query(text(previous_sql), connection)
        
    return result

### Get COMMON information for lotno (Short-1-Common)
@tool
def get_common_information(lotno: str, department: str):
    """Get common LOTNO information focusing on target lotno. Require input LOTNO, DEPARTMENT."""
    
    # Initial "prefix_table" and "sql" for retrieving data
    prefix_table = str(mapping_prefix_table(department))
    common_sql = f"""
        SELECT 
            t_final.lotno, 
            t_final.process_code, 
            t2.DHC0013 AS process_name, 
            t3.DHC0009 AS lot_type,
            t_final.murata_type, 
            t_final.machine_name, 
            t_final.line_name, 
            t_final.emp_code,
            t_final.lot_status, 
            t_final.time_waiting_hr, 
            t_final.time_actual_hr, 
            t_final.start_date
        FROM (
            SELECT * FROM (
                SELECT 
                    noc0027 AS lotno, 
                    noc0008 AS process_code, 
                    cdc0145 AS emp_code, 
                    cdc0163 AS murata_type, 
                    dhc2022 AS machine_name, 
                    cdc0019 AS line_name,
                    kbc0010,
                    CASE 
                        WHEN kbc0012 = 0 THEN 'WIP'
                        WHEN kbc0012 = 1 THEN 'Completed'
                        WHEN kbc0012 = 8 THEN 'Lot Out'
                        WHEN kbc0012 = 9 THEN 'Lot Remove'
                        ELSE '---Error---'
                    END AS lot_status,
                    ROUND(nin0016 * 24, 2) AS time_waiting_hr,
                    ROUND(nin0015 * 24, 2) AS time_actual_hr,
                    TO_CHAR(hid0013_1, 'DD-MM-YY HH12:MI AM') AS start_date,
                    ROW_NUMBER() OVER (PARTITION BY noc0008 ORDER BY hid0004_1 DESC) as rn
                FROM {prefix_table}_PTH0001
                WHERE NOC0027 = '{lotno}'
            ) inner_t
            WHERE rn = 1
        ) t_final
        LEFT JOIN {prefix_table}_PTM0014 t2 ON t_final.process_code = t2.NOC0008
        LEFT JOIN {prefix_table}_PTC0006 t3 ON t_final.KBC0010 = t3.KBC0010
        ORDER BY t_final.process_code
    """
    
    # oracle connection
    with engine.connect() as connection:
        result = pd.read_sql_query(text(common_sql), connection)
    
    return result

### Get 4M-Man factor information for lotno (Short-1-Man)
@tool
def get_man_factor_information(lotno: str, department: str):
    """Get 4M Factor (man) information for target lotno. Require input LOTNO, DEPARTMENT."""
    
    # Initial "prefix_table", "process_list" as str, "sql" for retrieving data
    prefix_table = str(mapping_prefix_table(department))
    process_list = get_process_list(lotno, department)
    process_list_str = ",".join(process_list)
    man_factor_sql = f"""
        SELECT 
            process_code,
            emp_code,
            current_op_lotno,
            previous_op_lotno,
            TO_CHAR(current_op_date, 'DD-MM-YY HH12:MI AM') as current_op_date,
            TO_CHAR(previous_op_date, 'DD-MM-YY HH12:MI AM') as previous_op_date,
            ROUND((current_op_date - previous_op_date)*24, 2) AS previous_op_date_gap_hr
        FROM (
            SELECT
                t1.NOC0008 AS process_code,
                t1.CDC0145 AS emp_code,
                '{lotno}' AS current_op_lotno,
                t1.NOC0027 AS previous_op_lotno,
                t2.HID0013_1 AS current_op_date,
                LAG(t1.HID0013_1) OVER (PARTITION BY  t1.CDC0145,t1.NOC0008 ORDER BY t1.HID0013_1) AS previous_op_date,
                ROW_NUMBER() OVER (PARTITION BY t1.NOC0008 ORDER BY t1.HID0013_1 DESC) as rn
            FROM (
                SELECT noc0008, cdc0145, noc0027, hid0013_1 
                FROM {prefix_table}_PTH0001
                WHERE (NOC0008, CDC0145) IN (
                    SELECT DISTINCT NOC0008,CDC0145  
                    FROM {prefix_table}_PTH0001
                    WHERE NOC0027 = '{lotno}'
                    AND NOC0008 IN ({process_list_str})
                ) AND TRUNC(HID0013_1) >= TRUNC(SYSDATE) - 100
            ) t1
            JOIN (
                SELECT 
                    HID0013_1,
                    NOC0008
                FROM {prefix_table}_PTH0001 
                WHERE noc0027 = '{lotno}' 
                AND NOC0008 IN ({process_list_str})
            ) t2
            ON t2.hid0013_1 > t1.hid0013_1 AND t2.NOC0008 = t1.noc0008
            ORDER BY t1.NOC0008, t1.HID0013_1 DESC
        ) WHERE rn=1
    """
    
    # oracle connection
    with engine.connect() as connection:
        result = pd.read_sql_query(text(man_factor_sql), connection)
        
    return result

### Get 4M-Machine factor information for lotno (Short-1-Machine)
@tool
def get_mc_factor_information(lotno: str, department: str):
    """Get 4M Factor (machine) information for target lotno. Require input LOTNO, DEPARTMENT."""
    
    # Initial "prefix_table", "sql" for retrieving data
    prefix_table = str(mapping_prefix_table(department))
    mc_factor_sql = f"""
        SELECT 
            t1.*,
            t2.bm_time_sum_min_last_7days,
            t2.bm_count_last_7days,
            ROUND(t2.bm_time_sum_min_last_7days/ t2.bm_count_last_7days ) AS bm_avg_min_last_7days
        FROM 
        (
            SELECT t1_inner.*, t2_inner.lineinfo_id FROM (
                SELECT 
                    NOC0008 AS process_code,  
                    DHC2022 AS machine_name
                FROM {prefix_table}_PTH0001 
                WHERE NOC0027 = '{lotno}'
            ) t1_inner
            LEFT JOIN (
                SELECT machine_id, machine_name, lineinfo_id, process_cd
                FROM pstm0002@mtlb 
                WHERE department = '{department.upper()}' 
                AND machine_name IN (SELECT DISTINCT dhc2022 FROM {prefix_table}_PTH0001 WHERE NOC0027 = '{lotno}')
            ) t2_inner
            ON t2_inner.machine_name = t1_inner.machine_name AND t2_inner.process_cd = t1_inner.process_code
        ) t1
        LEFT JOIN
        (
            SELECT lineinfo_id,
            SUM(ROUND(
                (TO_DATE(TO_CHAR(mnrec_datetime, 'DD-MM-YYYY HH24:MI:SS'), 'DD-MM-YYYY HH24:MI:SS') - 
                TO_DATE(TO_CHAR(requested_datetime, 'DD-MM-YYYY HH24:MI:SS'), 'DD-MM-YYYY HH24:MI:SS')) 
                * 1440
            )) AS bm_time_sum_min_last_7days,
            COUNT(*) AS bm_count_last_7days
            FROM psth0004@mtlb
            WHERE lineinfo_id IN (
                    SELECT DISTINCT lineinfo_id
                    FROM pstm0002@mtlb 
                    WHERE department = '{department.upper()}' 
                    AND machine_name IN (SELECT DISTINCT dhc2022 FROM {prefix_table}_PTH0001 WHERE NOC0027 = '{lotno}')
            )
            AND loss_cd = 'WR'
            AND lot_no IS NOT NULL
            AND mnrec_datetime IS NOT null
            AND requested_datetime >= CURRENT_DATE-30
            GROUP BY lineinfo_id
        ) t2
        ON t2.lineinfo_id = t1.lineinfo_id
        ORDER BY t1.process_code
    """
    
    # oracle connection
    with engine.connect() as connection:
        result = pd.read_sql_query(text(mc_factor_sql), connection)
        
    # Fill NULL value with default text
    result.fillna("--- No data in PST ---", inplace=True)
        
    return result

### Get 4M-Material factor information for lotno (Short-1-Material)
@tool
def get_mat_factor_information(lotno: str, department: str):
    """Get 4M Factor (material) information for target lotno. Require input LOTNO, DEPARTMENT."""
    
    # Initial "prefix_table", "sql" for retrieving data
    prefix_table = str(mapping_prefix_table(department))
    mat_factor_sql = f"""
        SELECT 
            t1.process_code,
            t2.material_part,
            t2.material_desc,
            t2.material_lot_1,
            t2.material_lot_2,
            t2.material_lot_3,
            t2.material_lot_4
        FROM (SELECT 
                NOC0008 AS process_code
            FROM {prefix_table}_PTH0001 
            WHERE NOC0027 = '{lotno}'
        ) t1
        LEFT JOIN (
            SELECT 
                NOC0008 AS process_code, 
                DHC6188 AS material_part,
                DHC6189_1 AS material_desc,
                DHC0909_1 AS material_lot_1,
                DHC0909_2 AS material_lot_2,
                DHC0909_3 AS material_lot_3,
                DHC0909_4 AS material_lot_4
            FROM {prefix_table}_PTH4733 
            WHERE CDC3041 = 'Material' AND noc0027 = '{lotno}'
        ) t2 ON t2.process_code = t1.process_code
        ORDER BY t1.process_code
    """
    
    # oracle connection
    with engine.connect() as connection:
        result = pd.read_sql_query(text(mat_factor_sql), connection)
        
    # drop "ALL NULL" column, Then fill NULL value with default text
    result.dropna(axis=1, how='all', inplace=True)        
    # result.fillna("--- No Material data ---", inplace=True)
    
    ### Pivot Phase
    # 1. Capture ALL unique process_codes first (to ensure we don't lose the empty ones)
    all_processes = pd.DataFrame({'process_code': result['process_code'].unique()})
    
    # 2. Filter out rows that are just placeholders ('--- No Material data ---') --> build objects for rows that actually have data
    df_valid = result[result['material_part'] != '--- No Material data ---'].copy()
    
    # 3. Define the object builder
    def build_mat_object(row):
        # If the essential part ID is missing, return None
        if pd.isna(row['material_part']) or row['material_part'] == '':
            return None
        
        return {
            "mat_part": row['material_part'],
            "mat_desc": row['material_desc'],
            "mat_lot": [row['material_lot_1'], row['material_lot_2']]
        }
        
    # 4. Apply builder and create running numbers for valid data only
    df_valid['json_obj'] = df_valid.apply(build_mat_object, axis=1)
    df_valid['rank'] = df_valid.groupby('process_code').cumcount() + 1

    # 5. Pivot the valid data
    df_pivoted = df_valid.pivot(
        index='process_code', 
        columns='rank', 
        values='json_obj'
    )

    # Rename columns to mat_part_1, mat_part_2...
    df_pivoted.columns = [f"mat_part_{c}" for c in df_pivoted.columns]

    # 6. Merge back to the master list to keep ALL process_codes
    df_final = all_processes.merge(df_pivoted, on='process_code', how='left')

    # Optional: Fill the resulting NaNs with None for consistency
    df_final = df_final.where(pd.notnull(df_final), None)

    df_final.fillna("--- No Material data ---", inplace=True)

    
    return df_final

### Get Defective Ratio information for lotno (Short-1-Defective)
@tool
def get_defective_information(lotno: str, department: str):
    """Get defective ratio information for target lotno. Require input LOTNO, DEPARTMENT."""
    
    # Initial "def_sql" for  retrieving data (Base on condition for MT600, MT700, otherwise -> same pattern)
    def_sql = ""
    if department.upper() == "MT600":
        def_sql = f"""
            SELECT
                PRODUCT_NAME , PROCESS_CODE , GROUP_CUSTOMER, MODEL,
                MACHINENO,INPUT_QTY,OUTPUT_QTY,DEF_QTY,DEF_MODE
            FROM M_DEFECTIVE_MAWTR 
            WHERE lotno = '{lotno}'
            AND startdate_process >= sysdate-100
            ORDER BY PROCESS_CODE
        """
    elif department.upper() == "MT700":
        def_sql = f"""
            SELECT
                PRODUCT_NAME , PROCESS_CODE , GROUP_CUSTOMER, MODEL,
                MACHINENO,INPUT_QTY,OUTPUT_QTY,DEF_QTY,DEF_MODE
            FROM M_DEFECTIVE_{department} 
            WHERE lotno = '{lotno}'
            AND startdate_process >= sysdate-100
            ORDER BY PROCESS_CODE
        """
    else:
        def_sql = f"""
            SELECT
                PRODUCT_NAME , PROCESS_CODE , GROUP_CUSTOMER, MODEL,
                MACHINENO,INPUT_QTY,OUTPUT_QTY,DEF_QTY,DEF_MODE
            FROM M_DEFECTIVE_{department} 
            WHERE lotno = '{lotno}'
            AND startdate_process >= sysdate-100
            ORDER BY PROCESS_CODE
        """
    
    # oracle connection
    with engine.connect() as connection:
        result = pd.read_sql_query(text(def_sql), connection)
        
    result["def_mode"] = result["def_mode"].str.replace("N/A", "No Defective")
    result.fillna({"machineno": "No M/C"}, inplace=True)
    
    # 1. Prepare the data: Remove "No Defective" rows to keep the JSON clean
    # Only rows with actual defects will be turned into objects
    df_defects = result[result['def_mode'] != 'No Defective'].copy()

    # 2. Create the JSON object for each defect row
    # Result: {'RFL': 342}
    df_defects['def_obj'] = df_defects.apply(
        lambda x: {x['def_mode']: x['def_qty']}, axis=1
    )

    # 3. Group by process_code and aggregate
    # We keep the static columns (customer, model, machine, qty) and collect the objects
    df_final = result.groupby(['process_code', 'group_customer', 'model', 'machineno', 'input_qty', 'output_qty']).apply(
        lambda x: [obj for obj in x['def_mode'].map(
            lambda m: {m: x.loc[x['def_mode']==m, 'def_qty'].iloc[0]} if m != 'No Defective' else None # type: ignore
        ) if obj is not None]
    ).reset_index(name='def_mode')

    # 4. Handle cases with no defects: Change empty lists [] to None/Null
    df_final['def_mode'] = df_final['def_mode'].apply(lambda x: x if len(x) > 0 else None)
    df_final.fillna("--- No Defective ---", inplace=True)

    return df_final
    
    
llm = ChatOpenAI(
    model="gpt-5.2",
    base_url="https://prd-aoai-api.prd.other.internal.api-platform.murata.com/api",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'), # type: ignore
    default_headers={
        "client_id": os.getenv('AZURE_OPENAI_CLIENT_ID'),
        "client_secret": os.getenv('AZURE_OPENAI_CLIENT_SECRET'),
        "Content-Type": "application/json",
    }, # type: ignore
    default_query={
        "api-version": os.getenv('AZURE_OPENAI_API_VERSION')
    }
)

# Initialize the memory saver
memory = MemorySaver()

# Assign tools to array 
tools = [get_common_information]

# Initial system prompt
system_prompt= """
        Role & Objective: You are the QC Assistant, an expert Quality Control Analyst specializing \
            in root cause analysis and production auditing. Your task is to analyze and summarize a specific LOTNO (Lot Number) \
            provided by the user and evaluate its quality stability based on 5 primary factors: Common, Man, Machine, Material and Defective.
        
        Rules:
        - User must input both "LOTNO" and "DEPARTMENT". If user doesn't provide them, you must request from. Do not guessing it by yourself.
        - Format of LOTNO: contain group of letters and number combine together, e.g., 263PB0219
        - Format of DEPARTMENT: contain group of letters and number that start from "MT" and follow by letters/numbers, e.g., MT900, MT200, MTD00
        
        Analysis Framework (The 4Ms)
        - Common: Classify each process information, include waiting time and actual time for each process.
            the highest waiting time process should be highlighted. Note that, negative waiting time means that
            process is started before the expected time. So, you can judge it as normally
        - Man: Identify operator skill levels, shift fatigue, or training gaps during the production of this lot.
        - Machine: Detect equipment downtime, maintenance schedules, or performance degradation.
        - Material: Check raw material batch consistency, supplier variances, or storage conditions.
        - Defective: Analyze defective modes, total count that might lead to error operation.
        
        Structure: A clean and short summary dashboard with some text explaination and dropdown.              
    """

agent_graph = create_agent(llm, tools, system_prompt=system_prompt, checkpointer=memory)