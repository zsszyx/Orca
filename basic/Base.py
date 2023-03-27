import pandas as pd


async def i_sql(df=pd.DataFrame, code=''):
    # 数据库写入表
    df.to_hdf('data.h5', key=code, format='table')


async def o_sql(code=None):
    # 数据库输出表
    df = pd.read_hdf('data.h5', key=code)
    return df
