import datetime
import akshare as ak
import pandas as pd
from Base import i_sql, o_sql
import fake_useragent as ua
import asyncio
import logging
import aiohttp
from lxml import etree
import re
import numpy as np
from bs4 import BeautifulSoup
import collections
import pickle
u = ua.UserAgent()
cpi = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery1123015447681149269776_1674022590297&columns=REPORT_DATE%2CNATIONAL_BASE&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_CPI&_=1674022590298'
ppi = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112304356361556021564_1674027580327&columns=REPORT_DATE%2CBASE&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_PPI&_=1674027580328'
pmi = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112307234147573048613_1674553217806&columns=REPORT_DATE%2CMAKE_INDEX%2CNMAKE_INDEX&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_PMI&_=1674553217807'
currency = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery1123021969944596080904_1674554011650&columns=REPORT_DATE%2CBASIC_CURRENCY_SAME%2CCURRENCY_SAME%2CFREE_CASH_SAME&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_CURRENCY_SUPPLY&_=1674554011651'
export_import = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112306225288436653229_1674554743780&columns=REPORT_DATE%2CEXIT_BASE_SAME%2CIMPORT_BASE_SAME&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_CUSTOMS&_=1674554743781'
fdi = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112309882510387428938_1674566732718&columns=REPORT_DATE%2CACTUAL_FOREIGN_SAME&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_FDI&_=1674566732719'
loan = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112307306245787667975_1674567202414&columns=REPORT_DATE%2CRMB_LOAN_SAME&sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&reportName=RPT_ECONOMY_RMB_LOAN&_=1674567202415'
stock_account = 'https://datacenter-web.eastmoney.com/api/data/v1/get?callback=jQuery112307072105489517773_1679893517457&reportName=RPT_STOCK_OPEN_DATA&columns=STATISTICS_DATE%2CADD_INVESTOR&pageSize=1000&sortColumns=STATISTICS_DATE&sortTypes=-1&source=WEB&client=WEB&_=1679893517458'


async def trade_date():
    # 更新交易日历,daily
    tdf = ak.tool_trade_date_hist_sina()
    tdf['trade_date'] = pd.to_datetime(tdf['trade_date'], format="%Y-%m-%d")
    start = tdf.iloc[0].values[0].astype('M8[D]').astype('O')
    end = tdf.iloc[-1].values[0].astype('M8[D]').astype('O')
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days + 1)]
    df = pd.DataFrame({'trade_date': date_generated})
    df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y-%m-%d")
    tdf['trade'] = True
    df = pd.merge(df, tdf, how='outer', on='trade_date')
    df['trade'] = df['trade'].fillna(False)
    # print(df.iloc[-30:])
    await i_sql(df, 'trade_date')


async def industry_guide():
    url = 'https://data.eastmoney.com/cjsj/hyzs_list_EMI00018828.html'
    t = await request_page(url)
    soup = BeautifulSoup(t, 'html.parser')
    # 查找所有target="_self"的<a>标签，并提取出其中的文本信息
    elements = soup.find('ul', class_='hyzs at').find_all('a', target='_self')
    guides = collections.defaultdict(list)
    for element in elements:
        print(element.text)
        if element.text == '农林牧渔':
            guides[element.text] += ['agricultural_index', 'agricultural_price']
    filename = 'industry_guide.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(guides, f)


async def request_page(url):
    logging.info('scraping %s', url)
    async with aiohttp.ClientSession() as session:
        response = await session.get(url, headers={"User-Agent": u.random})
        return await response.text()


async def xprocess(text: str, xpath: str):
    html = etree.HTML(text)
    url_list = html.xpath(xpath)
    return url_list


async def cpi_get():
    # monthly
    t = await request_page(cpi)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"NATIONAL_BASE\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 120
    assert len(result) == len(result2), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'data': result2})
    await i_sql(df, 'cpi')


async def ppi_get():
    # monthly
    t = await request_page(ppi)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"BASE\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 120
    assert len(result) == len(result2), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'data': result2})
    # print(df)
    await i_sql(df, 'ppi')


async def pmi_get():
    # monthly
    t = await request_page(pmi)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"\"MAKE_INDEX\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    r = r"\"NMAKE_INDEX\":(.*?)}"
    result3 = re.findall(r, t)
    result3 = np.array(result3).astype(float)
    # result3 = result3 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    assert len(result) == len(result3), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'MAKE': result2, "NMAKE": result3})
    await i_sql(df, 'pmi')


async def currency_get():
    # monthly
    t = await request_page(currency)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"\"BASIC_CURRENCY_SAME\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    r = r"\"CURRENCY_SAME\":(.*?),"
    result3 = re.findall(r, t)
    result3 = np.array(result3).astype(float)
    # result3 = result3 / 100
    r = r"\"FREE_CASH_SAME\":(.*?)}"
    result4 = re.findall(r, t)
    result4 = np.array(result4).astype(float)
    # result4 = result4 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    assert len(result) == len(result3), '时间长度与数据不相等'
    assert len(result) == len(result4), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'M2': result2, "M1": result3, "M0": result4})
    # print(df)
    await i_sql(df, 'currency')


async def export_import_get():
    # monthly
    t = await request_page(export_import)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"EXIT_BASE_SAME\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    r = r"IMPORT_BASE_SAME\":(.*?)}"
    result3 = re.findall(r, t)
    result3 = np.array(result3).astype(float)
    # result3 = result3 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    assert len(result) == len(result3), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'exit': result2, 'import': result3})
    # print(df)
    await i_sql(df, 'ex_import')


async def fdi_get():
    # monthly
    t = await request_page(fdi)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"ACTUAL_FOREIGN_SAME\":(.*?)}"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'data': result2})
    # print(df)
    await i_sql(df, 'fdi')


async def loan_get():
    # monthly
    t = await request_page(loan)
    # print(t)
    # t = await xprocess(t, '//text()')
    # print(t)
    r = r"\"REPORT_DATE\":\"(.*?) "
    result = re.findall(r, t)
    r = r"RMB_LOAN_SAME\":(.*?)}"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'data': result2})
    # print(df)
    await i_sql(df, 'loan')


async def stock_account_get():
    # monthly
    t = await request_page(stock_account)
    r = r"\"STATISTICS_DATE\":\"(.*?)\""
    result = re.findall(r, t)
    r = r"ADD_INVESTOR\":(.*?),"
    result2 = re.findall(r, t)
    result2 = np.array(result2).astype(float)
    # result2 = result2 / 100
    assert len(result) == len(result2), '时间长度与数据不相等'
    df = pd.DataFrame({'date': result, 'data': result2})
    # print(df)
    await i_sql(df, 'stock_account')


async def agricultural_index():
    # daily
    urls = []
    for i in range(1, 5):
        urls.append(f'https://datacenter-web.eastmoney.com/api/data/v1/get?callback='
                    f'jQuery1123040151091890341917_1679896279089&reportName=RPT_INDUSTRY_INDEX&pageNumber={i}&'
                    f'columns=INDICATOR_VALUE%2CREPORT_DATE&filter=(INDICATOR_ID=%22EMI00662543%22)&'
                    f'sortColumns=REPORT_DATE&sortTypes=-1&source=WEB&client=WEB&_=1679896279088')
    dates = []
    results = []
    r1 = r"REPORT_DATE\":(.*?) "
    r2 = r"INDICATOR_VALUE\":(.*?),"
    for i in urls:
        t = await request_page(i)
        dates += re.findall(r1, t)
        results += re.findall(r2, t)
    results = np.array(results).astype(float)
    # result2 = result2 / 100
    assert len(dates) == len(results), '时间长度与数据不相等'
    df = pd.DataFrame({'date': dates, 'data': results})
    # print(df)
    await i_sql(df, 'agricultural_index')


async def agricultural_price():
    # daily,农产品价格批发指数
    urls = []
    for i in range(1, 5):
        urls.append(f'https://datacenter-web.eastmoney.com/api/data/v1/get?'
                    f'callback=jQuery112308348905545744743_1679898820910&'
                    f'reportName=RPT_INDUSTRY_INDEX&pageNumber={i}&columns=INDICATOR_VALUE%2CREPORT_DATE&'
                    f'filter=(INDICATOR_ID=%22EMI00009274%22)&sortColumns=REPORT_DATE&'
                    f'sortTypes=-1&source=WEB&client=WEB&_=1679898820915')
    dates = []
    results = []
    r1 = r"REPORT_DATE\":(.*?) "
    r2 = r"INDICATOR_VALUE\":(.*?),"
    for i in urls:
        t = await request_page(i)
        dates += re.findall(r1, t)
        results += re.findall(r2, t)
    results = np.array(results).astype(float)
    # result2 = result2 / 100
    assert len(dates) == len(results), '时间长度与数据不相等'
    df = pd.DataFrame({'date': dates, 'data': results})
    # print(df)
    await i_sql(df, 'agricultural_price')


corountine = industry_guide()
task = asyncio.ensure_future(corountine)
loop = asyncio.get_event_loop()
loop.run_until_complete(task)
# print(u.random)
filename = 'industry_guide.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
print(data)

