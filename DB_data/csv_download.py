import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from requests import get
from tika import parser
import urllib.request
import logging
import sys
import os

def download_save_img(url, file_name):
    urllib.request.urlretrieve(url, file_name)

def download_pdf(url, file_name):
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)
        
def read_and_write(file_name, text_name):
    with open(text_name, "w", encoding='utf8') as writer:
        parsed = parser.from_file(file_name)
        text = parsed['content']
        lines = text.split('\n')
        full_text = []
        for line in lines:
            if line == '':
                continue
            else:
                writer.write(line.replace('�', ' ')+'\n')
                
def read_and_return(file_name):
    parsed = parser.from_file(file_name)
    text = parsed['content']
    lines = text.split('\n')
    full_text = ""
    for line in lines:
        if line == '':
            continue
        else:
            full_text+=(line.replace('�', ' ')+'\n')
    return full_text

def main():
    
    logging.basicConfig(filename='csv_download.log', filemode='w', level=logging.INFO)
    
    logging.info("Opening excel files...")
    """
    data_detail = pd.read_excel('csv/pill_details.xls')
    data_detail = data_detail[['품목일련번호', '허가일자', '원료성분', '마약류분류', '완제원료구분', '포장단위', '보험코드', '전문일반', '총량', '취소상태',
           '변경내용', '등록자id', '첨부문서', '제심사기간', '효능효과', '저장방법', '신약여부', '유효기간',
           '용법용량', '첨가제명', '성상', '재심사대상', '허가및신고구분',
           '취소일자', '표준코드', '주의사항', '주성분명']]

    data_list = pd.read_csv('csv/pill_list.csv')
    data_list = data_list[['품목일련번호', '품목명', '업소일련번호', '업소명', '큰제품이미지', '표시앞', '표시뒤', '의약품제형',
           '색상앞', '색상뒤', '분할선앞', '분할선뒤', '크기장축', '크기단축', '크기두께', '이미지생성일자(약학정보원)',
           '분류번호', '분류명', '전문일반구분', '품목허가일자', '제형코드명', '표기내용앞', '표기내용뒤', '표기이미지앞',
           '표기이미지뒤', '표기코드앞', '표기코드뒤', '변경일자']]

    new_Data = pd.merge(data_list, data_detail, on='품목일련번호')
    """
    new_Data = pd.read_csv("test_data_T2.csv",sep=',', error_bad_lines=False, encoding='euc-kr')

    img = '큰제품이미지'
    eff = '효능효과'
    use = '용법용량'
    war = '주의사항'

    row_num = len(new_Data.index)
    logging.info("total row number: %d", row_num)

    for idx in range(row_num):
        pill_id = new_Data.loc[idx]['품목일련번호']
        msg = str(idx)+" Pill ID: "+str(pill_id)+"-"*20
        logging.info(msg)
        pill_img = new_Data.loc[idx][img]
        pill_eff = new_Data.loc[idx][eff]
        pill_use = new_Data.loc[idx][use]
        pill_war = new_Data.loc[idx][war]

        ## 큰제품이미지
        try:
            file_name = str(pill_id)+'_img'
            download_save_img(pill_img, 'img/'+file_name+'.jpg')
            new_Data.iloc[idx,7] = file_name+'.jpg'
            logging.info("image Done.")
        except:
            logging.info("image FAIL")

        ## 효능효과
        try:
            file_name = str(pill_id)+'_effect'
            download_pdf(pill_eff, 'pdf/'+file_name+'.pdf')
            text = read_and_return('pdf/'+file_name+'.pdf')
            new_Data.iloc[idx,4] = text
            logging.info("effect Done.")
        except:
            logging.info("effect FAIL")
            
        if os.path.isfile('pdf/'+file_name+'.pdf'):
            os.remove('pdf/'+file_name+'.pdf')

        ## 용법용량
        try:
            file_name = str(pill_id)+'_usage'
            download_pdf(pill_use, 'pdf/'+file_name+'.pdf')
            text = read_and_return('pdf/'+file_name+'.pdf')
            new_Data.iloc[idx,5] = text
            logging.info("usage Done.")
        except:
            logging.info("usage FAIL")
            
        if os.path.isfile('pdf/'+file_name+'.pdf'):
            os.remove('pdf/'+file_name+'.pdf')

        ## 주의사항
        try:
            file_name = str(pill_id)+'_warning'
            download_pdf(pill_war, 'pdf/'+file_name+'.pdf')
            text = read_and_return('pdf/'+file_name+'.pdf')
            new_Data.iloc[idx,8] = text
            logging.info("warning Done.")
        except:
            logging.info("warning FAIL")
            
        if os.path.isfile('pdf/'+file_name+'.pdf'):
            os.remove('pdf/'+file_name+'.pdf')
            
    new_Data.to_csv("csv/test_data_T2.tsv", sep="\t")
    
    logging.info("Saving an csv file...")
    logging.info("Done")
    
if __name__ == "__main__":
    main()