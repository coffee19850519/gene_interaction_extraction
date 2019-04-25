import table as table
import xlrd as xlrd
import chardet
import traceback

def readExcelDataByName(fileName, sheetName):
    table = None
    errorMsg = None
    try:
        data = xlrd.open_workbook(fileName)
        table = data.sheet_by_name(sheetName)
    except Exception as msg:
        errorMsg = msg
    return table, errorMsg
def getColumnIndex(table, columnName):
     columnIndex = None
      #print table
     for i in range(table.ncols):
         #print columnName
         #print table.cell_value(0, i)
         if(table.cell_value(0, i) == columnName):
             columnIndex = i
             break
     return columnIndex
def load_genename_from_excl(filename):
    table,err=readExcelDataByName(filename,'Sheet1')
    rowNum=table.nrows
    L=[]
    for i in range(1,rowNum):
        gene= table.cell_value(i,0)
        L.append(gene)
    return L

if __name__=='__main__':
    filename=r'C:\Users\cuixin\Downloads\gene_dictionary.xlsx'
    L=load_genename_from_excl(filename)
    # table,err=readExcelDataByName(filename,'Sheet1')
    # rowNum=table.nrows
    # l=[]
    # for i in range(1,rowNum):
    #     gene= table.cell_value(i,0)
    #     l.append(gene)
    print(len(L))