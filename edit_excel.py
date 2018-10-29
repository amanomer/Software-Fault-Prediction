import xlrd
import xlwt
loc = ("D:/CM1/")
rdfile = ("final1db.xlsx")
wtfile = ("final2db.xls")
wb = xlrd.open_workbook(loc+rdfile)
sheet = wb.sheet_by_index(0)
nwb = xlwt.Workbook()
nsheet = nwb.add_sheet('CM')
#sheet.cell_value(0,0)
#print(sheet.nrows)
rows = sheet.nrows
cols = sheet.ncols
#print(sheet.cell(0,cols-1).value)

for x in range(rows):
    for y in range(cols):
        word = sheet.cell(x,y).value
        if y == 0:
            word = word[1:]
        elif y == cols-1:
            word = word[:-1]
        nsheet.write(x,y,word)
        
nwb.save(loc+wtfile)