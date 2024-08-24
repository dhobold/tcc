import ProgressBar as pb

#GERAR PDF
from fpdf import FPDF
def createPage(pdf,title,score):
    pdf.add_page()
    pdf.set_font('Times', 'B', 10)
    pdf.cell(40, 10, 'Analise da variável:',align='L')
    pdf.set_font('Times', 'B', 9)
    pdf.cell(90, 10,title,align='L')
    pdf.set_font('Times', 'B', 10)
    pdf.cell(0, 10,"Score: "+str(round(score,3)),align='R')
    
    WIDTH = 210
    HEIGHT = 297
    path = "C:\\Users\\pohed\\OneDrive\\DIH\\Pós FACENS\\TCC\\Reports\\"
    #path = "c:\\ftp\\Reports\\"
    pdf.image(path+"Images\\"+title+"_image1.jpg", 5, 20, WIDTH/2-10)
    pdf.image(path+"Images\\"+title+"_image2.jpg", WIDTH/2, 20, WIDTH/2-10)
    pdf.image(path+"Images\\"+title+"_image3.jpg", 5, 100, WIDTH/2-10)
    pdf.image(path+"Images\\"+title+"_image4.jpg", WIDTH/2, 100, WIDTH/2-10)


def Do(df_var):
    print("Inicio da geração do relatório")
    
    #Definir parametros da página PDF
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_font('Times', 'B', 16)

    #Quantas variaveis serão analisadas
    tam = len(df_var['variavel'])
    pb.printProgressBar(0, tam, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    #Gerar uma página de relatório para cada variavel
    for i in range(len(df_var['nivel_1'])): 
        if df_var['Score'].iloc[i] > 0 : 
            title = df_var['variavel'].iloc[i]
            score = df_var['Score'].iloc[i]
            createPage(pdf,title,score)
            pb.printProgressBar(i+1, tam, prefix = 'Progress:', suffix = 'Complete', length = 50)
    #pdf.output('c:\\ftp\\Reports\\relatorio.pdf')
    pdf.output('C:\\Users\\pohed\\OneDrive\\DIH\\Pós FACENS\\TCC\\Reports\\relatorio.pdf')
    print("Fim da geração do relatório")

