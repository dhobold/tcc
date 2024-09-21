import os
import shutil

path_images = "Reports\Images\\"

# --------- metodos auxiliares --------------------------------
def deleteAllFilesInFolder(folder_path):
    try:
        # Verifica se o diretório existe
        if os.path.exists(folder_path):
            # Remove todos os arquivos e subdiretórios dentro da pasta especificada
            shutil.rmtree(folder_path)
            # Recria a pasta vazia
            os.makedirs(folder_path)
            print(f"Todos os arquivos na pasta '{folder_path}' foram deletados com sucesso.")
        else:
            print(f"A pasta '{folder_path}' não existe.")
    except Exception as e:
        print(f"Erro ao deletar arquivos: {e}")