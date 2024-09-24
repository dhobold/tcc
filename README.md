# AnomalyAnalysis
Este repositório contém um projeto focado em Análise de Anomalias, com dois principais modos de execução: um modo de linha de comando e um modo de servidor. O projeto pode ser executado e depurado utilizando o Visual Studio Code, conforme especificado no arquivo de configuração launch.json.

## Estrutura do Projeto
AnomalyAnalysis/TCC.py: Script principal do TCC, que pode ser executado via linha de comando.
AnomalyAnalysis/app.py: Script do servidor que também inclui o frontend, para execução no modo servidor.
## Requisitos
Certifique-se de ter os seguintes itens instalados:

## Python 3.x
VSCode com a extensão Python e o debugpy configurado
Bibliotecas necessárias (geralmente listadas em um arquivo requirements.txt)
## Instalação de dependências
Execute o seguinte comando para instalar as dependências do projeto:

pip install -r requirements.txt


## Modos de Execução
O projeto pode ser executado de duas maneiras, conforme descrito nas configurações do VSCode.

## 1. Modo Linha de Comando (TCC.py)
Este modo executa a análise no terminal, aceitando parâmetros via linha de comando.

## Como executar:

Abra o VSCode.
Selecione o modo TCC - cmd no depurador.
Inicie a depuração.
Ou execute o seguinte comando no terminal diretamente:

python AnomalyAnalysis/TCC.py --type cmd


## 2. Modo Servidor + Frontend (app.py)
Este modo inicia um servidor web que também serve um frontend para interagir com a aplicação.

## Como executar:

Abra o VSCode.
Selecione o modo TCC - Server + Front no depurador.
Inicie a depuração.
Ou execute o seguinte comando no terminal diretamente:

python AnomalyAnalysis/app.py --type server

O servidor será iniciado, e você poderá acessá-lo via um navegador web.

## Debug no VSCode
O projeto está configurado para ser facilmente depurado utilizando o Visual Studio Code.

Certifique-se de que o Python e o debugpy estão configurados corretamente.
Use o depurador do VSCode para iniciar o projeto em qualquer um dos modos disponíveis.
Pontos de interrupção podem ser definidos normalmente no código para análise detalhada.

## Configurações do VSCode (launch.json)
O arquivo launch.json já está configurado para dois modos de depuração:

TCC - cmd: Executa o arquivo TCC.py no modo de linha de comando.
TCC - Server + Front: Executa o arquivo app.py no modo servidor.

## Contribuições
Contribuições são bem-vindas! Para relatar bugs ou sugerir melhorias, por favor, abra uma issue neste repositório.

## Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
