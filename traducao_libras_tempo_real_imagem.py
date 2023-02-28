'''Função: Tradução de LIBRAS por captação de 
de fotos da câmera em tempo real

Este arquivo tem como finalidade realizar a 
tradução da LIBRAS por imagens captadas através 
da câmera.

A tradução é realizada com o auxilio dos extratores
e de algumas bibliotecas'''

#Importando as bilbiotecas
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import io
from PIL import Image

#Criando a função para tirar a foto através da WebCam
VIDEO_HTML = """
<video autoplay
 width=%d height=%d style='cursor: pointer;'></video>
<script>
 
var video = document.querySelector('video')
 
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream=> video.srcObject = stream)
  
var data = new Promise(resolve=>{
  video.onclick = ()=>{
    var canvas = document.createElement('canvas')
    var [w,h] = [video.offsetWidth, video.offsetHeight]
    canvas.width = w
    canvas.height = h
    canvas.getContext('2d')
          .drawImage(video, 0, 0, w, h)
    video.srcObject.getVideoTracks()[0].stop()
    video.replaceWith(canvas)
    resolve(canvas.toDataURL('image/jpeg', %f))
  }
})
</script>
"""
def take_photo(filename='photo.jpg', quality=2, size=(400,300)):
  display(HTML(VIDEO_HTML % (size[0],size[1],quality)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  f = io.BytesIO(binary)
  return np.asarray(Image.open(f))

#Tirando a foto
cap = take_photo() #clicar na imagem da webcam para tirar uma foto

#Importando as bilbiotecas para fazer o reconhecimento
#e a verificação da imagem
import time
import sys
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Importando os arquivos das pastas base de dados e extratores
pose_path = '../Base_De_Dados/Pose'
imagens_path = '../Base_De_Dados/Imagens'
modulos_path = '../Extratores'

#Importando os módulos e adicionando ao sys
sys.path.append('../Extratores')
print(sys.path)

#Importando os extratores e as referências
import extrator_POSICAO as posicao
import extrator_ALTURA as altura
import extrator_PROXIMIDADE as proximidade
import alfabeto

#Carregando o modelo e as estruturas da rede neural pré treinada
arquivoProto = "../Base_De_Dados/Pose/hand/pose_deploy.prototxt"
modeloCaffe = "..Base_De_Dados/Pose/hand/pose_iter_102000.caffemodel"
nPontos = 22
PARES_POSE = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20]]

letras = ['A', 'B', 'C', 'D', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W']

#Lendo o modelo e carregando a importação de módulos
net = cv2.dnn.readNetFromCaffe(arquivoProto, modeloCaffe)

limite=0.1

frame = cap
frameCopia = np.copy(frame)
frameLargura = frame.shape[1]
frameAltura = frame.shape[0]
janela = frameLargura / frameAltura

corPonto_A, corPonto_B, corLinha, corTxtPonto, corTxtAprov, corTxtWait = (14, 201, 255), (255, 0, 128), (192, 192, 192), \
                                                                         (10, 216, 245), (255, 0, 128), (192, 192, 192)
tamFont, tamLine, tamCircle, espessura = 2, 1, 4, 2
fonte = cv2.FONT_HERSHEY_SIMPLEX

t = time.time()
entradaAltura = 368
entradaLargura = int(((janela * entradaAltura) * 8) // 8)

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (entradaLargura, entradaAltura), 
(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
saida = net.forward()
print("Tempo da Rede: {:.2f}sec".format(time.time() - t))

pontos = []

tamanho = cv2.resize(frame, (frameLargura, frameAltura))
mapaSuave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
fundo = np.uint8(mapaSuave > limite)


for i in range(nPontos):
    mapaConfianca = saida[0, i, :, :]
    mapaConfianca = cv2.resize(mapaConfianca, (frameLargura, frameAltura))

    minVal, confianca, minLoc, ponto = cv2.minMaxLoc(mapaConfianca)

    if confianca > limite:
        cv2.circle(frameCopia, (int(ponto[0]), int(ponto[1])), 5, corPonto_A, thickness=espessura, lineType=cv2.FILLED)
        cv2.putText(frameCopia, ' ' + (str(int(ponto[0]))) + ',' + str(int(ponto[1])), (int(ponto[0]), int(ponto[1])),
                    fonte, 0.3, corTxtAprov, 0, lineType=cv2.LINE_AA)
        
        cv2.circle(frame, (int(ponto[0]), int(ponto[1])), tamCircle, corPonto_A, thickness=espessura,
                   lineType=cv2.FILLED)
        cv2.putText(frame, ' ' + "{}".format(i), (int(ponto[0]), int(ponto[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    corTxtAprov,
                    0, lineType=cv2.LINE_AA)
        
        cv2.circle(fundo, (int(ponto[0]), int(ponto[1])), tamCircle, corPonto_A, thickness=espessura,
                   lineType=cv2.FILLED)
        cv2.putText(fundo, ' ' + "{}".format(i), (int(ponto[0]), int(ponto[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    corTxtAprov,
                    0, lineType=cv2.LINE_AA)

        pontos.append((int(ponto[0]), int(ponto[1])))

    else:
        pontos.append((0, 0))


for par in PARES_POSE:
    partA = par[0]
    partB = par[1]

    if pontos[partA] != (0, 0) and pontos[partB] != (0, 0):
        cv2.line(frameCopia, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)
        cv2.line(frame, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)        
        cv2.line(fundo, pontos[partA], pontos[partB], corLinha, tamLine, lineType=cv2.LINE_AA)

#Verificando a posição dos dedos
posicao.posicoes = []
#Dedo polegar
posicao.verificar_posicao_DEDOS(pontos[1:5], 'polegar', altura.verificar_altura_MAO(pontos))
#Dedo indicador
posicao.verificar_posicao_DEDOS(pontos[5:9], 'indicador', altura.verificar_altura_MAO(pontos))
#Dedo médio
posicao.verificar_posicao_DEDOS(pontos[9:13], 'medio', altura.verificar_altura_MAO(pontos))
#Dedo anelar
posicao.verificar_posicao_DEDOS(pontos[13:17], 'anelar', altura.verificar_altura_MAO(pontos))
#Dedinho
posicao.verificar_posicao_DEDOS(pontos[17: 21], 'minimo', altura.verificar_altura_MAO(pontos))

#Exibindo a posição dos dedos
print(posicao.posicoes)

#Verificando a proximidade entre os dedos
p = proximidade.verificar_proximidade_DEDOS(pontos)

#Plotando os resultados da verificação de proximidade
print(p)

#Comparando as características com a base de dados
#i = indice
#a = valores da lista
for i, a in enumerate(alfabeto.letras):
  if proximidade.verificar_proximidade_DEDOS(pontos) == alfabeto.letras[i]:

    cv2.putText(imagem, ' ' + letras[i], (50, 50), fonte, 1, cor_txtponto,
                tamanho_linha, lineType = cv2.LINE_AA)

#Exibindo as saídas
plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))