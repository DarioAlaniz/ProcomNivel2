#back-end desarrolla toda la conexion web que no es visible 
import argparse
import asyncio
import subprocess
import json
import logging
import os
import ssl
import uuid
import numpy as np
import os
import cv2
# import sysv_ipc
import time

from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from shapeDetection import ShapeDetector
import fileKernels as filter

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

def auto_canny(image,sigma=0.33):
    #obtengo la media de intencidades de los pixeles
    m       = np.median(image) 
    # construyo el valor bajo y alto del umbral en base a un porcentaje controlado por sigma
    # bajo valor de sigma indica un menor umbral
    # alto valor de sigma indica un mayor umbral
    low     = int(max(0,(1.0-sigma)*m)) 
    upper   = int(min(255,(1.0+sigma)*m))
    edged   = cv2.Canny(image,low,upper)
    return edged

############################################
# Class Shape Detector
sd = ShapeDetector()
############################################
# Dictionary of different kernels
kernels = { 1       : filter.smallBlur,
            2       : filter.largeBlur,
            3       : filter.sharpen,
            3       : filter.laplacian,
            4       : filter.edge_detect,
            5       : filter.edge_detect2,
            6       : filter.sobelX,
            7       : filter.sobelY }


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, rows, cols, kernel):
        super().__init__()  # don't forget this!
        self.track = track
        self.rows = rows
        self.cols = cols
        self.kernel = kernel
        self.edgeMatrix = np.array((
	    [-1, -1, -1],
	    [-1,  8, -1],
	    [-1, -1, -1]), dtype="int")

    async def recv(self):
        frame = await self.track.recv() #espera a que llegue un frame, cuando llega lo procesa. Por eso la funcion es asincrona ya que la comunicacion sigue corriendo por dentras
 
        img  = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if(self.kernel >= 1 and self.kernel<=8):
            gauss = cv2.GaussianBlur(gray, (5,5), 0)
            img = cv2.filter2D(gauss,-1,kernels[self.kernel])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #devuelve el formato a img
        elif(self.kernel == 9):   
            gauss = cv2.GaussianBlur(gray, (5,5), 0)
            canny = cv2.Canny(gauss,50,150)
            (countours,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,countours,-1,(255,0,0), 1)
        elif(self.kernel == 10):
            gauss = cv2.GaussianBlur(gray, (5,5), 0)
            autoCanny = auto_canny(gauss)
            (countours,_) = cv2.findContours(autoCanny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,countours,-1,(255,0,0), 1)
        elif(self.kernel == 11):
            #valores de thresholdo y area obtenido para cierto ambiento de una imagen obtenida de internet, se debe setear para distintos valores dependiendo el entorno
            _, thrash   = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
            countours,_ = cv2.findContours(thrash,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            for countour in countours:
                shape,approx = sd.detect(countour)
                area= cv2.contourArea(approx)
                if  295 <= area <=38721:
                    x,y,w,h = cv2.boundingRect(approx)
                    cv2.drawContours(img,[approx],0,(255,0,0),2)
                    #coordenads para escribir el texto    
                    M = cv2.moments(countour) #momento de una imagen: promedio ponderado de la intencidad de pixeles 
                    if M["m00"] != 0:
                        #Calculo en centro de una imagen
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                    else:
                        cx = approx.ravel()[0]
                        cy = approx.ravel()[1]  
                    cv2.putText(img,shape, (cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            
        new_frame = VideoFrame.from_ndarray(img, format="bgr24") #vuelve a dar el formato a la imagen 
        new_frame.pts = frame.pts #conserva el orden de los frame que tiene mi frame original
        new_frame.time_base = frame.time_base #

        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index2.html"), "r").read()#utiliza el html como modo de lectura
    return web.Response(content_type="text/html", text=content) #responde al cliente que se quiere conectar con el archivos html, 


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()#usa el .js como modo de lectura
    return web.Response(content_type="application/javascript", text=content)#responde al cliente que se quiere conectar con el archivo cliente


async def offer(request):
    params = await request.json() #espera el json que se obtiene de .js por medio del html
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"]) #genera la vinculacion

    pc = RTCPeerConnection() #vinculacion punto a punto
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)


    @pc.on("track") #cuando ya se establecio la vinculacion, llama a un funcion "on_track" 
    def on_track(track): #deja de ser asincrono el procesamiento por lo que el tiempo es importante
        log_info("Track %s received", track.kind)
        
        if track.kind == "video": #si tengo contenido de video, donde el track tiene el frame 
            print(params["kernel_ref"]) #muestro el kernel obtenido
            local_video = VideoTransformTrack(
                track, rows=int(params["width_height"].split('x')[1]) , cols=int(params["width_height"].split('x')[0]), kernel=int(params["kernel_ref"])
            )# creo un objeto VideoTransforTrack con sus parametros que obtengo del .json que genera el .js por medio del .html

            pc.addTrack(local_video) #devuelvo el frame procesador 

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app): #limpia las conecciones 
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG) #muestra mayor informacion de la conexion 
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:#carga los certificados
        ssl_context = ssl.SSLContext() #vincula los certificados en python
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application() #Habilito la aplicacion Web
    app.on_shutdown.append(on_shutdown) #vincula la funcion para limpiar la conexion
    #################
    #Defino directorios
    app.router.add_get("/", index) #trabajo con el root 
    app.router.add_get("/client.js", javascript) #.js para vincular
    app.router.add_post("/offer", offer) #levanta la gestion del cliente que se quiere conectar
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context) ##Lanzo la aplicacion y servidor
