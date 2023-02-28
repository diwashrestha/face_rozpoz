import asyncio
import base64
import dash, cv2
import dash_html_components as html
import threading

from dash.dependencies import Output, Input
from quart import Quart, websocket
from dash_extensions import WebSocket


class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


# Setup small Quart server for streaming via websocket.
server = Quart(__name__)
delay_between_frames = 0.00  # add delay (in seconds) if CPU usage is too high


@server.websocket("/stream")
async def stream():
    camera = VideoCamera(0)  # zero means webcam
    while True:
        if delay_between_frames is not None:
            await asyncio.sleep(delay_between_frames)  # add delay if CPU usage is too high
        frame = camera.get_frame()
        await websocket.send(f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}")


# Create small Dash application for UI.
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Img(style={'width': '40%', 'padding': 10}, id="video1"),
    WebSocket(url=f"ws://127.0.0.1:5000/stream", id="ws")
])
# Copy data from websocket to Img element.
app.clientside_callback("function(m){return m? m.data : '';}", Output(f"video1", "src"), Input(f"ws", "message"))

if __name__ == '__main__':
    threading.Thread(target=app.run_server).start()
    server.run()