import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import asyncio
import base64
import dash, cv2
import threading
from dash.dependencies import Output, Input
from quart import Quart, websocket
from dash_extensions import WebSocket



# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

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



app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],)

navbar = dbc.NavbarSimple(
        dbc.Button(children=[html.I(className="fas fa-solid fa-bars")], outline=True, color="secondary", className="mr-1", id="btn_sidebar",style={"margin-left": "0px"}),
    brand="Brand",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)




sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-home me-2"), html.Span("Dashboard")],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-calendar-alt me-2"),
                        html.Span("Events"),
                    ],
                    href="/events",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-envelope-open-text me-2"),
                        html.Span("Admin"),
                    ],
                    href="/admin",
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div([
    html.Img(style={'width': '40%', 'padding': 10}, id="video1"),
    WebSocket(url=f"ws://127.0.0.1:5000/stream", id="ws"),
    html.Div(id = "page-content",style = CONTENT_STYLE)
])
# Copy data from websocket to Img element.
app.clientside_callback("function(m){return m? m.data : '';}", Output(f"video1", "src"), Input(f"ws", "message"))

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.P("This is the content of page 1!")
    elif pathname == "/events":
        return html.P("This is the content of page 2. Yay!")
    elif pathname == "/admin":
        return html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == "__main__":
    threading.Thread(target=app.run_server).start()
    server.run()
