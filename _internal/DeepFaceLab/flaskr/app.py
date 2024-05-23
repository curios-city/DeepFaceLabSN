from pathlib import Path

from flask import Flask, send_file, Response, render_template, render_template_string, request, g
from flask_socketio import SocketIO, emit
import logging


def create_flask_app(s2c, c2s, s2flask, kwargs):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    log = logging.getLogger('werkzeug')
    log.disabled = True
    model_path = Path(kwargs.get('saved_models_path', ''))
    filename = 'preview.png'
    preview_file = str(model_path / filename)

    def gen():
        frame = open(preview_file, 'rb').read()
        while True:
            try:
                frame = open(preview_file, 'rb').read()
            except:
                pass
            yield b'--frame\r\nContent-Type: image/png\r\n\r\n'
            yield frame
            yield b'\r\n\r\n'

    def send(queue, op):
        queue.put({'op': op})

    def send_and_wait(queue, op):
        while not s2flask.empty():
            s2flask.get()
        queue.put({'op': op})
        while s2flask.empty():
            pass
        s2flask.get()

    @app.route('/save', methods=['POST'])
    def save():
        send(s2c, 'save')
        return '', 204

    @app.route('/exit', methods=['POST'])
    def exit():
        send(c2s, 'close')
        # 由于退出整个程序，应该不需要这个权限。request.environ.get('werkzeug.server.shutdown')()
        return '', 204

    @app.route('/update', methods=['POST'])
    def update():
        send(c2s, 'update')
        return '', 204

    @app.route('/next_preview', methods=['POST'])
    def next_preview():
        send(c2s, 'next_preview')
        return '', 204

    @app.route('/change_history_range', methods=['POST'])
    def change_history_range():
        send(c2s, 'change_history_range')
        return '', 204

    @app.route('/zoom_prev', methods=['POST'])
    def zoom_prev():
        send(c2s, 'zoom_prev')
        return '', 204

    @app.route('/zoom_next', methods=['POST'])
    def zoom_next():
        send(c2s, 'zoom_next')
        return '', 204

    @app.route('/')
    def index():
        return render_template('index.html')

    # @app.route('/preview_image')
    # def preview_image():
    #     return Response(gen(), mimetype='multipart/x-mixed-replace;boundary=frame')

    @app.route('/preview_image')
    def preview_image():
        return send_file(preview_file, mimetype='image/png')

    socketio = SocketIO(app)

    @socketio.on('connect', namespace='/')
    def test_connect():
        emit('my response', {'data': 'Connected'})

    @socketio.on('disconnect', namespace='/test')
    def test_disconnect():
        print('Client disconnected')

    return socketio, app






