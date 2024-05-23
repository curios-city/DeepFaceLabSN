import os
import sys
import traceback
import queue
import threading
import time
from enum import Enum
import webbrowser
import numpy as np
import itertools
from pathlib import Path
from core import imagelib
import cv2
import models
from core.interact import interact as io
import logging
import os

COLAB_TRAIN_STOPPER_FILENAME = 'stopper.txt'

class GlobalMeanLoss:
    def __init__(self):
        self.src = "未记录"
        self.dst = "未记录"
        
# adapted from https://stackoverflow.com/a/52295534
class TensorBoardTool:
    def __init__(self, dir_path):
        self.dir_path = dir_path
    def run(self):
        from tensorboard import default
        from tensorboard import program
        from tensorboard import version as tb_version

        # remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins())
        tb_argv = [None, '--logdir', self.dir_path, '--host', '0.0.0.0', '--port', '6006']

        #if int(tb_version.VERSION[0])>=2:
            #tb_argv.append("--bind_all")
        tb.configure(argv=tb_argv)
        tb.launch()


def process_img_for_tensorboard(input_img):
    # convert format from bgr to rgb
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # adjust axis to put channel count at the beginning
    img = np.moveaxis(img, -1, 0)
    return img

def log_tensorboard_previews(iter, previews, folder_name, train_summary_writer):
    for preview in previews:
        (preview_name, preview_bgr) = preview
        preview_rgb = process_img_for_tensorboard(preview_bgr)
        train_summary_writer.add_image('{}/{}'.format(folder_name, preview_name), preview_rgb, iter)

def log_tensorboard_model_previews(iter, model, train_summary_writer):
    log_tensorboard_previews(iter, model.get_previews(), 'preview', train_summary_writer)
    log_tensorboard_previews(iter, model.get_static_previews(), 'static_preview', train_summary_writer)

def trainerThread (s2c, c2s, e,
                    socketio=None,
                    model_class_name = None,
                    saved_models_path = None,
                    training_data_src_path = None,
                    training_data_dst_path = None,
                    pretraining_data_path = None,
                    pretrained_model_path = None,
                    no_preview=False,
                    force_model_name=None,
                    force_gpu_idxs=None,
                    cpu_only=None,
                    silent_start=False,
                    execute_programs = None,
                    debug=False,
                    tensorboard_dir=None,
                    start_tensorboard=False,
                    config_training_file=None,
                    gen_snapshot=False,
                    **kwargs):
    
    global global_mean_loss
    global_mean_loss = GlobalMeanLoss()
    
    while True:
        try:
            start_time = time.time()

            save_interval_min = kwargs.get('saving_time', 25)
            tensorboard_preview_interval_min = 5

            if not training_data_src_path.exists():
                training_data_src_path.mkdir(exist_ok=True, parents=True)

            if not training_data_dst_path.exists():
                training_data_dst_path.mkdir(exist_ok=True, parents=True)

            if not saved_models_path.exists():
                saved_models_path.mkdir(exist_ok=True, parents=True)
                            
            model = models.import_model(model_class_name)(
                        is_training=True,
                        saved_models_path=saved_models_path,
                        training_data_src_path=training_data_src_path,
                        training_data_dst_path=training_data_dst_path,
                        pretraining_data_path=pretraining_data_path,
                        pretrained_model_path=pretrained_model_path,
                        src_pak_name=kwargs.get('src_pak_name', None),
                        dst_pak_name=kwargs.get('dst_pak_name', None),
                        no_preview=no_preview,
                        force_model_name=force_model_name,
                        force_gpu_idxs=force_gpu_idxs,
                        cpu_only=cpu_only,
                        silent_start=silent_start,
                        config_training_file=config_training_file,
                        auto_gen_config=kwargs.get("auto_gen_config", False),
                        debug=debug,
                        reduce_clutter= kwargs.get('reduce_clutter', False)
                    )

            is_reached_goal = model.is_reached_iter_goal()

            if tensorboard_dir is not None:
                c2s.put({ 
                    'op': 'tb', 
                    'action': 'init', 
                    'model_name': model.model_name,
                    'tensorboard_dir': tensorboard_dir,
                    'start_tensorboard': start_tensorboard
                })

            shared_state = { 'after_save' : False }
            shared_state = {'after_save': False}
            loss_string = ""
            save_iter = model.get_iter()

            def model_save():
                if not debug and not is_reached_goal:
                    io.log_info("保存中....", end='\r')
                    model.save()
                    shared_state['after_save'] = True

            def model_backup():
                if not debug and not is_reached_goal:
                    model.create_backup()

            def read_stopping_file():
                path = Path(saved_models_path / COLAB_TRAIN_STOPPER_FILENAME)
                if not os.path.exists(path):
                    write_stopping_file('false')
                    return False
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        return True if f.read() == 'true' else False

            def write_stopping_file(value):
                path = Path(saved_models_path / COLAB_TRAIN_STOPPER_FILENAME)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(value)
                    
            def log_step(step, step_time, src_loss, dst_loss):
                c2s.put({ 
                    'op': 'tb', 
                    'action': 'step', 
                    'step': step,
                    'step_time': step_time,
                    'src_loss': src_loss,
                    'dst_loss': dst_loss
                })
            
            def log_previews(step, previews, static_previews):
                c2s.put({
                    'op': 'tb',
                    'action': 'preview',
                    'step': step,
                    'previews': previews,
                    'static_previews': static_previews
                })

            def send_preview():
                if not debug:
                    previews = model.get_previews()
                    c2s.put({'op': 'show', 'previews': previews, 'iter': model.get_iter(),
                             'loss_history': model.get_loss_history().copy()})
                else:
                    previews = [('debug, press update for new', model.debug_one_iter())]
                    c2s.put({'op': 'show', 'previews': previews})
                e.set()  # Set the GUI Thread as Ready

            def open_browser(url):
                webbrowser.open(url)
                
            if model.get_target_iter() != 0:
                if is_reached_goal:
                    io.log_info('模型已经训练到目标迭代。您可以使用预览功能.')
                else:
                    io.log_info('开始运行中. 目标迭代: %d. 按下 Enter 停止训练并保存模型.' % (
                        model.get_target_iter()))
            else:
                io.log_info('')
                io.log_info('启动中.....')
                io.log_info('按 Enter 停止训练并保存进度')
                io.log_info('按 Space 可以切换视图')
                io.log_info('按 P 可以刷新预览图')
                io.log_info('按 S 可以保存训练进度')
                io.log_info('')
 

                if flask_preview == True:
                    io.log_info('请在浏览器进入：127.0.0.1:6006')
                    io.log_info('若是在服务器训练，可远程访问！')
                    io.log_info('')
                    url ='http://127.0.0.1:6006/'
                    thread_browser = threading.Timer(15, open_browser, args=[url])
                    thread_browser.start()
                    if start_tensorboard == True:
                        io.log_info('抱歉！两种WEBUI无法同时启用！实时WEB预览优先')       
                        io.log_info('')
                elif start_tensorboard == True:
                    io.log_info('（30秒后自动打开）或在浏览器进入：127.0.0.1:6006')
                    io.log_info('请不要打开右上方的下拉菜单，只需使用左上方的三个面板！')
                    io.log_info('首次开启WEB面板，或者刚删过log数据，需要训练五分钟以上，保存数据后才会显示训练内容')
                    io.log_info('')
                    url ='http://127.0.0.1:6006/?darkMode=true#timeseries'
                    thread_browser = threading.Timer(30, open_browser, args=[url])
                    thread_browser.start()       
                    
                io.log_info('[保存时间][迭代次数][单次迭代][SRC损失][DST损失]')


                
            last_save_time = time.time()
            last_preview_time = time.time()

            execute_programs = [[x[0], x[1], time.time()] for x in execute_programs]

            for i in itertools.count(0, 1):
                if not debug:
                    cur_time = time.time()

                    for x in execute_programs:
                        prog_time, prog, last_time = x
                        exec_prog = False
                        if 0 < prog_time <= (cur_time - start_time):
                            x[0] = 0
                            exec_prog = True
                        elif prog_time < 0 and (cur_time - last_time) >= -prog_time:
                            x[2] = cur_time
                            exec_prog = True

                        if exec_prog:
                            try:
                                exec(prog)
                            except Exception as e:
                                print("无法执行程序: %s" % prog)

                    if not is_reached_goal:

                        if model.is_first_run():
                            io.log_info(
                                "尝试进行第一次迭代。如果发生错误，请减少模型参数.")
                            if sys.platform[0:3] == 'win':
                                io.log_info("按下Enter键可停止训练并保存模型.")

                        if gen_snapshot:
                            model.generate_training_state()
                            break

                        iter, iter_time = model.train_one_iter()

                        loss_history = model.get_loss_history()
                        time_str = time.strftime("[%H:%M:%S]")
                        if iter_time >= 10:
                            loss_string = "{0}[#{1:06d}][{2:.5s}s]".format(time_str, iter, '{:0.4f}'.format(iter_time))
                        else:
                            loss_string = "{0}[#{1:06d}][{2:04d}ms]".format(time_str, iter, int(iter_time * 1000))

                        if shared_state['after_save']:
                            shared_state['after_save'] = False

                            mean_loss = np.mean(loss_history[save_iter:iter], axis=0)
                            
                            global_mean_loss.src="[{:.4f}]".format(mean_loss[0])
                            global_mean_loss.dst="[{:.4f}]".format(mean_loss[1])
                            
                            for loss_value in mean_loss:
                                loss_string += "[%.4f]" % (loss_value)

                            io.log_info(loss_string)

                            save_iter = iter
                        else:
                            for loss_value in loss_history[-1]:
                                loss_string += "[%.4f]" % (loss_value)

                            if io.is_colab():
                                io.log_info('\r' + loss_string, end='')
                            else:
                                io.log_info(loss_string, end='\r')

                        if socketio is not None:
                            socketio.emit('loss', loss_string)

                        loss_entry = loss_history[-1]
                        log_step(iter, iter_time, loss_entry[0], loss_entry[1] if len(loss_entry) > 1 else None)

                        if model.get_iter() == 1 and not model.reset_training:
                            model_save()

                        # if model.get_iter() % 5000 == 0:
                        #     print ('Doing a training analysis.')
                        #     model.generate_training_state()

                        if model.get_target_iter() != 0 and model.is_reached_iter_goal():
                            io.log_info('达到目标迭代.')
                            model_save()
                            is_reached_goal = True
                            io.log_info('可以开始使用快捷键 P 刷新预览.')

                if not is_reached_goal and (time.time() - last_preview_time) >= tensorboard_preview_interval_min*60:
                    last_preview_time += tensorboard_preview_interval_min*60
                    previews = model.get_previews()
                    static_previews = model.get_static_previews()
                    log_previews(iter, previews, static_previews)

                if not is_reached_goal and (time.time() - last_save_time) >= save_interval_min*60:
                    last_save_time += save_interval_min*60
                    model_save()
                    send_preview()

                if io.is_colab():
                    if read_stopping_file():
                        io.log_info('由于停止文件，停止训练!')
                        write_stopping_file('false')
                        s2c.put({'op': 'close'})

                if i == 0:
                    if is_reached_goal:
                        model.pass_one_iter()
                    send_preview()

                if debug:
                    time.sleep(0.005)

                while not s2c.empty():
                    item = s2c.get()
                    op = item['op']
                    if op == 'save':
                        model_save()
                    elif op == 'backup':
                        model_backup()
                    elif op == 'preview':
                        if is_reached_goal:
                            model.pass_one_iter()
                        send_preview()
                    elif op == 'close':
                        model_save()
                        i = -1
                        break

                if i == -1:
                    break

            model.finalize()

        except Exception as e:
            print('Error: %s' % (str(e)))
            traceback.print_exc()
        break
    c2s.put ( {'op':'close'} )

_train_summary_writer = None

def cv_set_titile(oldTitle,newTitle='神农',oneRun=False):
    """
    设置窗口标题
    :param oldTitle: 旧标题
    :param newTitle: 新标题
    :param oneRun: 是否只运行一次
    :return:
    """
    if oneRun == False:
        # 根据窗口名称查找其句柄 然后使用函数修改其标题
        # 尽量选择一个不常见的英文名 防止误该已有#的窗口标题 初始化时通常取无意义的名字  比如这里取‘aa’
        handle = win32gui.FindWindow(0, oldTitle)
        win32gui.SetWindowText(handle, newTitle)
        oneRun= True
    return oneRun

def init_writer(model_name, tensorboard_dir, start_tensorboard):
    import tensorboardX
    global _train_summary_writer

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    summary_writer_folder = os.path.join(tensorboard_dir, model_name)
    _train_summary_writer = tensorboardX.SummaryWriter(summary_writer_folder)

    if start_tensorboard:
        tb_tool = TensorBoardTool(tensorboard_dir)
        tb_tool.run()

    return _train_summary_writer

def get_writer():
    global _train_summary_writer
    return _train_summary_writer

def handle_tensorboard_op(input):
    train_summary_writer = get_writer()
    action = input['action']
    if action == 'init':
        model_name = input['model_name']
        tensorboard_dir = input['tensorboard_dir']
        start_tensorboard = input['start_tensorboard']
        train_summary_writer = init_writer(model_name, tensorboard_dir, start_tensorboard)
    if train_summary_writer is not None:
        if action == 'step':
            step = input['step']
            step_time = input['step_time']
            src_loss = input['src_loss']
            dst_loss = input['dst_loss']
            # report iteration time summary
            train_summary_writer.add_scalar('iteration time', step_time, step)
            # report loss summary
            train_summary_writer.add_scalar('loss/src', src_loss, step)
            if dst_loss is not None:
                train_summary_writer.add_scalar('loss/dst', dst_loss, step)
        elif action == 'preview':
            step = input['step']
            previews = input['previews']
            static_previews = input['static_previews']
            if previews is not None:
                log_tensorboard_previews(step, previews, 'preview', train_summary_writer)
            if static_previews is not None:
                log_tensorboard_previews(step, static_previews, 'static_preview', train_summary_writer)
    

class Zoom(Enum):
    ZOOM_25 = (1 / 4, '25%')
    ZOOM_33 = (1 / 3, '33%')
    ZOOM_50 = (1 / 2, '50%')
    ZOOM_67 = (2 / 3, '67%')
    ZOOM_75 = (3 / 4, '75%')
    ZOOM_80 = (4 / 5, '80%')
    ZOOM_90 = (9 / 10, '90%')
    ZOOM_100 = (1, '100%')
    ZOOM_110 = (11 / 10, '110%')
    ZOOM_125 = (5 / 4, '125%')
    ZOOM_150 = (3 / 2, '150%')
    ZOOM_175 = (7 / 4, '175%')
    ZOOM_200 = (2, '200%')
    ZOOM_250 = (5 / 2, '250%')
    ZOOM_300 = (3, '300%')
    ZOOM_400 = (4, '400%')
    ZOOM_500 = (5, '500%')

    def __init__(self, scale, label):
        self.scale = scale
        self.label = label

    def prev(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) - 1
        if index < 0:
            return self
        return members[index]

    def next(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) + 1
        if index >= len(members):
            return self
        return members[index]


def scale_previews(previews, zoom=Zoom.ZOOM_100):
    scaled = []
    for preview in previews:
        preview_name, preview_rgb = preview
        scale_factor = zoom.scale
        if scale_factor < 1:
            scaled.append((preview_name, cv2.resize(preview_rgb, (0, 0),
                                                    fx=scale_factor,
                                                    fy=scale_factor,
                                                    interpolation=cv2.INTER_AREA)))
        elif scale_factor > 1:
            scaled.append((preview_name, cv2.resize(preview_rgb, (0, 0),
                                                    fx=scale_factor,
                                                    fy=scale_factor,
                                                    interpolation=cv2.INTER_LANCZOS4)))
        else:
            scaled.append((preview_name, preview_rgb))
    return scaled


def create_preview_pane_image(previews, selected_preview, loss_history,
                              show_last_history_iters_count, iteration, batch_size, zoom=Zoom.ZOOM_100):
    scaled_previews = scale_previews(previews, zoom)
    selected_preview_name = scaled_previews[selected_preview][0]
    selected_preview_rgb = scaled_previews[selected_preview][1]
    h, w, c = selected_preview_rgb.shape

    # HEAD
    head_lines = [
        '[s]:保存 save          [b]:备份 backup          [enter]:退出 exit',
        '[p]:刷新预览 update    [space]:切换预览模式 next preview',
        '[l]:loss range         [-/+]:缩放 zoom: %s' % zoom.label,
        '当前预览模式 Preview: "%s" [%d/%d]' % (selected_preview_name,selected_preview+1, len(previews) )
        ]

    head_line_height = int(20 * zoom.scale)
    head_height = len(head_lines) * head_line_height
    head = np.ones((head_height, w, c)) * 0.1

    for i in range(0, len(head_lines)):
        t = i * head_line_height
        b = (i + 1) * head_line_height
        head[t:b, 0:w] += imagelib.get_text_image((head_line_height, w, c), head_lines[i], color=[0.8] * c)

    final = head

    if loss_history is not None:
        if show_last_history_iters_count == 0:
            loss_history_to_show = loss_history
        else:
            loss_history_to_show = loss_history[-show_last_history_iters_count:]
        lh_height = int(100 * zoom.scale)
        lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iteration, w, c, lh_height)
        final = np.concatenate([final, lh_img], axis=0)

    final = np.concatenate([final, selected_preview_rgb], axis=0)
    final = np.clip(final, 0, 1)
    return (final * 255).astype(np.uint8)


def main(**kwargs):
    io.log_info("启动训练程序.\r\n")

    no_preview = kwargs.get('no_preview', False)
    
    global flask_preview
    flask_preview = kwargs.get('flask_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()

    previews = None
    loss_history = None
    selected_preview = 0
    update_preview = False
    is_waiting_preview = False
    show_last_history_iters_count = 0
    iteration = 0
    batch_size = 1
    zoom = Zoom.ZOOM_100

    if flask_preview:
        from flaskr.app import create_flask_app
        s2flask = queue.Queue()
        socketio, flask_app = create_flask_app(s2c, c2s, s2flask, kwargs)

        thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e, socketio), kwargs=kwargs)
        thread.start()

        e.wait()  # Wait for inital load to occur.

        flask_t = threading.Thread(target=socketio.run, args=(flask_app,),
                                   kwargs={'debug': False, 'use_reloader': False, 'host': '0.0.0.0','port':6006})
        
        flask_t.start()

        while True:
            if not c2s.empty():
                item = c2s.get()
                op = item['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = item['loss_history'] if 'loss_history' in item.keys() else None
                    previews = item['previews'] if 'previews' in item.keys() else None
                    iteration = item['iter'] if 'iter' in item.keys() else 0
                    # batch_size = input['batch_size'] if 'iter' in input.keys() else 1
                    if previews is not None:
                        update_preview = True
                elif op == 'update':
                    if not is_waiting_preview:
                        is_waiting_preview = True
                    s2c.put({'op': 'preview'})
                elif op == 'next_preview':
                    selected_preview = (selected_preview + 1) % len(previews)
                    update_preview = True
                elif op == 'change_history_range':
                    if show_last_history_iters_count == 0:
                        show_last_history_iters_count = 5000
                    elif show_last_history_iters_count == 5000:
                        show_last_history_iters_count = 10000
                    elif show_last_history_iters_count == 10000:
                        show_last_history_iters_count = 50000
                    elif show_last_history_iters_count == 50000:
                        show_last_history_iters_count = 100000
                    elif show_last_history_iters_count == 100000:
                        show_last_history_iters_count = 0
                    update_preview = True
                elif op == 'close':
                    s2c.put({'op': 'close'})
                    break
                elif op == 'zoom_prev':
                    zoom = zoom.prev()
                    update_preview = True
                elif op == 'zoom_next':
                    zoom = zoom.next()
                    update_preview = True

            if update_preview:
                update_preview = False
                selected_preview = selected_preview % len(previews)
                preview_pane_image = create_preview_pane_image(previews,
                                                               selected_preview,
                                                               loss_history,
                                                               show_last_history_iters_count,
                                                               iteration,
                                                               batch_size,
                                                               zoom)
                # io.show_image(wnd_name, preview_pane_image)
                model_path = Path(kwargs.get('saved_models_path', ''))
                filename = 'preview.png'
                preview_file = str(model_path / filename)
                cv2.imwrite(preview_file, preview_pane_image)
                s2flask.put({'op': 'show'})
                socketio.emit('preview', {'iter': iteration, 'loss': loss_history[-1]})
            try:
                io.process_messages(0.01)
            except KeyboardInterrupt:
                s2c.put({'op': 'close'})
    else:
        thread = threading.Thread(target=trainerThread, args=(s2c, c2s, e), kwargs=kwargs)
        thread.start()

        e.wait()  # Wait for inital load to occur.

    if no_preview:
        while True:
            if not c2s.empty():
                input = c2s.get()
                op = input.get('op','')
                if op == 'tb':
                    handle_tensorboard_op(input)
                elif op == 'close':
                    break
            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put({'op': 'close'})
    else:
        wnd_name = "--- ShenNong SAEHD --- Training preview"
        io.named_window(wnd_name)
        io.capture_keys(wnd_name)

        previews = None
        loss_history = None
        selected_preview = 0
        update_preview = False
        is_showing = False
        is_waiting_preview = False
        show_last_history_iters_count = 0
        iter = 0
        while True:
            if not c2s.empty():
                item = c2s.get()
                op = item['op']
                if op == 'show':
                    is_waiting_preview = False
                    loss_history = item['loss_history'] if 'loss_history' in item.keys() else None
                    previews = item['previews'] if 'previews' in item.keys() else None
                    iter = item['iter'] if 'iter' in item.keys() else 0
                    if previews is not None:
                        max_w = 0
                        max_h = 0
                        for (preview_name, preview_rgb) in previews:
                            (h, w, c) = preview_rgb.shape
                            max_h = max(max_h, h)
                            max_w = max(max_w, w)

                        max_size = 800
                        if max_h > max_size:
                            max_w = int(max_w / (max_h / max_size))
                            max_h = max_size

                        # make all previews size equal
                        for preview in previews[:]:
                            (preview_name, preview_rgb) = preview
                            (h, w, c) = preview_rgb.shape
                            if h != max_h or w != max_w:
                                previews.remove(preview)
                                previews.append((preview_name, cv2.resize(preview_rgb, (max_w, max_h))))
                        selected_preview = selected_preview % len(previews)
                        update_preview = True
                elif op == 'tb':
                    handle_tensorboard_op(item)
                elif op == 'close':
                    break

            if update_preview:
                update_preview = False
                # 获取当前选择的预览名称和对应的RGB数据
                selected_preview_name = previews[selected_preview][0]
                selected_preview_rgb = previews[selected_preview][1]
                # 获取预览图像的高度、宽度和通道数
                (h,w,c) = selected_preview_rgb.shape

                # HEAD
                head_lines = [
                    '[s]:保存 save                [b]:备份 backup                [enter]:退出 exit',
                    '[p]:刷新预览 update          [space]:切换预览模式 next preview',
                    '[l]:loss range               当前预览模式 Preview: "%s" [%d/%d]                单击图像也可刷新' % (selected_preview_name,selected_preview+1, len(previews) )
                    ]
                
                head_line_height = 15
                head_height = len(head_lines) * head_line_height
                head = np.ones ( (head_height,w,c) ) * 0.1# 创建头部区域的图像，初始为灰色背景

                for i in range(0, len(head_lines)):
                    t = i * head_line_height
                    b = (i + 1) * head_line_height
                    # 将文本图像叠加到头部区域
                    head[t:b, 0:w] += imagelib.get_text_image (  (head_line_height,w,c) , head_lines[i], color=[0.8]*c )

                final = head # 将头部区域作为最终的图像预览结果

                if loss_history is not None:
                    if show_last_history_iters_count == 0:
                        loss_history_to_show = loss_history
                    else:
                        loss_history_to_show = loss_history[-show_last_history_iters_count:]

                    lh_img = models.ModelBase.get_loss_history_preview(loss_history_to_show, iter, w, c)
                    final = np.concatenate([final, lh_img], axis=0)

                final = np.concatenate([final, selected_preview_rgb], axis=0)
                final = np.clip(final, 0, 1)

                io.show_image(wnd_name, (final * 255).astype(np.uint8))
                is_showing = True

            key_events = io.get_key_events(wnd_name)
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (
            0, 0, False, False, False)

            if key == ord('\n') or key == ord('\r'):
                s2c.put({'op': 'close'})
            elif key == ord('s'):
                s2c.put({'op': 'save'})
            elif key == ord('b'):
                s2c.put({'op': 'backup'})
            elif key == ord('p'):
                if not is_waiting_preview:
                    is_waiting_preview = True
                    s2c.put({'op': 'preview'})
            elif key == ord('l'):
                if show_last_history_iters_count == 0:
                    show_last_history_iters_count = 5000
                elif show_last_history_iters_count == 5000:
                    show_last_history_iters_count = 10000
                elif show_last_history_iters_count == 10000:
                    show_last_history_iters_count = 50000
                elif show_last_history_iters_count == 50000:
                    show_last_history_iters_count = 100000
                elif show_last_history_iters_count == 100000:
                    show_last_history_iters_count = 0
                update_preview = True
            elif key == ord(' '):
                selected_preview = (selected_preview + 1) % len(previews)
                update_preview = True

            try:
                io.process_messages(0.1)
            except KeyboardInterrupt:
                s2c.put({'op': 'close'})

        io.destroy_all_windows()
