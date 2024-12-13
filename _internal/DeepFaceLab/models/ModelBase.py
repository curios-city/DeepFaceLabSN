import colorsys
import inspect
import multiprocessing
import operator
import os
import pickle
import shutil
import time
import datetime
from pathlib import Path
import yaml
from jsonschema import validate, ValidationError
import numpy as np
from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase
from prettytable import PrettyTable


class ModelBase(object):
    # 构造函数，初始化模型的各种参数
    def __init__(
        self,
        is_training=False,
        is_exporting=False,
        saved_models_path=None,
        training_data_src_path=None,
        training_data_dst_path=None,
        pretraining_data_path=None,
        pretrained_model_path=None,
        src_pak_name=None,
        dst_pak_name=None,
        no_preview=False,
        force_model_name=None,
        force_gpu_idxs=None,
        cpu_only=False,
        debug=False,
        force_model_class_name=None,
        config_training_file=None,
        auto_gen_config=False,
        silent_start=False,
        reduce_clutter=False,
        **kwargs,
    ):
        self.is_training = is_training  # 是否处于训练模式
        self.is_exporting = is_exporting  # 是否处于导出模式
        self.saved_models_path = saved_models_path  # 保存模型的路径
        self.training_data_src_path = training_data_src_path  # 训练数据源路径
        self.training_data_dst_path = training_data_dst_path  # 训练数据目标路径
        self.pretraining_data_path = pretraining_data_path  # 预训练数据路径
        self.pretrained_model_path = pretrained_model_path  # 预训练模型路径
        self.src_pak_name = src_pak_name  # 源数据包名称
        self.dst_pak_name = dst_pak_name  # 目标数据包名称
        self.config_training_file = config_training_file  # 训练配置文件
        self.auto_gen_config = auto_gen_config  # 是否自动生成配置
        self.config_file_path = None  # 配置文件路径
        self.no_preview = no_preview  # 是否不显示预览
        self.debug = debug  # 是否处于调试模式
        self.reset_training = False  # 是否重置训练
        self.reduce_clutter = reduce_clutter  # 是否减少杂乱信息
        self.author_name = "神农汉化"

        # 初始化模型类名和模型名
        self.model_class_name = model_class_name = Path(
            inspect.getmodule(self).__file__
        ).parent.name.rsplit("_", 1)[1]
        # 根据输入参数或者模型文件自动设置模型名
        if force_model_class_name is None:
            if force_model_name is not None:
                self.model_name = force_model_name
            else:
                while True:
                    # 收集所有模型的数据文件
                    saved_models_names = []
                    for filepath in pathex.get_file_paths(saved_models_path):
                        filepath_name = filepath.name
                        if filepath_name.endswith(f"{model_class_name}_data.dat"):
                            saved_models_names += [
                                (
                                    filepath_name.split("_")[0],
                                    os.path.getmtime(filepath),
                                )
                            ]

                    # 根据修改时间对模型进行排序
                    saved_models_names = sorted(
                        saved_models_names, key=operator.itemgetter(1), reverse=True
                    )
                    saved_models_names = [x[0] for x in saved_models_names]

                    if len(saved_models_names) != 0:
                        if silent_start:
                            self.model_name = saved_models_names[0]
                            io.log_info(f'静默启动：选择的模型 "{self.model_name}"')
                        else:
                            io.log_info("选择一个已保存的模型，或输入名称创建新模型。")
                            io.log_info("[r] : 重命名")
                            io.log_info("[d] : 删除")
                            io.log_info("")
                            for i, model_name in enumerate(saved_models_names):
                                s = f"[{i}] : {model_name} "
                                if i == 0:
                                    s += "- 上次进行"
                                io.log_info(s)

                            inp = io.input_str(f"", "0", show_default_value=False)
                            model_idx = -1
                            try:
                                model_idx = np.clip(
                                    int(inp), 0, len(saved_models_names) - 1
                                )
                            except:
                                pass

                            if model_idx == -1:
                                if len(inp) == 1:
                                    is_rename = inp[0] == "r"
                                    is_delete = inp[0] == "d"

                                    if is_rename or is_delete:
                                        if len(saved_models_names) != 0:
                                            if is_rename:
                                                name = io.input_str(
                                                    f"输入你想重命名的模型名称"
                                                )
                                            elif is_delete:
                                                name = io.input_str(
                                                    f"输入你想删除的模型名称"
                                                )

                                            if name in saved_models_names:
                                                if is_rename:
                                                    new_model_name = io.input_str(
                                                        f"输入模型的新名称"
                                                    )

                                                for filepath in pathex.get_paths(
                                                    saved_models_path
                                                ):
                                                    filepath_name = filepath.name
                                                    try:
                                                        (
                                                            model_filename,
                                                            remain_filename,
                                                        ) = filepath_name.split("_", 1)
                                                    except ValueError:
                                                        # 当无法正确分割文件名时的处理逻辑
                                                        print(
                                                            "警告: model目录下有其他文件（比如压缩包）"
                                                        )
                                                        print(
                                                            "非法文件名:", filepath_name
                                                        )
                                                        continue  # 跳过当前循环，继续下一个文件

                                                    if model_filename == name:
                                                        if is_rename:
                                                            new_filepath = (
                                                                filepath.parent
                                                                / (
                                                                    new_model_name
                                                                    + "_"
                                                                    + remain_filename
                                                                )
                                                            )
                                                            filepath.rename(
                                                                new_filepath
                                                            )
                                                        elif is_delete:
                                                            filepath.unlink()
                                        continue

                                self.model_name = inp
                            else:
                                self.model_name = saved_models_names[model_idx]

                    else:
                        self.model_name = io.input_str(
                            f"未找到已保存的模型。创建一个新模型，输入名称", "new"
                        )
                        self.model_name = self.model_name.replace("_", " ")
                    break

            self.model_name = self.model_name + "_" + self.model_class_name
        else:
            self.model_name = force_model_class_name

        self.iter = 0  # 当前迭代次数
        self.options = {}  # 模型选项
        self.formatted_dictionary = {}  # 格式化的字典
        self.options_show_override = {}  # 用于覆盖显示的选项
        self.loss_history = []  # 损失历史
        self.sample_for_preview = None  # 预览样本
        self.choosed_gpu_indexes = None  # 选择的GPU索引
        model_data = {}
        # 如果yaml配置文件存在，则为True
        self.config_file_exists = False
        # 如果用户选择从外部或内部配置文件读取选项，则为True
        self.read_from_conf = False
        config_error = False

        # 检查是否启用了config_training_file模式
        if config_training_file is not None:
            if not Path(config_training_file).exists():
                # 如果配置文件不存在，记录错误信息
                io.log_err(f"{config_training_file} 不存在，不使用配置！")
            else:
                # 设置配置文件路径
                self.config_file_path = Path(self.get_strpath_def_conf_file())
        elif self.auto_gen_config:
            # 如果启用了自动生成配置，则设置配置文件路径
            self.config_file_path = Path(self.get_model_conf_path())

        # 如果配置文件路径存在
        if self.config_file_path is not None:
            # 询问用户是否从文件中读取训练选项
            self.read_from_conf = (
                io.input_bool(
                    f"您是否从文件中读取训练选项？",
                    True,
                    "从配置文件中读取选项，而不是逐个询问每个选项",
                )
                if not silent_start
                else True
            )

            # 如果用户决定从外部或内部配置文件中读取
            if self.read_from_conf:
                # 尝试从外部或内部的yaml文件中读取字典，根据auto_gen_config的值
                self.options = self.read_from_config_file(self.config_file_path)
                # 如果选项字典为空，则选项将从dat文件中加载
                if self.options is None:
                    io.log_info(f"配置文件验证错误，请检查您的配置")
                    config_error = True
                elif not self.options.keys():
                    io.log_info(f"配置文件不存在。将创建标准配置文件。")
                else:
                    io.log_info(f"使用来自 {self.config_file_path} 的配置文件")
                    self.config_file_exists = True

        # 设置模型数据文件的路径
        self.model_data_path = Path(self.get_strpath_storage_for_file("data.dat"))
        # 如果模型数据文件存在
        if self.model_data_path.exists():
            io.log_info(f"加载模型 {self.model_name}...")
            # 从文件中读取并解析模型数据
            model_data = pickle.loads(self.model_data_path.read_bytes())
            self.iter = model_data.get("iter", 0)
            try:
                self.author_name = model_data.get("author_name", "神农汉化")
            except KeyError:
                # 如果取值失败，忽略错误并将属性设置为 "神农汉化"
                print(f"读取作者名失败，模型是从旧版DFL升级")
                self.author_name = "神农汉化"

            # 如果迭代次数不为0，表示模型非首次运行
            if self.iter != 0:
                # 如果用户选择不从yaml文件读取选项，则从.dat文件中读取选项
                if not self.config_file_exists:
                    self.options = model_data["options"]
                # 从模型数据中读取损失历史、预览样本和选择的GPU索引
                self.loss_history = model_data.get("loss_history", [])
                self.sample_for_preview = model_data.get("sample_for_preview", None)
                self.choosed_gpu_indexes = model_data.get("choosed_gpu_indexes", None)

        # 如果是模型的首次运行
        if self.is_first_run():
            io.log_info("\n模型首次运行。")

        # 如果是静默启动
        if silent_start:
            # 设置设备配置
            if force_gpu_idxs is not None:
                self.device_config = (
                    nn.DeviceConfig.GPUIndexes(force_gpu_idxs)
                    if not cpu_only
                    else nn.DeviceConfig.CPU()
                )
                io.log_info(
                    f"静默启动：选择的设备{'s' if len(force_gpu_idxs) > 0 else ''} {'CPU' if self.device_config.cpu_only else [device.name for device in self.device_config.devices]}"
                )
            else:
                self.device_config = nn.DeviceConfig.BestGPU()
                io.log_info(
                    f"静默启动：选择的设备{'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}"
                )
        else:
            # 在非静默启动下设置设备配置
            self.device_config = (
                nn.DeviceConfig.GPUIndexes(
                    force_gpu_idxs
                    or nn.ask_choose_device_idxs(suggest_best_multi_gpu=True)
                )
                if not cpu_only
                else nn.DeviceConfig.CPU()
            )

        # 初始化神经网络
        nn.initialize(self.device_config)

        ####
        # 设置默认选项文件的路径
        self.default_options_path = (
            saved_models_path / f"{self.model_class_name}_default_options.dat"
        )
        self.default_options = {}
        # 如果默认选项文件存在
        if self.default_options_path.exists():
            try:
                # 从文件中读取并解析默认选项
                self.default_options = pickle.loads(
                    self.default_options_path.read_bytes()
                )
            except:
                pass

        # 初始化预览历史选择和批处理大小
        self.choose_preview_history = False
        self.batch_size = self.load_or_def_option("batch_size", 1)
        #####

        # 跳过所有挂起的输入
        io.input_skip_pending()
        # 初始化选项
        self.on_initialize_options()

        # 如果是模型的首次运行
        if self.is_first_run():
            # 只在首次运行模型时将当前选项保存为默认选项
            self.default_options_path.write_bytes(pickle.dumps(self.options))

        # 保存配置文件
        if (
            self.read_from_conf
            and not self.config_file_exists
            and not config_error
            and not self.config_file_path is None
        ):
            self.save_config_file(self.config_file_path)

        # 从选项中获取各种配置参数
        # self.author_name = self.options.get("author_name", "")
        self.autobackup_hour = self.options.get("autobackup_hour", 0)
        self.maximum_n_backups = self.options.get("maximum_n_backups", 24)
        self.write_preview_history = self.options.get("write_preview_history", False)
        self.target_iter = self.options.get("target_iter", 0)
        self.random_flip = self.options.get("random_flip", True)
        self.random_src_flip = self.options.get("random_src_flip", False)
        self.random_dst_flip = self.options.get("random_dst_flip", True)
        self.retraining_samples = self.options.get("retraining_samples", False)
        
        if self.model_class_name =="ME":
            self.super_warp = self.options.get("super_warp", False)
            if self.options["super_warp"] == True:
                self.rotation_range=[-12,12]
                self.scale_range=[-0.2, 0.2]
        # 完成初始化
        self.on_initialize()
        # 更新选项中的批处理大小
        self.options["batch_size"] = self.batch_size

        self.preview_history_writer = None
        # 如果处于训练状态
        if self.is_training:
            # 设置预览历史和自动备份的路径
            self.preview_history_path = self.saved_models_path / (
                f"{self.get_model_name()}_history"
            )
            self.autobackups_path = self.saved_models_path / (
                f"{self.get_model_name()}_autobackups"
            )

            # 如果启用了写入预览历史或在Colab环境中
            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    # 如果迭代次数为0，清理预览历史文件夹
                    if self.iter == 0:
                        for filename in pathex.get_image_paths(
                            self.preview_history_path
                        ):
                            Path(filename).unlink()

            # 检查是否设置了训练数据生成器
            if self.generator_list is None:
                raise ValueError("You didnt set_training_data_generators()")
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError(
                            "training data generator is not subclass of SampleGeneratorBase"
                        )

            # 更新预览样本
            self.update_sample_for_preview(
                choose_preview_history=self.choose_preview_history
            )

            # 如果设置了自动备份时间
            if self.autobackup_hour != 0:
                self.autobackup_start_time = time.time()

                if not self.autobackups_path.exists():
                    self.autobackups_path.mkdir(exist_ok=True)

        # 打印训练摘要，加了if是因为前面已经打印过一次了，这是修改后才需再现
        if self.ask_override:
            try:
                io.log_info( self.get_summary_text() )
            except KeyError:
                print(f"参数尚未填写完整！将无法保存！")
                
    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        # 更新预览样本
        if self.sample_for_preview is None or choose_preview_history or force_new:
            # 如果选择了预览历史并且在Windows环境中
            if choose_preview_history and io.is_support_windows():
                wnd_name = (
                    "[p] - next. [space] - switch preview type. [enter] - confirm."
                )
                io.log_info(f"选择预览图演变的图片. {wnd_name}")
                io.named_window(wnd_name)
                io.capture_keys(wnd_name)
                choosed = False
                preview_id_counter = 0
                mask_changed = False
                while not choosed:
                    if not mask_changed:
                        self.sample_for_preview = self.generate_next_samples()
                        previews = self.get_history_previews()

                    io.show_image(
                        wnd_name,
                        (previews[preview_id_counter % len(previews)][1] * 255).astype(
                            np.uint8
                        ),
                    )

                    while True:
                        key_events = io.get_key_events(wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = (
                            key_events[-1]
                            if len(key_events) > 0
                            else (0, 0, False, False, False)
                        )
                        if key == ord("\n") or key == ord("\r"):
                            choosed = True
                            break
                        elif key == ord(" "):
                            preview_id_counter += 1
                            mask_changed = True
                            break
                        elif key == ord("p"):
                            if mask_changed:
                                mask_changed = False
                            break

                        try:
                            io.process_messages(0.1)
                        except KeyboardInterrupt:
                            choosed = True

                io.destroy_window(wnd_name)
            else:
                # 生成下一个样本作为预览
                self.sample_for_preview = self.generate_next_samples()

        try:
            self.get_history_previews()
        except:
            self.sample_for_preview = self.generate_next_samples()

        self.last_sample = self.sample_for_preview

    def load_or_def_option(self, name, def_value):
        # 从options中加载指定名称的值，如果不存在则尝试从default_options中加载，最后使用默认值
        options_val = self.options.get(name, None)
        if options_val is not None:
            return options_val

        def_opt_val = self.default_options.get(name, None)
        if def_opt_val is not None:
            return def_opt_val

        return def_value

    def load_inter_dims(self):
        # 尝试从options中加载inter_dims值，如果不存在则返回False
        try:
            v = self.options["inter_dims"]
        except KeyError:
            return False
        return v

    def ask_override(self):
        # 打印模型摘要
        try:
            io.log_info( self.get_summary_text() )
        except KeyError:
            print(f"模型首次在神农运行,强制询问参数")
            return True

        # 设置延迟时间，如果在Colab环境中为5秒，否则为10秒
        time_delay = 5 if io.is_colab() else 10
        # 如果处于训练状态且不是首次运行，询问用户是否覆盖模型设置
        return (
            self.is_training
            and not self.is_first_run()
            and io.input_in_time(
                f"在{time_delay}秒内按Enter键修改模型设置。", time_delay
            )
        )

    def ask_reset_training(self):
        # 询问用户是否要重置迭代计数器和损失图表
        self.reset_training = io.input_bool(
            "您是否要重置迭代计数器和损失图表？",
            False,
            "重置模型的迭代计数器和损失图表，但您的模型不会失去训练进度。"
            "如果您总是使用同一个模型进行多次训练，这会很有用。",
        )

        if self.reset_training:
            self.set_iter(0)
                
    def ask_author_name(self, default_value="神农汉化"):

        # 询问用户输入名
        self.author_name = io.input_str(
            "模型作者 Author name",
            "神农汉化",
            help_message="显示的作者署名",
        )

    def ask_autobackup_hour(self, default_value=0):
        # 加载autobackup_hour选项，如果不存在则使用默认值
        default_autobackup_hour = self.options["autobackup_hour"] = (
            self.load_or_def_option("autobackup hour", default_value)
        )
        # 询问用户输入自动备份的时间间隔
        self.options["autobackup_hour"] = io.input_int(
            f"每N小时自动备份 Autobackup hour",
            default_autobackup_hour,
            add_info="0..24",
            help_message="每N小时自动备份模型文件和预览图。最新的备份是按名称升序排列时位于model/<>_autobackups的最后一个文件夹",
        )

    def ask_maximum_n_backups(self, default_value=24):
        # 加载maximum_n_backups选项，如果不存在则使用默认值
        default_maximum_n_backups = self.options["maximum_n_backups"] = (
            self.load_or_def_option("maximum_n_backups", default_value)
        )
        # 询问用户输入最大备份数量
        self.options["maximum_n_backups"] = io.input_int(
            f"最大备份数量 Maximum n backups",
            default_maximum_n_backups,
            help_message="位于model/<>_autobackups中的最大备份数量。输入0将允许它根据发生次数进行任意次数的自动备份。",
        )

    def ask_write_preview_history(self, default_value=False):
        # 加载write_preview_history选项，如果不存在则使用默认值
        default_write_preview_history = self.load_or_def_option(
            "write_preview_history", default_value
        )
        # 询问用户是否写入预览历史
        self.options["write_preview_history"] = io.input_bool(
            f"记录预览图演变史 Write preview history",
            default_write_preview_history,
            help_message="预览图演变史将被写入<ModelName>_history文件夹。",
        )

        if self.options["write_preview_history"]:
            if io.is_support_windows():
                self.choose_preview_history = io.input_bool(
                    "选择要记录预览的图片序号", False
                )
            elif io.is_colab():
                self.choose_preview_history = io.input_bool(
                    "随机选择要记录预览的图片序号",
                    False,
                    help_message="如果你在不同的人物上重用同一个模型，预览图演变史将记录旧脸。除非您要将源src/目标dst更改为新的人物，否则请选择否",
                )

    def ask_target_iter(self, default_value=0):
        # 加载target_iter选项，如果不存在则使用默认值
        default_target_iter = self.load_or_def_option("target_iter", default_value)
        # 询问用户输入目标迭代次数
        self.options["target_iter"] = max(
            0, io.input_int("目标迭代次数 Target iter", default_target_iter)
        )

    def ask_random_flip(self):
        # 加载random_flip选项，如果不存在则使用默认值
        default_random_flip = self.load_or_def_option("random_flip", True)
        # 询问用户是否随机翻转脸部
        self.options["random_flip"] = io.input_bool(
            "随机翻转脸部 Random flip",
            default_random_flip,
            help_message="预测的脸部看起来更自然，但src脸部集应覆盖所有与dst脸部集相同的方向。",
        )

    def ask_random_src_flip(self):
        # 加载random_src_flip选项，如果不存在则使用默认值
        default_random_src_flip = self.load_or_def_option("random_src_flip", False)
        # 询问用户是否随机翻转SRC脸部
        self.options["random_src_flip"] = io.input_bool(
            "随机翻转SRC脸部 Random src flip",
            default_random_src_flip,
            help_message="随机水平翻转SRC脸部集。覆盖更多角度，但脸部可能看起来不太自然。",
        )

    def ask_random_dst_flip(self):
        # 加载random_dst_flip选项，如果不存在则使用默认值
        default_random_dst_flip = self.load_or_def_option("random_dst_flip", True)
        # 询问用户是否随机翻转DST脸部
        self.options["random_dst_flip"] = io.input_bool(
            "随机翻转DST脸部 Random dst flip",
            default_random_dst_flip,
            help_message="随机水平翻转DST脸部集。如果src随机翻转未启用，则使src->dst的泛化更好。",
        )

    def ask_batch_size(self, suggest_batch_size=None, range=None):
        # 加载batch_size选项，如果不存在则使用建议值或当前批处理大小
        default_batch_size = self.load_or_def_option(
            "batch_size", suggest_batch_size or self.batch_size
        )
        # 询问用户输入批处理大小
        batch_size = max(
            0,
            io.input_int(
                "批处理大小 Batch size",
                default_batch_size,
                valid_range=range,
                help_message="更大的批处理大小有助于神经网络的泛化，但可能导致内存溢出错误。请手动调整以适应您的显卡。",
            ),
        )

        if range is not None:
            batch_size = np.clip(batch_size, range[0], range[1])

        self.options["batch_size"] = self.batch_size = batch_size

    def ask_retraining_samples(self, default_value=False):
        # 加载retraining_samples选项，如果不存在则使用默认值
        default_retraining_samples = self.load_or_def_option(
            "retraining_samples", default_value
        )
        # 询问用户是否重新训练高损失样本
        self.options["retraining_samples"] = io.input_bool(
            "重新训练高损失样本 Retraining samples",
            default_retraining_samples,
            help_message="定期重新训练最后16个高损失样本",
        )
    
    def ask_quick_opt(self):
        # 加载random_src_flip选项，如果不存在则使用默认值
        default_quick_opt = self.load_or_def_option("quick_opt", False)
        # 询问用户是否随机翻转SRC脸部
        self.options["quick_opt"] = io.input_bool(
            "训练眼嘴 train eye_mouth (y)  训练皮肤 train skin (n)",
            default_quick_opt,
            help_message="先训练眼嘴，等每次保存loss的变化小于0.01的时候训练皮肤（GAN）",
        )
        
    # 可被重写的方法
    def on_initialize_options(self):
        pass

    # 可被重写的方法
    def on_initialize(self):
        """
        初始化你的模型

        在self.options['']中存储和检索你的模型选项

        参见示例
        """
        pass

    # 可被重写的方法
    def onSave(self):
        # 在这里保存你的模型
        pass

    # 可被重写的方法
    def onTrainOneIter(self, sample, generator_list):
        # 在这里训练你的模型

        # 返回损失数组
        return (("loss_src", 0), ("loss_dst", 0))

    # 可被重写的方法
    def onGetPreview(self, sample, for_history=False, filenames=None):
        # 你可以返回多个预览
        # 返回 [('preview_name', preview_rgb), ...]
        return []

    # 可被重写的方法，如果你希望模型名称与文件夹名称不同
    def get_model_name(self):
        return self.model_name

    # 可被重写的方法，返回 [[model, filename], ...] 列表
    def get_model_filename_list(self):
        return []

    # 可被重写的方法
    def get_MergerConfig(self):
        # 返回模型的predictor_func、predictor_input_shape和MergerConfig()
        raise NotImplementedError

    # 可被重写的方法
    def get_config_schema_path(self):
        raise NotImplementedError

    # 可被重写的方法
    def get_formatted_configuration_path(self):
        return "None"
        # raise NotImplementedError

    def get_pretraining_data_path(self):
        # 返回预训练数据的路径
        return self.pretraining_data_path

    def get_target_iter(self):
        # 返回目标迭代次数
        return self.target_iter

    def is_reached_iter_goal(self):
        # 检查是否达到了目标迭代次数
        return self.target_iter != 0 and self.iter >= self.target_iter

    def get_previews(self):
        # 获取预览图像
        return self.onGetPreview(self.last_sample, filenames=self.last_sample_filenames)

    def get_static_previews(self):
        # 获取静态预览图像
        return self.onGetPreview(self.sample_for_preview)

    def get_history_previews(self):
        # 获取历史预览图像
        return self.onGetPreview(self.sample_for_preview, for_history=True)

    def get_preview_history_writer(self):
        # 获取或创建预览历史写入器
        if self.preview_history_writer is None:
            self.preview_history_writer = PreviewHistoryWriter()
        return self.preview_history_writer

    def save(self):
        # 保存模型摘要
        Path(self.get_summary_path()).write_text(
            self.get_summary_text(), encoding="utf-8"
        )

        # 调用保存模型的函数
        self.onSave()

        # 如果启用了自动生成配置
        if self.auto_gen_config:
            path = Path(self.get_model_conf_path())
            self.save_config_file(path)

        # 准备保存的模型数据
        model_data = {
            "iter": self.iter,
            "author_name": self.author_name,
            "options": self.options,
            "loss_history": self.loss_history,
            "sample_for_preview": self.sample_for_preview,
            "choosed_gpu_indexes": self.choosed_gpu_indexes,
        }

        # 创建临时文件路径
        temp_model_data_path = Path(self.model_data_path).with_suffix(".tmp")

        # 将序列化的模型数据写入临时文件
        with open(temp_model_data_path, "wb") as f:
            pickle.dump(model_data, f)

        # 使用write_bytes_safe将临时文件移动到最终目的地
        pathex.write_bytes_safe(Path(self.model_data_path), temp_model_data_path)

        # 如果设置了自动备份时间
        if self.autobackup_hour != 0:
            diff_hour = int((time.time() - self.autobackup_start_time) // 3600)

            if diff_hour > 0 and diff_hour % self.autobackup_hour == 0:
                self.autobackup_start_time += self.autobackup_hour * 3600
                self.create_backup()

    def __convert_type_write(self, value):
        # 转换数据类型以便写入
        if isinstance(value, (np.int32, np.float64, np.int64)):
            return value.item()
        else:
            return value

    def __update_nested_dict(self, nested_dict, key, val):
        # 更新嵌套字典中的键值
        if key in nested_dict:
            nested_dict[key] = self.__convert_type_write(val)
            return True
        for v in nested_dict.values():
            if isinstance(v, dict):
                if self.__update_nested_dict(v, key, val):
                    return True
        return False

    def __iterate_read_dict(self, nested_dict, new_dict=None):
        # 迭代读取嵌套字典
        if new_dict is None:
            new_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                new_dict.update(self.__iterate_read_dict(v, new_dict))
            else:
                new_dict[k] = v
        return new_dict

    def read_from_config_file(self, filepath, keep_nested=False, validation=True):
        """
        从配置yaml文件中读取选项。

        参数:
            filepath (str|Path): 读取配置文件的路径。
            keep_nested (bool, optional): 如果为false，则保持字典嵌套，否则不保持。默认为False。
            validation (bool, optional): 如果为true，则验证字典。默认为True。

        返回:
            [dict]: optional 字典。
        """
        data = {}
        try:
            # 打开文件并读取数据
            with open(filepath, "r", encoding="utf-8") as file, open(
                self.get_config_schema_path(), "r"
            ) as schema:
                data = yaml.safe_load(file)
                if not keep_nested:
                    data = self.__iterate_read_dict(data)
                if validation:
                    validate(data, yaml.safe_load(schema))
        except FileNotFoundError:
            return {}
        except ValidationError as ve:
            io.log_err(f"{ve}")
            return None

        return data

    def save_config_file(self, filepath):
        """
        将选项保存到配置yaml文件

        参数:
            filepath (str|Path): 保存配置文件的路径。
        """
        formatted_dict = self.read_from_config_file(
            self.get_formatted_configuration_path(), keep_nested=True, validation=False
        )

        # 更新字典并保存
        for key, value in self.options.items():
            if not self.__update_nested_dict(formatted_dict, key, value):
                pass
                # print(f"'{key}' 未在配置文件中保存")

        try:
            with open(filepath, "w", encoding="utf-8") as file:
                yaml.dump(
                    formatted_dict,
                    file,
                    Dumper=yaml.SafeDumper,
                    allow_unicode=True,
                    default_flow_style=False,
                    encoding="utf-8",
                    sort_keys=False,
                )
        except OSError as exception:
            io.log_info("无法写入YAML配置文件 -> ", exception)

    def create_backup(self):
        io.log_info("正在创建备份...", end="\r")

        # 确保备份路径存在
        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        # 准备备份文件列表
        bckp_filename_list = [
            self.get_strpath_storage_for_file(filename)
            for _, filename in self.get_model_filename_list()
        ]
        bckp_filename_list += [str(self.get_summary_path()), str(self.model_data_path)]

        # 创建新备份
        idx_str = (
            datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S_")
            + str(self.iter // 1000)
            + "k"
        )
        idx_backup_path = self.autobackups_path / idx_str
        idx_backup_path.mkdir()
        for filename in bckp_filename_list:
            shutil.copy(str(filename), str(idx_backup_path / Path(filename).name))
        # 生成预览图并保存在新备份中
        previews = self.get_previews()
        plist = []
        for i in range(len(previews)):
            name, bgr = previews[i]
            plist += [(bgr, idx_backup_path / (("preview_%s.jpg") % (name)))]

        if len(plist) != 0:
            self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        # 检查是否超出了最大备份数量
        if self.maximum_n_backups != 0:
            all_backups = sorted(
                [x for x in self.autobackups_path.iterdir() if x.is_dir()]
            )
            while len(all_backups) > self.maximum_n_backups:
                oldest_backup = all_backups.pop(0)
                pathex.delete_all_files(oldest_backup)
                oldest_backup.rmdir()

    def debug_one_iter(self):
        # 调试一次迭代，将生成的图像堆叠成一个方形图像
        images = []
        for generator in self.generator_list:
            for i, batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append(batch[0])

        return imagelib.equalize_and_stack_square(images)

    def generate_next_samples(self):
        # 生成下一批样本

        sample = []  # 用于存储样本数据
        sample_filenames = []  # 用于存储样本文件名

        # 遍历生成器列表
        for generator in self.generator_list:
            # 检查生成器是否已初始化
            if generator.is_initialized():
                # 生成下一批样本
                batch = generator.generate_next()
                # 如果生成的是元组形式的样本（包含数据和文件名）
                if type(batch) is tuple:
                    sample.append(batch[0])  # 将样本数据添加到样本列表中
                    sample_filenames.append(batch[1])  # 将样本文件名添加到文件名列表中
                else:
                    sample.append(batch)  # 将样本添加到样本列表中
            else:
                sample.append([])  # 若生成器未初始化，则将空列表添加到样本列表中

        # 更新上一批样本和文件名
        self.last_sample = sample  # 更新上一批样本数据
        self.last_sample_filenames = sample_filenames  # 更新上一批样本文件名

        # 返回生成的样本
        return sample

    # 可被重写的方法
    def should_save_preview_history(self):
        # 判断是否应该保存预览历史
        return (not io.is_colab() and self.iter % 10 == 0) or (
            io.is_colab() and self.iter % 100 == 0
        )

    def train_one_iter(self):
        # 训练一次迭代
        iter_time = time.time()
        losses = self.onTrainOneIter()
        iter_time = time.time() - iter_time

        self.loss_history.append([float(loss[1]) for loss in losses])

        # 如果需要保存预览历史
        if self.should_save_preview_history():
            plist = []

            # 在Colab环境下处理
            if io.is_colab():
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [
                        (
                            bgr,
                            self.get_strpath_storage_for_file(
                                "preview_%s.jpg" % (name)
                            ),
                        )
                    ]

            if self.write_preview_history:
                previews = self.get_history_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    path = self.preview_history_path / name
                    plist += [(bgr, str(path / (f"{self.iter:07d}.jpg")))]
                    if not io.is_colab():
                        plist += [(bgr, str(path / ("_last.jpg")))]

            if len(plist) != 0:
                self.get_preview_history_writer().post(
                    plist, self.loss_history, self.iter
                )

        self.iter += 1

        return self.iter, iter_time

    def pass_one_iter(self):
        # 执行一次迭代
        self.generate_next_samples()

    def finalize(self):
        # 结束训练会话
        nn.close_session()

    def is_first_run(self):
        # 判断是否是首次运行
        return self.iter == 0 and not self.reset_training

    def is_debug(self):
        # 判断是否处于调试模式
        return self.debug

    def set_batch_size(self, batch_size):
        # 设置批处理大小
        self.batch_size = batch_size

    def get_batch_size(self):
        # 获取批处理大小
        return self.batch_size

    def get_iter(self):
        # 获取当前迭代次数
        return self.iter

    def get_author_name(self):
        # 获取当前迭代次数
        return self.author_name

    def set_iter(self, iter):
        # 设置迭代次数并更新损失历史
        self.iter = iter
        self.loss_history = self.loss_history[:iter]

    def get_loss_history(self):
        # 获取损失历史
        return self.loss_history

    def set_training_data_generators(self, generator_list):
        # 设置训练数据生成器
        self.generator_list = generator_list

    def get_training_data_generators(self):
        # 获取训练数据生成器
        return self.generator_list

    def get_model_root_path(self):
        # 获取模型根路径
        return self.saved_models_path

    def get_strpath_storage_for_file(self, filename):
        # 获取存储文件的字符串路径
        return str(self.saved_models_path / (self.get_model_name() + "_" + filename))

    def get_strpath_configuration_path(self):
        # 获取配置文件的字符串路径
        return str(self.config_file_path)

    def get_strpath_def_conf_file(self):
        # 获取默认配置文件的路径
        if Path(self.config_training_file).is_file():
            return str(Path(self.config_training_file))
        elif Path(
            self.config_training_file
        ).is_dir():  # 为了向后兼容，如果是目录则返回目录下的def_conf_file.yaml
            return str(Path(self.config_training_file) / "def_conf_file.yaml")
        else:
            return None

    def get_summary_path(self):
        # 获取摘要文件的路径
        return self.get_strpath_storage_for_file("summary.txt")

    def get_model_conf_path(self):
        # 获取模型配置文件的路径
        return self.get_strpath_storage_for_file("configuration_file.yaml")

    def get_summary_text(self, reduce_clutter=False):
        # 生成模型超参数的文本摘要
        # visible_options 是显示出来的，不一定是全部，也不影响本次训练
        visible_options = self.options.copy()
        # 参数覆写，用新的局部字典（不需要完整）刷新原字典的对应键值，可增可改
        visible_options.update(self.options_show_override)

        #把参数值汉化
        def str2cn(option):
            if str(option) == "False" or str(option) == "n":
                return "关"
            elif str(option) == "True" or str(option) == "y":
                return "开"
            else:
                return str(option)

        if self.model_class_name == "XSeg":
            xs_res = self.options.get("resolution", 256)
            table = PrettyTable(
                ["分辨率", "模型作者", "批处理大小", "预训练模式"]
            )
            table.add_row(
                [
                    str(xs_res),
                    str2cn(self.author_name),
                    str2cn(self.batch_size),
                    str2cn(self.options["pretrain"]),
                ]
            )    
            # 打印表格
            summary_text = table.get_string()
            return summary_text
            
        elif self.model_class_name =="ME":
            # 创建一个空表格对象，并指定列名
            table = PrettyTable(
                ["模型摘要", "增强选项", "开关", "参数设置", "数值", "本机配置"]
            )

            # 添加数据行
            table.add_row(
                [
                    "",
                    "重新训练高损失样本",
                    str2cn(self.options["retraining_samples"]),
                    "批处理大小",
                    str2cn(self.batch_size),
                    "AdaBelief优化器: "+str2cn(self.options["adabelief"]),
                ]
            )
            table.add_row(
                [
                    "模型名称: " + self.get_model_name(),
                    "随机翻转SRC",
                    str2cn(self.options["random_src_flip"]),
                    "",
                    "",
                    "优化器放到GPU上: "+str2cn(self.options["models_opt_on_gpu"]),
                ]
            )
            table.add_row(
                [
                    "模型作者: " + self.get_author_name(),
                    "随机翻转DST",
                    str2cn(self.options["random_dst_flip"]),
                    "学习率",
                    str2cn(self.options["lr"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "",
                    "遮罩训练",
                    str2cn(self.options["masked_training"]),
                    "真脸(src)强度",
                    str2cn(self.options["true_face_power"]),
                    "",
                ]
            )       
            table.add_row(
                [
                    "迭代数: " + str2cn(self.get_iter()),
                    "眼睛优先",
                    str2cn(self.options["eyes_prio"]),
                    "背景(src)强度",
                    str2cn(self.options["background_power"]),
                    "目标迭代次数: " + str2cn(self.options["target_iter"]),
                ]
            )
            table.add_row(
                [
                    "模型架构: " + str2cn(self.options["archi"]),
                    "嘴巴优先",
                    str2cn(self.options["mouth_prio"]),
                    "人脸(dst)强度",
                    str2cn(self.options["face_style_power"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "",
                    "侧脸优化",
                    str2cn(self.options["uniform_yaw"]),
                    "背景(dst)强度",
                    str2cn(self.options["bg_style_power"]),
                    "",
                ]
            )    
            table.add_row(
                [
                    "分辨率:" + str2cn(self.options["resolution"]),
                    "遮罩边缘模糊",
                    str2cn(self.options["blur_out_mask"]),
                    "色彩转换模式",
                    str2cn(self.options["ct_mode"]),
                    "启用梯度裁剪: "+str2cn(self.options["clipgrad"]),
                ]
            )
            table.add_row(
                [
                    "自动编码器(ae_dims): " + str2cn(self.options["ae_dims"]),
                    "使用学习率下降",
                    str2cn(self.options["lr_dropout"]),
                    "",
                    "",
                    "",
                ]
            )
            table.add_row(
                [
                    "编码器(e_dims): " + str2cn(self.options["e_dims"]),
                    "随机扭曲",
                    str2cn(self.options["random_warp"]),
                    "Loss function",
                    str2cn(self.options["loss_function"]),
                    "",
                ]
            )
            table.add_row(
                [
                    "解码器(d_dims): " +  str2cn(self.options["d_dims"]),
                    "随机颜色(hsv微变)",
                    str2cn(self.options["random_hsv_power"]),
                    "",
                    "",
                    "记录预览图演变史: " + str2cn(self.options["write_preview_history"]),
                ]
            )      
            table.add_row(
                [
                    "解码器遮罩(d_mask): " +  str2cn(self.options["d_mask_dims"]),
                    "随机颜色(亮度不变)",
                    str2cn(self.options["random_color"]),
                    "gan_power",
                    str2cn(self.options["gan_power"]),
                    "",
                ]
            )   
            table.add_row(
                [
                    "",
                    "随机降低采样",
                    str2cn(self.options["random_downsample"]),
                    "gan_patch_size",
                    str2cn(self.options["gan_patch_size"]),
                    "",
                ]
            )   
            table.add_row(
                [
                    "使用fp16:"+str2cn(self.options["use_fp16"]),
                    "随机添加噪音",
                    str2cn(self.options["random_noise"]),
                    "gan_dims",
                    str2cn(self.options["gan_dims"]),
                    "自动备份间隔: " + str2cn(self.options["autobackup_hour"]) + " 小时",
                ]
            )      
            table.add_row(
                [
                    "",
                    "随机产生模糊",
                    str2cn(self.options["random_blur"]),
                    "gan_smoothing",
                    str2cn(self.options["gan_smoothing"]),
                    "",
                ]
            )           
            table.add_row(
                [
                    "预训练模式:" +  str2cn(self.options["pretrain"]),
                    "随机压缩jpeg",
                    str2cn(self.options["random_jpeg"]),
                    "gan_noise",
                    str2cn(self.options["gan_noise"]),
                    "最大备份数量: " + str2cn(self.options["maximum_n_backups"]),
                ]
            )    
            table.add_row(
                [
                    "",
                    "超级扭曲",
                    str2cn(self.options["super_warp"]),
                    "",
                    "",
                    "",
                ]
            )    
            # 设置对齐方式（可选）["模型摘要", "增强选项", "开关", "参数设置", "数值", "本机配置"]
            table.align["模型摘要"] = "l"  # 左对齐
            table.align["增强选项"] = "r"  # 居中对齐
            table.align["开关"] = "l"  # 居中对齐
            table.align["参数设置"] = "r"  # 居中对齐
            table.align["数值"] = "l"  # 居中对齐
            table.align["本机配置"] = "r"  # 居中对齐
            # 打印表格
            summary_text = table.get_string()
        
            return summary_text
        else:
            summary_text = "未定义表格"
            return summary_text

    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c, lh_height=100):
        # 将损失历史转换为NumPy数组
        loss_history = np.array(loss_history.copy())

        # 创建损失历史图像
        lh_img = np.ones((lh_height, w, c)) * 0.1

        if len(loss_history) != 0:
            loss_count = len(loss_history[0])
            lh_len = len(loss_history)

            # 计算每一列的损失
            l_per_col = lh_len / w
            plist_max = [
                [
                    max(
                        0.0,
                        loss_history[int(col * l_per_col)][p],
                        *[
                            loss_history[i_ab][p]
                            for i_ab in range(
                                int(col * l_per_col), int((col + 1) * l_per_col)
                            )
                        ],
                    )
                    for p in range(loss_count)
                ]
                for col in range(w)
            ]

            plist_min = [
                [
                    min(
                        plist_max[col][p],
                        loss_history[int(col * l_per_col)][p],
                        *[
                            loss_history[i_ab][p]
                            for i_ab in range(
                                int(col * l_per_col), int((col + 1) * l_per_col)
                            )
                        ],
                    )
                    for p in range(loss_count)
                ]
                for col in range(w)
            ]

            # 计算最大的损失值，用于归一化
            plist_abs_max = np.mean(loss_history[len(loss_history) // 5 :]) * 2

            # 遍历每一列（w表示列数）
            for col in range(0, w):
                # 遍历每一个损失函数 (loss_count为损失函数的数量)
                for p in range(0, loss_count):
                    # 设置数据点的颜色，根据HSV颜色空间生成
                    point_color = [1.0] * 3
                    # point_color_src = (0.0, 0.8, 0.9)
                    point_color_dst = (0.8, 0.3, 0.0)
                    point_color_mix = (0.1, 0.8, 0.0)
                    # 根据实验，应该是BGR的顺序
                    if p == 0:
                        point_color = point_color_dst
                    if p == 1:
                        point_color = point_color_src

                    # 检查plist_max[col][p] 和 plist_abs_max 是否为 NaN 或零
                    if np.isnan(plist_max[col][p]) or np.isnan(plist_abs_max) or plist_abs_max == 0:
                        # 如果是 NaN 或零，将它们设为1
                        plist_max[col][p] = 1.0
                        plist_abs_max = 1.0  # 避免除零错误

                    # 计算数据点在图像中的位置（最大值和最小值）
                    ph_max = int((plist_max[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_max = np.clip(ph_max, 0, lh_height - 1)

                    ph_min = int((plist_min[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_min = np.clip(ph_min, 0, lh_height - 1)  # 将最小值限制在图像高度范围内

                    # 遍历从最小值到最大值的范围
                    for ph in range(ph_min, ph_max + 1):
                        # 在图像数组中根据计算得到的位置和颜色添加标记点
                        # 注意：由于数组的原点通常位于左上角，所以需要使用(lh_height-ph-1)来将y坐标转换为数组索引
                        if p == 0:
                            lh_img[(lh_height - ph - 1), col] = point_color
                        if p == 1:
                            current_point_color = lh_img[(lh_height - ph - 1), col]
                            # 叠加新的颜色到当前颜色
                            if (current_point_color == point_color_dst).all():
                                lh_img[(lh_height - ph - 1), col] = point_color_mix
                            else:
                                lh_img[(lh_height - ph - 1), col] = point_color_src

                                
        lh_lines = 8
        # 计算每行的高度
        lh_line_height = (lh_height-1)/lh_lines
        
        # 设置线条颜色和透明度
        line_color = (0.2, 0.2, 0.2)  # 灰色
        
        for i in range(0,lh_lines+1):
            # 获取当前分割线所在行的索引
            line_index = int(i * lh_line_height)
            # 将当前行的像素值设置为线条颜色和透明度
            lh_img[line_index, :] = line_color
            # 原灰色 lh_img[ int(i*lh_line_height), : ] = (0.8,)*c

        # 计算最后一行文字的高度位置
        last_line_t = int((lh_lines-1)*lh_line_height)
        last_line_b = int(lh_lines*lh_line_height)

        lh_text = '迭代数: %d  iter' % (iter) if iter != 0 else ''
        
        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image(
            (last_line_b - last_line_t, w, c), lh_text, color=[0.8] * c
        )
        return lh_img


class PreviewHistoryWriter:
    def __init__(self):
        # 初始化时创建一个多进程队列和一个处理进程
        self.sq = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.process, args=(self.sq,))
        self.p.daemon = True  # 设置进程为守护进程
        self.p.start()

    def process(self, sq):
        # 处理函数，用于处理队列中的项目
        while True:
            while not sq.empty():
                # 从队列中获取项目
                plist, loss_history, iter = sq.get()

                # 缓存预览损失历史图像
                preview_lh_cache = {}
                for preview, filepath in plist:
                    filepath = Path(filepath)
                    i = (preview.shape[1], preview.shape[2])

                    # 获取或创建损失历史预览图像
                    preview_lh = preview_lh_cache.get(i, None)
                    if preview_lh is None:
                        preview_lh = ModelBase.get_loss_history_preview(
                            loss_history, iter, preview.shape[1], preview.shape[2]
                        )
                        preview_lh_cache[i] = preview_lh

                    # 合并并保存图像
                    img = (np.concatenate([preview_lh, preview], axis=0) * 255).astype(
                        np.uint8
                    )
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2_imwrite(filepath, img)

            time.sleep(0.01)

    def post(self, plist, loss_history, iter):
        # 向队列发送项目
        self.sq.put((plist, loss_history, iter))

    # 禁用序列化
    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        self.__dict__.update(d)
