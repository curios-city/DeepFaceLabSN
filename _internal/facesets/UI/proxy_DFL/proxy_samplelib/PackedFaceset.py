# -*- coding: utf-8 -*-
import pickle
import shutil
import struct
from pathlib import Path
import os

from utils.DFLoperate import get_all_dir_names,get_image_paths
from samplelib import Sample
import samplelib.SampleLoader

class PackedFaceset():
    VERSION = 1
    packed_faceset_filename = 'faceset.pak'

    @staticmethod
    def pack(srcFolderPath:Path,dstPakPath:Path,b_delOriImgs:bool=False):

        if not srcFolderPath.is_dir():
            msg=str(srcFolderPath)+":路径不存在或不是文件夹！"
            return [False,msg]
        
        if not dstPakPath.parent.exists():
            msg=str(dstPakPath.parent)+":路径不存在或不是文件夹！"
            return [False,msg]

        if dstPakPath.exists():
            print("路径上存在已知文件！")

        as_person_faceset = False

        dir_names = get_all_dir_names(srcFolderPath)
        if len(dir_names) != 0:
            as_person_faceset = True

        if as_person_faceset:
            image_paths = []

            for dir_name in dir_names:
                image_paths += get_image_paths(srcFolderPath / dir_name)
        else:
            image_paths = get_image_paths(srcFolderPath)

        samples = samplelib.SampleLoader.load_face_samples(image_paths)
        samples_len = len(samples)

        samples_configs = []
        for sample in samples:
            sample_filepath = Path(sample.filename)
            sample.filename = sample_filepath.name

            if as_person_faceset:
                sample.person_name = sample_filepath.parent.name
            samples_configs.append ( sample.get_config() )
        samples_bytes = pickle.dumps(samples_configs, 4)

        of = open(dstPakPath, "wb")
        of.write ( struct.pack ("Q", PackedFaceset.VERSION ) )
        of.write ( struct.pack ("Q", len(samples_bytes) ) )
        of.write ( samples_bytes )

        del samples_bytes   #just free mem
        del samples_configs

        sample_data_table_offset = of.tell()
        of.write ( bytes( 8*(samples_len+1) ) ) #sample data offset table

        data_start_offset = of.tell()
        offsets = []

        for sample in samples:
            try:
                if sample.person_name is not None:
                    sample_path = srcFolderPath / sample.person_name / sample.filename
                else:
                    sample_path = srcFolderPath / sample.filename


                with open(sample_path, "rb") as f:
                   b = f.read()


                offsets.append ( of.tell() - data_start_offset )
                of.write(b)
            except:
                raise Exception(f"error while processing sample {sample_path}")

        offsets.append ( of.tell() )

        of.seek(sample_data_table_offset, 0)
        for offset in offsets:
            of.write ( struct.pack("Q", offset) )
        of.seek(0,2)
        of.close()
        if b_delOriImgs:
            for filename in image_paths:
                    Path(filename).unlink()

            if as_person_faceset:
                    for dir_name in dir_names:
                        dir_path = srcFolderPath / dir_name
                        try:
                            shutil.rmtree(dir_path)
                        except:
                            print("unable to remove:"+str(dir_path))

    @staticmethod
    def unpack(Pak_pathstr,FolderPathStr):
        if not os.path.exists(Pak_pathstr) or not os.path.isfile(Pak_pathstr):
            msg="unpack():pak文件不存在！"
            print(msg)
            return  [False,msg]

        if not os.path.exists(FolderPathStr) or not os.path.isfile(FolderPathStr):
            msg="unpack():目标解压文件夹不存在！"
            print(msg)
            return  [False,msg]

        samples = PackedFaceset.load(Pak_pathstr)

        for sample in samples:
            person_name = sample.person_name
            if person_name is not None:
                person_path = os.path.join(FolderPathStr,person_name)
                if not os.path.exists(person_path) or not os.path.isdir(person_path):
                    try:
                        os.mkdir(person_path)
                    except Exception as ex:
                        msg=person_path+"文件夹无法创建！"
                        return  [False,msg]
                target_filepath = os.path.join(person_path, sample.filename)
            else:
                target_filepath = os.path.join(FolderPathStr,sample.filename)

            with open(target_filepath, "wb") as f:
                f.write( sample.read_raw_file() )
        Path(Pak_pathstr).unlink()
    
    @staticmethod
    def load(pak_path):
        if not os.path.exists(pak_path) or os.path.isfile(pak_path):
            print(pak_path+":该路径不存在！")
            return None


        f = open(pak_path, "rb")
        version, = struct.unpack("Q", f.read(8) )
        if version != PackedFaceset.VERSION:
            print("版本不匹配："+"Pak版本为-"+str(PackedFaceset.VERSION)+",软件版本为-"+str(version))
            return None

        sizeof_samples_bytes, = struct.unpack("Q", f.read(8) )

        samples_configs = pickle.loads ( f.read(sizeof_samples_bytes) )
        samples = []
        for sample_config in samples_configs:
            sample_config = pickle.loads(pickle.dumps (sample_config))
            samples.append ( Sample (**sample_config) )

        offsets = [ struct.unpack("Q", f.read(8) )[0] for _ in range(len(samples)+1) ]
        data_start_offset = f.tell()
        f.close()

        for i, sample in enumerate(samples):
            start_offset, end_offset = offsets[i], offsets[i+1]
            sample.set_filename_offset_size( pak_path, data_start_offset+start_offset, end_offset-start_offset )

        return samples
