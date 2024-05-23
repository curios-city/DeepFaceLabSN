import pickle
import shutil
import struct
from pathlib import Path

import samplelib.SampleLoader
from core.interact import interact as io
from samplelib import Sample
from core import pathex

import zipfile
import hashlib
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

packed_faceset_filename = "faceset.pak"
packed_faceset_filename_zip = "faceset.zip"
packed_faceset_filename_config = "config.pak"
PACK_EXTENSION = 'pak'



class PackedFaceset():
    VERSION = 1
    
    @staticmethod
    def pack(samples_path, ext=".pak"):
        
        if ext == None:
            ext = io.input_str (f"Archive type for faceset", "pak", ['pak', 'zip'])
        if "pak" == ext:
            samples_dat_path = samples_path / packed_faceset_filename
        elif "zip" == ext:
            samples_dat_path = samples_path / packed_faceset_filename_zip
        else:
            raise NotImplementedError
        
        if samples_dat_path.exists():
            io.log_info(f"{samples_dat_path} : 文件已存在 !")
            io.input("按回车键继续并覆盖.")

        as_person_faceset = False
        dir_names = pathex.get_all_dir_names(samples_path)
        if len(dir_names) != 0:
            as_person_faceset = io.input_bool(f"{len(dir_names)} subdirectories found, process as person faceset?", True)

        if as_person_faceset:
            image_paths = []

            for dir_name in dir_names:
                image_paths += pathex.get_image_paths(samples_path / dir_name)
        else:
            image_paths = pathex.get_image_paths(samples_path)

        samples = samplelib.SampleLoader.load_face_samples(image_paths)
        samples_len = len(samples)

        samples_configs = []
        for sample in io.progress_bar_generator (samples, "Processing"):
            sample_filepath = Path(sample.filename)
            sample.filename = sample_filepath.name

            if as_person_faceset:
                sample.person_name = sample_filepath.parent.name
            samples_configs.append ( sample.get_config() )            

        samples_bytes = pickle.dumps(samples_configs, 4)
        
        if "pak" == ext:
            of = open(samples_dat_path, "wb")
            of.write ( struct.pack ("Q", PackedFaceset.VERSION ) )
            of.write ( struct.pack ("Q", len(samples_bytes) ) )
            of.write ( samples_bytes )
            sample_data_table_offset = of.tell()
            of.write ( bytes( 8*(samples_len+1) ) ) #sample data offset table

            data_start_offset = of.tell()
            offsets = []
        elif "zip" == ext:
            samples_config_path = samples_path / packed_faceset_filename_config
            with samples_config_path.open(mode="wb") as cfg_file:
                cfg_file.write(samples_bytes)
                
            zipObj = zipfile.ZipFile(samples_dat_path, 'w')
            zipObj.write(samples_config_path, packed_faceset_filename_config, compress_type=compression)
            samples_config_path.unlink()
            
        del samples_bytes   #just free mem
        del samples_configs

        

        for sample in io.progress_bar_generator(samples, "Packing"):
            try:
                if sample.person_name is not None:
                    sample_path = samples_path / sample.person_name / sample.filename
                else:
                    sample_path = samples_path / sample.filename

                if "pak" == ext:
                    with open(sample_path, "rb") as f:
                       b = f.read()

                    offsets.append ( of.tell() - data_start_offset )
                    of.write(b)
                elif "zip" == ext:
                    zipObj.write(sample_path, sample.filename, compress_type=compression)
                    
            except:
                raise Exception(f"error while processing sample {sample_path}")

        if "pak" == ext:
            offsets.append ( of.tell() )

            of.seek(sample_data_table_offset, 0)
            for offset in offsets:
                of.write ( struct.pack("Q", offset) )
            of.seek(0,2)
            of.close()
        elif "zip" == ext:
            zipObj.comment = hashlib.md5(str(zipObj.namelist()).encode()).digest()
            zipObj.close()
          
        
        if io.input_bool(f"删除原始文件?", True):
            for filename in io.progress_bar_generator(image_paths, "Deleting files"):
                Path(filename).unlink()

            if as_person_faceset:
                for dir_name in io.progress_bar_generator(dir_names, "Deleting dirs"):
                    dir_path = samples_path / dir_name
                    try:
                        shutil.rmtree(dir_path)
                    except:
                        io.log_info (f"无法移除: {dir_path} ")

    @staticmethod
    def unpack(samples_path):
    
        if (samples_path / packed_faceset_filename).exists():
            samples_dat_path = samples_path / packed_faceset_filename
            samples = PackedFaceset.load(samples_path)
            for sample in io.progress_bar_generator(samples, "Unpacking"):
                person_name = sample.person_name
                if person_name is not None:
                    person_path = samples_path / person_name
                    person_path.mkdir(parents=True, exist_ok=True)

                    target_filepath = person_path / sample.filename
                else:
                    target_filepath = samples_path / sample.filename

                with open(target_filepath, "wb") as f:
                    f.write( sample.read_raw_file() )
                
        elif (samples_path / packed_faceset_filename_zip).exists():
            samples_dat_path = samples_path / packed_faceset_filename_zip
            with zipfile.ZipFile(samples_dat_path, 'r') as zipObj:
                zipObj.extractall(path=samples_path)
            samples_config_path = samples_path / packed_faceset_filename_config
            samples_config_path.unlink()
        else:
            io.log_info(f"{samples_path} : 未找到文件.")

        samples_dat_path.unlink()

    @staticmethod
    def path_contains(samples_path):
        samples_path = Path(samples_path)
        return (samples_path / packed_faceset_filename).exists() or (samples_path / packed_faceset_filename_zip).exists()
    
    @staticmethod
    def load(samples_path, pak_name=None):
        ext = ''
        samples_dat_path = None

        if pak_name is not None:
            if (samples_path / (f"{pak_name}.{PACK_EXTENSION}")).exists():
                ext = 'pak'
                samples_dat_path = samples_path / (f"{pak_name}.{PACK_EXTENSION}")
        else:
            if (samples_path / packed_faceset_filename).exists():
                ext = "pak"
                samples_dat_path = samples_path / packed_faceset_filename

        if (samples_path / packed_faceset_filename_zip).exists() and ext != 'pak':
            samples_dat_path = samples_path / packed_faceset_filename_zip
            samples = []

            with zipfile.ZipFile(samples_dat_path, 'r') as zipObj:
                rebuild_zip = False 
                if zipObj.comment != hashlib.md5(str(zipObj.namelist()).encode()).digest():
                    io.log_err("压缩文件损坏, 正在检查每个文件索引. 解压并重新打包!")
                    rebuild_zip = True
                samples_configs = pickle.loads(zipObj.read(packed_faceset_filename_config))
                for sample_config in samples_configs:
                    sample_config = pickle.loads(pickle.dumps (sample_config))
                    sample = Sample (**sample_config) 
                    if rebuild_zip:
                        if sample.filename not in zipObj.namelist():
                            continue
                    sample.set_filename_offset_size(str(samples_dat_path), -1, -1)
                    samples.append(sample)
            return samples

        if samples_dat_path is not None:
        
            f = open(samples_dat_path, "rb")
            version, = struct.unpack("Q", f.read(8) )
            if version != PackedFaceset.VERSION:
                raise NotImplementedError

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
                sample.set_filename_offset_size( str(samples_dat_path), data_start_offset+start_offset, end_offset-start_offset )

            return samples
        else:
            return None
