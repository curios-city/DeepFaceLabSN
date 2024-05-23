import traceback
import json
import multiprocessing
import shutil
from pathlib import Path
import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor
from . import Extractor, Sorter
from .Extractor import ExtractSubprocessor


def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    bb_csv = input_path / 'loose_bb_train.csv'
    if not bb_csv.exists():
        raise ValueError('loose_bb_train.csv found. Please ensure it exists.')

    bb_lines = bb_csv.read_text().split('\n')
    bb_lines.pop(0)

    bb_dict = {}
    for line in bb_lines:
        name, l, t, w, h = line.split(',')
        name = name[1:-1]
        l, t, w, h = [ int(x) for x in (l, t, w, h) ]
        bb_dict[name] = (l,t,w, h)


    output_path = input_path.parent / (input_path.name + '_out')

    dir_names = pathex.get_all_dir_names(input_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    data = []
    for dir_name in io.progress_bar_generator(dir_names, "Collecting"):
        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        input_path_image_paths = pathex.get_image_paths(cur_input_path)

        for filename in input_path_image_paths:
            filename_path = Path(filename)

            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue

            l,t,w,h = bb_dict[name]
            if min(w,h) < 128:
                continue

            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False, force_output_path=cur_output_path ) ]

    face_type = FaceType.fromString('full_face')

    io.log_info ('Performing 2nd pass...')
    data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()

    io.log_info ('Performing 3rd pass...')
    ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=None).run()


"""
    import code
    code.interact(local=dict(globals(), **locals()))

    data_len = len(data)
    i = 0
    while i < data_len-1:
        i_name = Path(data[i].filename).parent.name

        sub_data = []

        for j in range (i, data_len):
            j_name = Path(data[j].filename).parent.name
            if i_name == j_name:
                sub_data += [ data[j] ]
            else:
                break
        i = j

        cur_output_path = output_path / i_name

        io.log_info (f"Processing: {str(cur_output_path)}, {i}/{data_len} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)








    for dir_name in dir_names:

        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        input_path_image_paths = pathex.get_image_paths(cur_input_path)
        l = len(input_path_image_paths)
        #if l < 250 or l > 350:
        #    continue

        io.log_info (f"Processing: {str(cur_input_path)} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)


        data = []
        for filename in input_path_image_paths:
            filename_path = Path(filename)

            name = filename_path.parent.name + '/' + filename_path.stem
            if name not in bb_dict:
                continue

            bb = bb_dict[name]
            l,t,w,h = bb
            if min(w,h) < 128:
                continue

            data += [ ExtractSubprocessor.Data(filename=filename,rects=[ (l,t,l+w,t+h) ], landmarks_accurate=False ) ]



        io.log_info ('Performing 2nd pass...')
        data = ExtractSubprocessor (data, 'landmarks', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False).run()

        io.log_info ('Performing 3rd pass...')
        data = ExtractSubprocessor (data, 'final', 256, face_type, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False, final_output_path=cur_output_path).run()


        io.log_info (f"Sorting: {str(cur_output_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')

        import code
        code.interact(local=dict(globals(), **locals()))

        #try:
        #    io.log_info (f"Removing: {str(cur_input_path)} ")
        #    shutil.rmtree(cur_input_path)
        #except:
        #    io.log_info (f"unable to remove: {str(cur_input_path)} ")




def extract_vggface2_dataset(input_dir, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name + '_out')

    dir_names = pathex.get_all_dir_names(input_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)



    for dir_name in dir_names:

        cur_input_path = input_path / dir_name
        cur_output_path = output_path / dir_name

        l = len(pathex.get_image_paths(cur_input_path))
        if l < 250 or l > 350:
            continue

        io.log_info (f"Processing: {str(cur_input_path)} ")

        if not cur_output_path.exists():
            cur_output_path.mkdir(parents=True, exist_ok=True)

        Extractor.main( str(cur_input_path),
              str(cur_output_path),
              detector='s3fd',
              image_size=256,
              face_type='full_face',
              max_faces_from_image=1,
              device_args=device_args )

        io.log_info (f"Sorting: {str(cur_input_path)} ")
        Sorter.main (input_path=str(cur_output_path), sort_by_method='hist')

        try:
            io.log_info (f"Removing: {str(cur_input_path)} ")
            shutil.rmtree(cur_input_path)
        except:
            io.log_info (f"unable to remove: {str(cur_input_path)} ")

"""

#unused in end user workflow
def dev_test_68(input_dir ):
    # process 68 landmarks dataset with .pts files
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name+'_aligned')

    io.log_info(f'Output dir is % {output_path}')

    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) > 0:
            io.input_bool("WARNING !!! \n %s contains files! \n They will be deleted. \n Press enter to continue." % (str(output_path)), False )
            for filename in output_images_paths:
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    images_paths = pathex.get_image_paths(input_path)

    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        filepath = Path(filepath)


        pts_filepath = filepath.parent / (filepath.stem+'.pts')
        if pts_filepath.exists():
            pts = pts_filepath.read_text()
            pts_lines = pts.split('\n')

            lmrk_lines = None
            for pts_line in pts_lines:
                if pts_line == '{':
                    lmrk_lines = []
                elif pts_line == '}':
                    break
                else:
                    if lmrk_lines is not None:
                        lmrk_lines.append (pts_line)

            if lmrk_lines is not None and len(lmrk_lines) == 68:
                try:
                    lmrks = [ np.array ( lmrk_line.strip().split(' ') ).astype(np.float32).tolist() for lmrk_line in lmrk_lines]
                except Exception as e:
                    print(e)
                    print(filepath)
                    continue

                rect = LandmarksProcessor.get_rect_from_landmarks(lmrks)

                output_filepath = output_path / (filepath.stem+'.jpg')

                img = cv2_imread(filepath)
                img = imagelib.normalize_channels(img, 3)
                cv2_imwrite(output_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95] )
                
                raise Exception("unimplemented")
                #DFLJPG.x(output_filepath, face_type=FaceType.toString(FaceType.MARK_ONLY),
                #                                landmarks=lmrks,
                #                                source_filename=filepath.name,
                #                                source_rect=rect,
                #                                source_landmarks=lmrks
                #                    )

    io.log_info("Done.")

#unused in end user workflow
def extract_umd_csv(input_file_csv,
                    face_type='full_face',
                    device_args={} ):

    #extract faces from umdfaces.io dataset csv file with pitch,yaw,roll info.
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)
    face_type = FaceType.fromString(face_type)

    input_file_csv_path = Path(input_file_csv)
    if not input_file_csv_path.exists():
        raise ValueError('input_file_csv not found. Please ensure it exists.')

    input_file_csv_root_path = input_file_csv_path.parent
    output_path = input_file_csv_path.parent / ('aligned_' + input_file_csv_path.name)

    io.log_info("Output dir is %s." % (str(output_path)) )

    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) > 0:
            io.input_bool("WARNING !!! \n %s contains files! \n They will be deleted. \n Press enter to continue." % (str(output_path)), False )
            for filename in output_images_paths:
                Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open( str(input_file_csv_path), 'r') as f:
            csv_file = f.read()
    except Exception as e:
        io.log_err("Unable to open or read file " + str(input_file_csv_path) + ": " + str(e) )
        return

    strings = csv_file.split('\n')
    keys = strings[0].split(',')
    keys_len = len(keys)
    csv_data = []
    for i in range(1, len(strings)):
        values = strings[i].split(',')
        if keys_len != len(values):
            io.log_err("Wrong string in csv file, skipping.")
            continue

        csv_data += [ { keys[n] : values[n] for n in range(keys_len) } ]

    data = []
    for d in csv_data:
        filename = input_file_csv_root_path / d['FILE']


        x,y,w,h = float(d['FACE_X']), float(d['FACE_Y']), float(d['FACE_WIDTH']), float(d['FACE_HEIGHT'])

        data += [ ExtractSubprocessor.Data(filename=filename, rects=[ [x,y,x+w,y+h] ]) ]

    images_found = len(data)
    faces_detected = 0
    if len(data) > 0:
        io.log_info ("Performing 2nd pass from csv file...")
        data = ExtractSubprocessor (data, 'landmarks', multi_gpu=multi_gpu, cpu_only=cpu_only).run()

        io.log_info ('Performing 3rd pass...')
        data = ExtractSubprocessor (data, 'final', face_type, None, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=output_path).run()
        faces_detected += sum([d.faces_detected for d in data])


    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')


    
def dev_test1(input_dir):
    # LaPa dataset
    
    image_size = 1024
    face_type = FaceType.HEAD
    
    input_path = Path(input_dir)
    images_path = input_path / 'images'    
    if not images_path.exists:
        raise ValueError('LaPa dataset: images folder not found.')
    labels_path = input_path / 'labels'    
    if not labels_path.exists:
        raise ValueError('LaPa dataset: labels folder not found.')
    landmarks_path = input_path / 'landmarks'    
    if not landmarks_path.exists:
        raise ValueError('LaPa dataset: landmarks folder not found.')
    
    output_path = input_path / 'out'    
    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) != 0:
            io.input(f"\n WARNING !!! \n {output_path} contains files! \n They will be deleted. \n Press enter to continue.\n")
            for filename in output_images_paths:
                Path(filename).unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    img_paths = pathex.get_image_paths (images_path)
    for filename in img_paths:
        filepath = Path(filename)

        landmark_filepath = landmarks_path / (filepath.stem + '.txt')
        if not landmark_filepath.exists():
            raise ValueError(f'no landmarks for {filepath}')
        
        #img = cv2_imread(filepath)
        
        lm = landmark_filepath.read_text()
        lm = lm.split('\n')
        if int(lm[0]) != 106:
            raise ValueError(f'wrong landmarks format in {landmark_filepath}')
        
        lmrks = []
        for i in range(106):
            x,y = lm[i+1].split(' ')
            x,y = float(x), float(y)
            lmrks.append ( (x,y) )
            
        lmrks = np.array(lmrks)
        
        l,t = np.min(lmrks, 0)
        r,b = np.max(lmrks, 0)
        
        l,t,r,b = ( int(x) for x in (l,t,r,b) )
        
        #for x, y in lmrks:
        #    x,y = int(x), int(y)
        #    cv2.circle(img, (x, y), 1, (0,255,0) , 1, lineType=cv2.LINE_AA)   
         
        #imagelib.draw_rect(img, (l,t,r,b), (0,255,0) )
         
        
        data += [ ExtractSubprocessor.Data(filepath=filepath, rects=[ (l,t,r,b) ]) ]

        #cv2.imshow("", img) 
        #cv2.waitKey(0)
 
    if len(data) > 0:
        device_config = nn.DeviceConfig.BestGPU()
        
        io.log_info ("Performing 2nd pass...")
        data = ExtractSubprocessor (data, 'landmarks', image_size, 95, face_type,  device_config=device_config).run()
        io.log_info ("Performing 3rd pass...")
        data = ExtractSubprocessor (data, 'final', image_size, 95, face_type, final_output_path=output_path, device_config=device_config).run()


        for filename in pathex.get_image_paths (output_path):
            filepath = Path(filename)
            
            
            dflimg = DFLJPG.load(filepath)
            
            src_filename = dflimg.get_source_filename()
            image_to_face_mat = dflimg.get_image_to_face_mat()

            label_filepath = labels_path / ( Path(src_filename).stem + '.png')        
            if not label_filepath.exists():
                raise ValueError(f'{label_filepath} does not exist')
            
            mask = cv2_imread(label_filepath)        
            #mask[mask == 10] = 0 # remove hair
            mask[mask > 0] = 1
            mask = cv2.warpAffine(mask, image_to_face_mat, (image_size, image_size), cv2.INTER_LINEAR)
            mask = cv2.blur(mask, (3,3) )
            
            #cv2.imshow("", (mask*255).astype(np.uint8) ) 
            #cv2.waitKey(0)
            
            dflimg.set_xseg_mask(mask)
            dflimg.save()
        
    
    import code
    code.interact(local=dict(globals(), **locals()))
                    

def dev_resave_pngs(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    images_paths = pathex.get_image_paths(input_path, image_extensions=['.png'], subdirs=True, return_Path_class=True)

    for filepath in io.progress_bar_generator(images_paths,"Processing"):
        cv2_imwrite(filepath, cv2_imread(filepath))


def dev_segmented_trash(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('input_dir not found. Please ensure it exists.')

    output_path = input_path.parent / (input_path.name+'_trash')
    output_path.mkdir(parents=True, exist_ok=True)

    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)

    trash_paths = []
    for filepath in images_paths:
        json_file = filepath.parent / (filepath.stem +'.json')
        if not json_file.exists():
            trash_paths.append(filepath)

    for filepath in trash_paths:

        try:
            filepath.rename ( output_path / filepath.name )
        except:
            io.log_info ('fail to trashing %s' % (src.name) )



def dev_test(input_dir):
    """
    extract FaceSynthetics dataset https://github.com/microsoft/FaceSynthetics
    
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    RIGHT_EYE = 3
    LEFT_EYE = 4
    RIGHT_BROW = 5
    LEFT_BROW = 6
    RIGHT_EAR = 7
    LEFT_EAR = 8
    MOUTH_INTERIOR = 9
    TOP_LIP = 10
    BOTTOM_LIP = 11
    NECK = 12
    HAIR = 13
    BEARD = 14
    CLOTHING = 15
    GLASSES = 16
    HEADWEAR = 17
    FACEWEAR = 18
    IGNORE = 255
    """
    
    
    image_size = 1024
    face_type = FaceType.WHOLE_FACE
    
    input_path = Path(input_dir)
    
    
    
    output_path = input_path.parent / f'{input_path.name}_out'    
    if output_path.exists():
        output_images_paths = pathex.get_image_paths(output_path)
        if len(output_images_paths) != 0:
            io.input(f"\n WARNING !!! \n {output_path} contains files! \n They will be deleted. \n Press enter to continue.\n")
            for filename in output_images_paths:
                Path(filename).unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    for filepath in io.progress_bar_generator(pathex.get_paths(input_path), "Processing"):
        if filepath.suffix == '.txt':
            
            image_filepath = filepath.parent / f'{filepath.name.split("_")[0]}.png'
            if not image_filepath.exists():
                print(f'{image_filepath} does not exist, skipping') 
                
            lmrks = []
            for lmrk_line in filepath.read_text().split('\n'):
                if len(lmrk_line) == 0:
                    continue
                    
                x, y = lmrk_line.split(' ')
                x, y = float(x), float(y)
                
                lmrks.append( (x,y) )
                
            lmrks = np.array(lmrks[:68], np.float32)
            rect = LandmarksProcessor.get_rect_from_landmarks(lmrks)
            data += [ ExtractSubprocessor.Data(filepath=image_filepath, rects=[rect], landmarks=[ lmrks ] ) ]

    if len(data) > 0:
        io.log_info ("Performing 3rd pass...")
        data = ExtractSubprocessor (data, 'final', image_size, 95, face_type, final_output_path=output_path, device_config=nn.DeviceConfig.CPU()).run()

        for filename in io.progress_bar_generator(pathex.get_image_paths (output_path), "Processing"):
            filepath = Path(filename)
            
            dflimg = DFLJPG.load(filepath)
            
            src_filename = dflimg.get_source_filename()
            image_to_face_mat = dflimg.get_image_to_face_mat()
            
            seg_filepath = input_path / ( Path(src_filename).stem + '_seg.png')        
            if not seg_filepath.exists():
                raise ValueError(f'{seg_filepath} does not exist')
            
            seg = cv2_imread(seg_filepath)     
            seg_inds = np.isin(seg, [1,2,3,4,5,6,9,10,11]) 
            seg[~seg_inds] = 0
            seg[seg_inds] = 1
            seg = seg.astype(np.float32)
            seg = cv2.warpAffine(seg, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
            dflimg.set_xseg_mask(seg)
            dflimg.save()
            