import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time , pytz, datetime
import requests
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#Face recognition
import face_recognition
import argparse
import imutils
import pickle

#Import Detec Coconut
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


flags.DEFINE_string('who', 'unknown','who is in the image')
flags.DEFINE_string('encodings', './encodings.pickle','Embedding for Face recognition')
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('thief_detector', False, 'enable to detect thief has taken coconut ')
flags.DEFINE_boolean('face_rec', False, 'enable face recognition')
detection_method = 'cnn'

def send_alert(message):
    url = 'https://notify-api.line.me/api/notify'
    token = '4AX9bOiqMD5cFRHmw2wDDnZwiFrTSuQ2LvyvUF0zTUP'
    headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
    fmt = '%Y-%m-%d %H:%M:%S'
    bkk = pytz.timezone('Asia/Bangkok')
    dt = datetime.datetime.now(bkk)
    msg = str(dt.strftime(fmt))+': '+message
    r = requests.post(url, headers=headers, data = {'message':msg})


def load_model_thief(path):
    with tf.compat.v2.io.gfile.GFile(path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def main(_argv):

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for Person detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    
    #loading encodings for Face recognition
    print("[INFO] loading encodings...")
    data = pickle.loads(open(FLAGS.encodings, "rb").read())

    # Load model for Coconut Thief detector
    if FLAGS.thief_detector:
        # path to the frozen graph:
        PATH_TO_FROZEN_GRAPH = 'model_data/frozen_inference_graph.pb'
        PATH_TO_LABEL_MAP = 'model_data/label_map.pbtxt'
        NUM_CLASSES = 1
        #Generate graph
        detection_graph = load_model_thief(PATH_TO_FROZEN_GRAPH)
        label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        

    # load tflite model if flag is set for Person detector
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    NextFrame = 0
    PreviousFrame = 0
    PreviouseNames = str()
    track_dict = dict()
    track_thief_dict = dict()
    unknown = 0
    yada = 0
    a_mom = 0
    # while video is running

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # update tracks
        for track in tracker.tracks:
            print('Track is confirmed:',track.is_confirmed(),'since: ',track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[2]), int(bbox[1])), color, -1)
            if int(track.track_id) in track_dict.keys():
                cv2.putText(frame, class_name + "-" + str(track.track_id)+"-"+track_dict[track.track_id],(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                if int(track.track_id) in track_thief_dict.keys() and track_dict[track.track_id] == 'Unknown':
                    cv2.putText(frame, class_name + "-" + str(track.track_id)+"-"+track_dict[track.track_id]+"-Thief",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (0,0,0),2)
            else:
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                if int(track.track_id) in track_thief_dict.keys():
                    cv2.putText(frame, class_name + "-" + str(track.track_id)+"-Thief",(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (0,0,0),2)

            #Face Recognition
            if int(track.track_id) not in track_dict.keys() and class_name == 'person' and FLAGS.face_rec:
            # if class_name == 'person' :
                #Frame recognition
                frame2 = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                #Face Recognition
                rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame2, width=750)
                r = frame2.shape[1] / float(rgb.shape[1])
                # detect the (x, y)-coordinates of the bounding boxes
                # coอrresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb,model=detection_method)
                encodings = face_recognition.face_encodings(rgb, boxes)
                names2 = []

                print('encondings length:',len(encodings))

                for encoding in encodings:
                    matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.5)
                    print(face_recognition.face_distance(data["encodings"],encoding))
                    name = "Unknown"
                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)
                    names2.append(name)

                for ((top, right, bottom, left), name) in zip(boxes, names2):
                    top = int(top * r)
                    right = int(right * r)
                    bottom = int(bottom * r)
                    left = int(left * r)
                    cv2.rectangle(frame2, (left, top), (right, bottom),(0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame2, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                    print('Name:',names2)
                    if str(names2) != '[]':
                        track_dict[track.track_id] = names2[0]
                        print('Found Person '+str(track.track_id)+' is '+str(track_dict[track.track_id]))
                        send_alert('Found Person '+str(track.track_id)+' is '+str(track_dict[track.track_id]))
                        if track_dict[track.track_id] == 'Yada2':
                            yada = yada+1
                        elif track_dict[track.track_id] == 'A':
                            a_mom = a_mom +1
                        elif track_dict[track.track_id] == 'Unknown':
                            unknown = unknown +1
                    print('accurancy yada/unknown/a:',yada,unknown,a_mom)

            # Coconut Thief detector
            if FLAGS.thief_detector  and track.track_id not in track_thief_dict.keys():
            # if FLAGS.thief_detector :
                frame2 = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame2, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes2 = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores2 = detection_graph.get_tensor_by_name('detection_scores:0')
                classes2 = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections2 = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                boxes2, scores2, classes2, num_detections2 = tf.compat.v1.Session(graph=detection_graph).run(
                    [boxes2, scores2, classes2, num_detections2],
                    feed_dict={image_tensor: image_np_expanded})
                
                
                box3_detect = vis_util.return_coordinates(
                    frame2,
                    np.squeeze(boxes2),
                    np.squeeze(classes2).astype(np.int32),
                    np.squeeze(scores2),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    )
                if len(box3_detect) > 1:
                    send_alert('Found '+str(track_dict[track.track_id])+' took coconut at '+str(int((frame_num/25)/60))+' min '+str(frame_num/25)+' sec')
                    track_thief_dict[track.track_id] = True
                    #Draw coconut box
                    for box3 in box3_detect: 
                        print('ymin3, xmin3, ymax3, xmax3,score : ',box3[0], box3[2], box3[1], box3[3],box3[4])
                        # coordinates_list.append([ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)])
                        # # print("type: ",str(box3),'Num detect: '+str(num_detections2.shape))
                        cv2.rectangle(frame2, (box3[2], box3[0]), (box3[3], box3[1]),(0, 255, 0), 2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name,  (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            if class_name == 'person' and PreviousFrame==0 :
                print('Video: '+str(FLAGS.video)+'\nFound Person '+str(track.track_id))
                send_alert('Video: '+str(FLAGS.video)+'\nFound Person '+str(track.track_id))
                PreviousFrame = 1

        if len(tracker.tracks) == 0 and PreviousFrame==1 :
            if yada+unknown+a_mom == 0 :
                accurancy_score = 0
            elif FLAGS.who == 'yada' :
                accurancy_score = yada/(yada+unknown+a_mom)
            elif FLAGS.who == 'a':
                accurancy_score = a_mom/(yada+unknown+a_mom)
            elif FLAGS.who == 'unknown':
                accurancy_score = unknown/(yada+unknown+a_mom)

            track_dict.pop(track.track_id, None)
            print('Person '+str(track.track_id)+' disappeared :'+FLAGS.who+': accurancy/yada/unknown/a: '+str(accurancy_score)+'/'+str(yada)+'/'+str(unknown)+'/'+str(a_mom)+' Video: '+str(FLAGS.video))
            # send_alert('Person '+str(track.track_id)+' disappeared :'+FLAGS.who+': accurancy/yada/unknown/a: '+str(accurancy_score)+'/'+str(yada)+'/'+str(unknown)+'/'+str(a_mom)+' Video: '+str(FLAGS.video))
            send_alert('Person '+str(track.track_id)+' disappeared.\nThis video is '+FLAGS.who+'\nFrom VDO: '+str(FLAGS.video))
            PreviousFrame = 0

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
