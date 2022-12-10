import ast
import pandas as pd
import random
import matplotlib.pyplot as plt

import config as cfg


def load_data(file_path):
    data_frame = pd.read_csv(file_path)
    print("[INFO] Number of pedestrians in the dataset : ", len(data_frame))
    for col_name in ['bounding_boxes', 'frame_numbers', 'moving slow', 'stopped', 'handwave', 'look', 'clear path',
                     'crossing', 'moving fast', 'looking', 'standing', 'slow down', 'nod', 'speed up']:
        data_frame[col_name] = data_frame[col_name].apply(ast.literal_eval)
    return data_frame


def explore_data(data_frame):
    count = 0
    for i, row in data_frame.iterrows():
        count += len(row['frame_numbers'])
    print("Number of Pedestrian-Frames: %d" % count)

    # Let's take a more in-depth look at that first row:
    print(data_frame.iloc[0])
    print(data_frame.iloc[0]['frame_numbers'])

    # calculate the number of features
    columns = []
    for col in data_frame.columns:
        if col != 'video_id' and col != 'ped_ind' and col != 'frame_numbers' and col != 'cross_overall' and col != 'crossing':
            columns.append(col)
    print("[INFO] Features : ", columns)
    print("[INFO] Number of features : ", len(columns))

    # generate a random data id to plot
    idx = random.randint(0, len(data_frame))
    print("[INFO] For Video : {} ; Pedestrian {}".format(data_frame.iloc[idx]['video_id'],
                                                         data_frame.iloc[idx]['ped_ind']))

    # plot the data for all frames for each feature and
    # compare it to the crossing trend
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    img_w, img_h = cfg.IMG_W, cfg.IMG_H
    for col, ax in zip(columns, axs.ravel()):
        # handle bounding boxes data separately
        if col == 'bounding_boxes':
            bboxes_xcoords = []
            for bbox in data_frame.iloc[idx][col]:
                # normalize the bounding box x-coordinate for each frame
                bboxes_xcoords.append(bbox[0] / img_w)
            ax.plot(data_frame.iloc[idx]['frame_numbers'], bboxes_xcoords)
        else:
            ax.plot(data_frame.iloc[idx]['frame_numbers'], data_frame.iloc[idx][col])
        ax.plot(data_frame.iloc[idx]['frame_numbers'], data_frame.iloc[idx]['crossing'], linestyle='dashed')
        ax.set_title('Plot : {} vs crossing'.format(col))
        ax.set_xlabel("Frames ==>")
        ax.legend([col, 'crossing'])
    fig.show()


def parse_bbox_x(elements, img_w):
    '''
    Function to parse and normalize the bounding box
    x-coordinate value for each pedestrian data point
    Input: Pedestrian bounding box data for all frames in the video
    Ouput: Normalized bounding box x-coordinate data
    '''
    output = []
    for i in range(len(elements)):
        output.append(
            round(int(float(elements[i][0]) + float(elements[i][2]) * 0.5) / img_w, 5))
    return output


def parse_bbox_y(elements, img_h):
    '''
    Function to parse and normalize the bounding box
    y-coordinate value for each pedestrian data point
    Input: Pedestrian bounding box data for all frames in the video
    Ouput: Normalized bounding box y-coordinate data
    '''
    output = []
    for i in range(len(elements)):
        output.append(
            round(int(float(elements[i][1])) / img_h, 5))
    return output


def arrange_data_per_frame(data, window_size, feature_set=cfg.FEATURE_SET_FULL, img_w=cfg.IMG_W, img_h=cfg.IMG_H):
    data_per_frame = []
    # loop through each data row, every row contains
    # frame-by-frame data of a pedestrian which is to be unraveled
    for i, row in data.iterrows():
        # segregate features of the dataset that model needs to be trained on
        attr_list = feature_set

        # get the frames and outputs of crossing for all the frames
        frames = row['frame_numbers']
        outputs = row['crossing']

        # empty container to hold data for each feature and populate it with data feature-wise
        # data for bounding box feature should be handled separately
        attributes = {}
        for attr in attr_list[:-1]:
            attributes[attr] = row[attr]
        attributes['bounding_boxes'] = parse_bbox_x(row['bounding_boxes'], img_w)

        # loop through every frame
        for j in range(len(frames)):
            if j >= window_size:
                # if sufficient frames have been initialized, curate data of previous frames
                # data for each frame is arranged above explained format
                data_pt = []
                data_pt.append(row['video_id'])
                data_pt.append(row['ped_ind'])
                data_pt.append(frames[j])
                data_pt.append(outputs[j])

                # gather data for previous frame for each feature, bounding boxes to be handled separately
                for k in range(window_size, 0, -1):
                    for attr in attr_list:
                        if attr == 'bounding_boxes':
                            data_pt.append(attributes[attr][j - k])
                        else:
                            data_pt.append(int(attributes[attr][j - k]))
                data_pt[3] = int(data_pt[3])
                # populate the output list for with data of each frame
                data_per_frame.append(data_pt)
    return data_per_frame
