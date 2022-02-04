import numpy as np
import cv2
import time
import random

class LabelGenerator():
    def __init__(self):
        self.file_path = None
        self.save_name = None

    def run(self):
        print('Enter parsed file path : ', end='')
        self.file_path = input()
        global_pts_all_frames = np.load(self.file_path)

        print('Enter save file path : ', end='')
        self.save_path = input()

        self.generate_label(global_pts_all_frames, self.save_path)

    def generate_label(self, data, save_path):
        # data : frames x markers x position(xyz)
        img_size = 52
        img_content_size = 48

        img_all_frames = np.empty((1, 52, 52))
        pts_local_all_frames = np.empty((1, 19, 3))

        n_frames = data.shape[0]
        count = 0

        backhand_idx = [16, 17, 18]

        for frame_idx, pts_3d in enumerate(data):
            print("\rProcess... (%d/%d)" % (frame_idx, n_frames), end="")

            # # ---Separate by frame
            # if frame_idx < begin:
            #     continue
            # if frame_idx >= end:
            #     break

            # # ---Step frame
            # if frame_idx % 2 != 0:
            #     continue
            # count += 1

            # Get two vector to determine backhand plane
            v = pts_3d[backhand_idx[1]] - pts_3d[backhand_idx[0]]
            u = pts_3d[backhand_idx[2]] - pts_3d[backhand_idx[0]]

            v = v / np.linalg.norm(v)
            u = u / np.linalg.norm(u)

            n = np.cross(v, u)
            v = np.cross(n, u)

            rotation_matrix = np.concatenate((u.reshape(3, 1),
                                              v.reshape(3, 1),
                                              n.reshape(3, 1)), axis=1)

            # Get projection matrix
            # move origin point into center of backhand markers
            translation = np.mean(np.concatenate((pts_3d[backhand_idx[0]].reshape(3, 1),
                                                  pts_3d[backhand_idx[1]].reshape(3, 1),
                                                  pts_3d[backhand_idx[2]].reshape(3, 1)), axis=1), axis=1)

            transform_matrix = np.concatenate((rotation_matrix, translation.reshape(3, 1)), axis=1)
            transform_matrix = np.concatenate((transform_matrix, np.array([[0, 0, 0, 1]])), axis=0)

            n = pts_3d.shape[0]

            # Homogeneous global marker position vector
            pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))

            # Homogeneous local marker position vector
            pts_3d_local = np.dot(pts_3d_extend, np.linalg.inv(np.transpose(transform_matrix)))

            # Statistic to normalize position(fit to img size)
            min_x = np.min(pts_3d_local[:, 0])
            max_x = np.max(pts_3d_local[:, 0])
            min_y = np.min(pts_3d_local[:, 1])
            max_y = np.max(pts_3d_local[:, 1])

            range_x = max_x - min_x
            range_y = max_y - min_y
            img_center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])

            if range_x > range_y:
                img_scale = img_content_size / range_x
            else:
                img_scale = img_content_size / range_y

            pts_image = (pts_3d_local[:, :2] - img_center) * img_scale + img_size / 2

            # normalize distance
            min_dist = np.min(pts_3d_local[:, 2])
            max_dist = np.max(pts_3d_local[:, 2])
            dist_range = max_dist - min_dist
            pts_norm_dist = (pts_3d_local[:, 2] - min_dist) / dist_range * 0.9 + 0.1

            # generate image array
            img = np.zeros((img_size, img_size))
            n_occluded = max(0, random.randrange(-5, 5))
            idx_occluded = random.sample(range(19), n_occluded)
            for point_idx in range(19):
                if point_idx in idx_occluded:
                    continue
                img[int(pts_image[point_idx][0])][int(pts_image[point_idx][1])] = pts_norm_dist[point_idx]

            # # visualize resize image
            # dst = cv2.resize(img, dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('spatial image', dst)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            img_all_frames = np.vstack([img_all_frames, img.reshape((1, 52, 52))])
            pts_local_all_frames = np.vstack([pts_local_all_frames, pts_3d_local[:, :3].reshape(1, 19, 3)])

        img_all_frames = np.delete(img_all_frames, [0, 0], axis=0)
        pts_local_all_frames = np.delete(pts_local_all_frames, [0, 0], axis=0)
        print('img_all_frames: ', img_all_frames.shape)
        print('pts_local_all_frames shape: ', pts_local_all_frames.shape)
        # np.save('img_' + str(begin) + '_' + str(end), img_all_frames)
        # np.save('pts_local_' + str(begin) + '_' + str(end), pts_local_all_frames)
        np.save(save_path + '_img', img_all_frames)
        np.save(save_path + '_pts_local', pts_local_all_frames)

        print('Save completed.')
        print('Saved img file path : ' + save_path + '_img.npy')
        print('Saved pts file path : ' + save_path + '_pts_local.npy')
