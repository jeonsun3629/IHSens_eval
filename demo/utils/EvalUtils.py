import numpy as np
import time


def align_with_scale_and_trans(coords3d_pred, coords3d_gt, vis_p=None, vis_gt=None):
    """ Finds the optimal scale and translation to align the point clouds optimally. """
    assert len(coords3d_pred.shape) == 2
    assert len(coords3d_gt.shape) == 2
    assert len(vis_p.shape) == 1
    assert len(vis_gt.shape) == 1

    if vis_p is None:
        vis_p = np.ones_like(coords3d_gt[:, 0])
    vis_p = vis_p == 1.0
    if vis_gt is None:
        vis_gt = np.ones_like(coords3d_gt[:, 0])
    vis_gt = vis_gt == 1.0

    coords3d_gt = np.copy(coords3d_gt)
    coords3d_pred = np.copy(coords3d_pred)

    # find common subset of visible points
    vis_j = np.logical_and(vis_p, vis_gt)
    coords3d_pred_m = coords3d_pred[vis_j, :]
    coords3d_gt_m = coords3d_gt[vis_j, :]

    if (np.sum(vis_j) == 0) or (coords3d_gt_m.shape[0] == 1):
        return False, []

    # calculate means and make pointclouds zero centered
    mean_p_m, mean_gt_m = np.mean(coords3d_pred_m, 0), np.mean(coords3d_gt_m, 0)
    coords3d_pred_m, coords3d_gt_m = coords3d_pred_m - mean_p_m, coords3d_gt_m - mean_gt_m

    # find scale
    lengths_p = np.sqrt(np.sum(np.square(coords3d_pred_m), 0))
    lengths_gt = np.sqrt(np.sum(np.square(coords3d_gt_m), 0))
    scale = np.mean(lengths_gt / lengths_p)

    # apply
    mean_p, mean_gt = np.mean(coords3d_pred[vis_p, :], 0), np.mean(coords3d_gt[vis_gt, :], 0)
    coords3d_p_aligned = ((coords3d_pred - mean_p) * scale) + mean_gt
    return True, coords3d_p_aligned


def match_coords(coords_gt, vis_gt, coords_pred, vis_pred, scale_trans_align=False, max_dist_thresh=None):
    """ Returns the permutation so gt and prediction match best. """
    if max_dist_thresh is None:
        max_dist_thresh = float('inf')

    vis_gt = vis_gt.astype('float32')
    vis_pred = vis_pred.astype('float32')

    num_gt = coords_gt.shape[0]
    num_pred = coords_pred.shape[0]

    dist_list = list()

    for i_gt in range(num_gt):
        for i_p in range(num_pred):
            vis_j = vis_gt[i_gt, :] * vis_pred[i_p, :]
            coords_pred_tmp = coords_pred[i_p, :, :]
            if scale_trans_align:
                suc, coords_pred_tmp = align_with_scale_and_trans(coords_pred_tmp,
                                                                  coords_gt[i_gt, :, :],
                                                                  vis_p=vis_pred[i_p, :],
                                                                  vis_gt=vis_gt[i_gt, :])

            # calculate L2 between all visible pairs
            error_l2 = vis_j * np.sqrt(np.sum(np.square(coords_gt[i_gt, :, :] - coords_pred_tmp), 1))
            error_l2 = np.sum(error_l2) / np.sum(vis_j + 1e-6)

            # calc some penalty, when keypoints are missing
            error_vis_penalty = 1.0 - np.sum(vis_pred[i_p, :]) / np.sum(vis_gt[i_gt, :])  # ranges in 0..1

            dist_list.append((i_gt, i_p, error_l2, error_l2 + error_vis_penalty))

    # sort dist_list
    dist_list = sorted(dist_list, key=lambda x: x[3])

    gt2pred = np.ones((num_gt, ))*-1
    gt_ind_taken = list()
    pred_ind_taken = list()
    for i_gt, i_p, error, _ in dist_list:
        if (i_gt not in gt_ind_taken) and (i_p not in pred_ind_taken):
            if error < max_dist_thresh:
                gt_ind_taken.append(i_gt)
                pred_ind_taken.append(i_p)
                # pred2gt[i_p] = i_gt
                gt2pred[i_gt] = i_p
                continue

    return gt2pred.astype(np.int32)


class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


class PckOverDistanceUtil:
    """ Util class for creating PCK over distance plot.
    """
    def __init__(self, bin_border_values):
        # init empty data storage
        self.bin_border_values = bin_border_values  # border values for binning
        self.bin_distances = [list() for _ in range(len(bin_border_values)+1)]  # list of distances

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        assert len(keypoint_gt.shape) == 2
        assert keypoint_gt.shape[1] == 3
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        # bin values
        ind = np.digitize(keypoint_gt[:, 2], self.bin_border_values)

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.bin_distances[ind[i]].append(euclidean_dist[i])

    def get(self, bin_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.bin_distances[bin_id]) == 0:
            return 0.0

        data = np.array(self.bin_distances[bin_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def create_curve(self, distance_thresh):
        """ Creates pck curves from the data aquired. """
        # calculate values associated with the bins
        values = [0.5*(self.bin_border_values[i] + self.bin_border_values[i+1]) for i in range(len(self.bin_border_values)-1)]

        curve = list()
        for i, v in enumerate(values):
            pck = self.get(i+1, distance_thresh)
            curve.append([v, pck])

        return np.array(curve), [len(x) for x in self.bin_distances]


class MovingAverageCalc:
    """Calculates the moving average of the values passed to it.

    Pass values of a fom to it with feed() and return the averaged value with get().

    Ignores NaN values for its calculation.

    """
    def __init__(self):
        self.value = None
        self.examples = 0

    def feed(self, value):
        if not np.isnan(value):
            if self.examples == 0:
                    self.value = value
            else:
                self.value = (self.examples*self.value + value)/(self.examples+1)

            self.examples += 1

    def get(self):
        if self.value is None:
            return float('nan')
        else:
            if (type(self.value) == float) or (type(self.value) == int):
                return self.value
            else:
                return np.asscalar(self.value)

    def reset(self):
        self.value = None
        self.examples = 0


class NamedTimer(object):
    def __init__(self, msg, divisor=1.0, show_fps=False):
        self.msg = msg
        self.divisor = divisor
        self.show_fps = show_fps

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('[%s: %.3f sec elapsed]' % (self.msg, self.interval))
        if not self.divisor == 1.0:
            self.interval /= float(self.divisor)
            print('[%s: %.3f sec elapsed]' % (self.msg, self.interval))
        if self.show_fps:
            print('[%s: fps = %.3f]' % (self.msg, 1.0/self.interval))