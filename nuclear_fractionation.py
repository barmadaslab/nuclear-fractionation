import os
import sys
import pickle
import cv2
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import skimage.measure
import skimage.morphology
import skimage.filters

from scipy.ndimage import measurements, morphology
from collections import defaultdict

import fileutils
from imgutils import tifffile
from imgutils import transforms

import annotate

import rmp

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, binary_erosion
from scipy.spatial.distance import pdist

def visualize(title, img, func=None):
    plt.close()
    plt.title(title)
    plt.imshow(img)
    if func is not None: func()
    plt.show()

def get_slice_and_offset(contour):
    # Note: slice is returned in numpy format.
    x, y, w, h = cv2.boundingRect(contour)
    return (slice(y, y+h), slice(x, x+w)), np.array([y, x])

def compute_dist(centroid, contour):
    mom = cv2.moments(contour)
    # Centroid is: (cx, cy).
    try: contour_centroid = np.array([mom['m10'] / mom['m00'], mom['m01'] / mom['m00']], dtype=np.uint32)
    except ZeroDivisionError: return 10000 #Arbitrarily chosen large number
    return np.linalg.norm(centroid - contour_centroid)

def extract_improved_cell_centroid(cell_subimg, cell_contour):
    cell_subimg = skimage.filters.median(cell_subimg)  
    # Ensure exterior of cell is set to zero.
    mask = np.zeros_like(cell_subimg)
    cv2.drawContours(mask, [cell_contour], -1, 255, -1)
    cell_subimg[mask == 0] = 0

    #visualize('cell_subimg', cell_subimg)

    # 1D GMM.
    X = cell_subimg[cell_subimg !=0 ].reshape(-1, 1)
    gmm = BayesianGaussianMixture(n_components=10)
    gmm.fit(X)
    gpred_1d = gmm.predict(cell_subimg.reshape(-1, 1)).reshape(cell_subimg.shape).astype(np.uint8)

    # Find maximum intensity label for 1D.
    label_1d = np.argmax(gmm.means_)

    # 3D GMM.
    xvals = np.arange(cell_subimg.shape[0])
    yvals = np.arange(cell_subimg.shape[1])
    xx, yy = np.meshgrid(xvals, yvals)
    S = np.vstack([xx.reshape(-1), yy.reshape(-1), cell_subimg.reshape(-1)]).T
    #gmm = GaussianMixture(n_components=COMP)
    gmm = BayesianGaussianMixture(n_components=3)
    gmm.fit(S)
    gpred_3d = gmm.predict(S).reshape(cell_subimg.shape)

    # Find maximum intensity label for 3D.
    label_3d = np.argmax(gmm.means_[:, 2])

    P = np.zeros_like(cell_subimg)

    P[np.logical_and(gpred_1d == label_1d, gpred_3d == label_3d)] = 1

    # Now compute the centroid.
    M = cv2.moments(P)

    try: 
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    # If unable to extract, choose the center of the bounding rectangle as the centroid.
    except ZeroDivisionError:
        x, y, w, h = cv2.boundingRect(cell_contour)
        cx, cy = (x + w) // 2, (y + h) // 2

    def plt_center():
        plt.plot(cx, cy, 'ro')

    return cx, cy

def find_nuclear_contours(cell_rois, cell_stack, nuclear_stack, um_to_px):
    nuclear_rois = {}
    unknown_rois = {}
    nuclear_centroids = {}
    status_tags = {}
    for ID, cell_roi in cell_rois.items():

        print(f'ID: {ID}')
        # although, 8 seems pyknotic
        # 45, and 41, are instances of multis
        ids = [7]
        ids = [id - 1 for id in ids]
        #if ID not in ids: continue

        cell_contours = cell_roi['contours']

        nuclear_contours = []
        unknown_contours = []
        nuclear_centroids_list = []
        status_tags_list = []

        for tp, cell_contour in enumerate(cell_contours):

            #if tp > 0: break

            # Acquire cellular slice and offset.
            cell_slice, cell_offset = get_slice_and_offset(cell_contour)

            # Use slice to acquire cell subimage.
            cell_subimg = cell_stack[tp][cell_slice]

            tmp = cell_contour - np.array([cell_offset[::-1]])
            cx, cy = extract_improved_cell_centroid(cell_subimg, tmp)

            cell_offset = cell_slice[0].start, cell_slice[1].start

            nuclear_centroids_list.append(np.array([[cx, cy]]) + np.array([cell_offset[::-1]]))

            nuc_subimg = nuclear_stack[tp][cell_slice]
            cell_mask = np.zeros_like(cell_subimg)
            tmp = cell_contour - np.array([cell_offset[::-1]])
            cv2.drawContours(cell_mask, [tmp], -1, 255, -1)
            nuc_subimg[cell_mask == 0] = 0

            # Remove everything outside of the cell.

            nuc_subimg[nuc_subimg < np.percentile(nuc_subimg, 10)] = 0
            nuc_subimg = rmp.subtract(nuc_subimg)
            nuc_subimg[nuc_subimg < np.percentile(nuc_subimg, 30)] = 0
            nuc_subimg = morphology.grey_opening(nuc_subimg, size=(3,3))

            DELTA = um_to_px(10) 
            mask = np.zeros_like(nuc_subimg)
            cv2.circle(mask, (cx, cy), DELTA, 255, -1)
            nuc_subimg[mask == 0] = 0

            median_filtered = skimage.filters.median(nuc_subimg)  
            # Median filtering leads to pixels with 0 value having much larger nonzero values. Ensure the pixels
            # with low values continue to be low.
            median_filtered[nuc_subimg <= np.percentile(nuc_subimg, 5)] = 0
            nuc_subimg = median_filtered

            def plt_center():
                plt.plot(cx, cy, 'r.')

            def fail_func(tag):
                nuclear_contours.append(np.empty(0))
                unknown_contours.append(np.empty(0))
                status_tags_list.append(tag)

            COMP = 5 
            # 1D GMM AREA

            # Fit only to the nonzero portion of the image to attain a sharper fit.
            gmm = BayesianGaussianMixture(n_components=COMP)
            nonzero_nuc_subimg = nuc_subimg[nuc_subimg != 0]

            if nonzero_nuc_subimg.size == 0:
                fail_func('insufficient')
                continue

            try: 
                gmm.fit(nonzero_nuc_subimg.reshape(-1, 1))
            except ValueError:
                fail_func('insufficient-2')
                continue

            #visualize('nuc_subimg', nuc_subimg)

            gpred = gmm.predict(nuc_subimg.reshape(-1, 1)).reshape(nuc_subimg.shape).astype(np.uint8)

            label_of_min = np.argmin(gmm.means_)

            

            # If the minimum's label is not zero, then switch it to zero.
            if label_of_min != 0:
                # Temporarily set zero-label to value guaranteed to exceed total Gaussian count.
                tmp = COMP + 1
                gpred[gpred == 0] = tmp 
                # Ensure the minimum label is zero.
                gpred[gpred == label_of_min] = 0
                # Ensure whatever label was zero is now set to the minimum label.
                gpred[gpred == tmp] = label_of_min
                # Switch the means to reflect the change.
                tmp = np.copy(gmm.means_[0])
                gmm.means_[0] = gmm.means_[label_of_min]
                gmm.means_[label_of_min] = tmp

            # Ensure whatever was background in the nuclear image remains as background. Without this line, it is possible
            # that GMM prediction, which was fed nonzero pixels only, would incorrectly classify background. This is necessary.
            gpred[nuc_subimg == 0] = 0

            #visualize(f'{ID+1}-T{tp+1}gpred', gpred, plt_center)

            # Extract contours and choose the one closest to the SCE.
            P = np.ones_like(nuc_subimg, dtype=np.uint8)
            P[gpred == 0] = 0

            _, contours, _ = cv2.findContours(P, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            compute_roi_centroid_dist = lambda contour: compute_dist(np.array([cx, cy]), contour)
            try: 
                nuclear_contour = min(contours, key=compute_roi_centroid_dist)

            except ValueError: 
                fail_func('unfound')
                continue

            #visualize('gpred', gpred)

            # Upon having found the closest contour, remove rest from image.
            Q = np.zeros_like(nuc_subimg, dtype=np.uint8)
            cv2.drawContours(Q, [nuclear_contour], -1, 255, -1)
            #visualize('Q', Q)
            gpred[Q == 0] = 0

            # Check for pyknotic.
            if gpred[cy, cx] == 0:
                fail_func('pyknotic')
                continue

            # Crop the image.
            x, y, w, h = cv2.boundingRect(nuclear_contour)

            max_y, max_x = gpred.shape
            crop_y, crop_x = max(0, y-1), max(0, x-1)
            cropped_gpred = gpred[crop_y:min(max_y, y+h+1), crop_x:min(max_x, x+w+1)]

            # Update cellular offset and SCE.
            cell_offset += np.array([crop_y, crop_x])
            cx -= crop_x
            cy -= crop_y

            #visualize('gpred', gpred)
            img, label_num = skimage.measure.label(cropped_gpred, return_num=True)

            if label_num == 1:
                pass
                #print(np.where(img == 1))
            img = img.astype(np.uint8)

            # Add one to label number because 0 is not included in count. 0 is background.
            label_num += 1

            def traverse_row_left_to_right(row, inward_links, incidence_counts):
                # Moving left-to-right, find each index where its right-neighbor differs from it.
                diffs = np.where(row[:-1] != row[1:])[0]
                # Make tuples of these labels and their right-neighbors; of the label that changes upon
                # right-traversal, and what it changes to.
                transitions = list(zip(row[diffs], row[diffs+1]))
                # The directed graph:
                labels_within = [0]
                for left, right in transitions:
                    incidence_counts[left, right] += 1
                    if right not in labels_within:
                        inward_links[left].add(right)
                        labels_within.append(right)

            def traverse_rows(img, inward_links, incidence_counts):
                for row in img:
                    traverse_row_left_to_right(row, inward_links, incidence_counts)
                    # Now reverse row and do same.
                    traverse_row_left_to_right(row[::-1], inward_links, incidence_counts)

            def build_graphs(img, label_num):
                inward_links = defaultdict(set)
                incidence_counts = np.zeros((label_num,) * 2)
                traverse_rows(img, inward_links, incidence_counts)
                # Transpose image and to do same for columns.
                traverse_rows(img.T, inward_links, incidence_counts)

                # Turn incidence counts into proportions.
                incidence_proportions = np.array([c / np.sum(c) for c in incidence_counts])
                return inward_links, incidence_proportions

            #visualize('img', img)
            inward_links, incidence_proportions = build_graphs(img, label_num)

            '''
            _labels = sorted(inward_links.keys())
            for l in _labels:
                print(f'{l}: {inward_links[l]}')

            for ix, row in enumerate(incidence_proportions):
                _row = [str(round(x, 2)) for x in row]
                print(f'{ix}: {_row}')
            '''

            # Determine which label is to be taken as the background interface.
            bg_interface_label = np.argmax(incidence_proportions[0])

            # Now will follow a series of rules that use the graph information to properly setup the subsequent portions.
            #visualize('img', img, plt_center)

            # Ensure that any ROI that is fully contained within another label is coalesced into that label.
            for label in range(label_num):
                # If the label has no inward links, then it is fully contained in another label.
                if label not in inward_links:
                    # Find which label it is contained inside.
                    for _label, links in inward_links.items():
                        if label in links:
                            #print(f'{label} not found. It is inside {_label}.')
                            img[img == label] = _label
                            break

            # Ensure everything that is not background or background interface is same label.
            # Multiplying by 2 here because: a) need label to be different than the background interface label,
            # and it helps visualization for the values to be separated.
            signal_label = bg_interface_label * 2 
                        
            img[np.logical_and(img != 0, img != bg_interface_label)] = signal_label

            # Find all signal ROIs. Keep one closest to SCE and eliminate rest.
            _img = np.copy(img)
            _img[img == bg_interface_label] = 0
            _, contours, _ = cv2.findContours(_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            compute_roi_centroid_dist = lambda contour: compute_dist(np.array([cx, cy]), contour)
            #visualize('img', _img)
            try: contour = min(contours, key=compute_roi_centroid_dist)
            except: 
                fail_func('strange-1')
                continue

            # Eliminate any other signal ROIs. 
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            img[np.logical_and(mask == 0, img != bg_interface_label)] = 0

            # Check that SCE is on signal. If not, strange. 
            sce_label = img[cy, cx]
            if sce_label in [0, bg_interface_label]:
                fail_func('strange-2')
                continue

            # Will now compute the centroid of the signal ROI to expand around for attaining the nuclear ROI estimate. 
            # To better estimate the centroid of the nuclear region, will erode the signal ROI. This will remove portions
            # of the ROI which may come from either other nuclei or other non-nuclear regions that make it non-circular. 
            # These portions would skew the centroid estimate.

            # Acquire the contour's bounding rectangle.
            x, y, w, h = cv2.boundingRect(contour)
            min_dim = min(w, h)

            # Will use a 3x3 structuring element to erode. To be conservative, only erode a quarter of the minimum dimension
            # of the bounding rectangle. Erosion will function on both sides of the ROI, so eroding half of the minimum dimension
            # on either side would make the ROI vanish. Eroding half of half will ensure the ROI persists. This is done somewhat
            # heuristically to improve results.

            iter = int(min_dim / 4)

            #visualize('mask', mask)
            mask = binary_erosion(mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]), iterations=iter).astype(np.uint8)
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            # Try to find a contour after erosion. If this fails, continue to use contour extracted before erosion.
            try: contour = contours[0]
            except IndexError: contour = contour
            #visualize('postmask', mask)

            # Compute the centroid of the chosen signal ROI.
            my, mx =  map(int, [np.mean(contour[:, :, 1]), np.mean(contour[:, :, 0])])

            # Now begin drawing concentric circles centered at this centroid with increasing radius. 
            # When the circle overlaps background, either the background interface or otherwise, determine that
            # it has encompassed the full extent of the nucleus that will be measured.
            center = mx, my
            mask = np.zeros_like(img)
            radius = 0
            # 2 is currently used as the signal ROI label.
            non_signal_labels = np.empty(0)
            while non_signal_labels.size == 0:
                radius += 1
                cv2.circle(mask, center, radius, 255, -1)
                non_signal_labels = img[np.logical_and(mask != 0, img != signal_label)]

            # Subtract 1 as it just failed the loop condition.
            radius -= 1

            # After fitting a circle, try to expand into an ellipse.
            # First try one axis, then the other.
            radius1 = radius
            mask = np.zeros_like(img)
            non_signal_labels = np.empty(0)
            while non_signal_labels.size == 0:
                radius1 += 1
                cv2.ellipse(mask, center, (radius1, radius), 0, 0, 360, 255, -1)
                non_signal_labels = img[np.logical_and(mask != 0, img != signal_label)]

            # Subtract 1 as it just failed the loop condition.
            radius1 -= 1
            radius2 = radius
            mask = np.zeros_like(img)
            non_signal_labels = np.empty(0)
            while non_signal_labels.size == 0:
                radius2 += 1
                cv2.ellipse(mask, center, (radius1, radius2), 0, 0, 360, 255, -1)
                non_signal_labels = img[np.logical_and(mask != 0, img != signal_label)]

            # Subtract 1 as it just failed the loop condition.
            radius2 -= 1

            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            nuclear_contour = contours[0]

            if nuclear_contour.size == 0:
                fail_func('unfound')
                continue

            # If SCE is not within nuclear contour, do not choose contour.
            if cv2.pointPolygonTest(nuclear_contour, (cx, cy), False) != 1:
                fail_func('external-SCE')
                continue

            def plt_m():
                plt.plot(mx, my, 'b.')


            # Now find all ROIs that are not background or interface. Choose one closest to SCE.
            #visualize('img', img, plt_m)

            # Now specify the indeterminate region. This will be entire contour, sans background.
            img[img != 0] = 255
            _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            unknown_contour = contours[0]

            #itercount = max(1, int(DELTA / 10))

            # Currently leaving out the containing own centroid portion.
            # If made it this far, believe that we've extracted the SCE containing contour.
            # NOT E: this could have failed. Need to have additional steps to account for failure.
            # Will need to check for own centroid. 

            # If all contained are of either this label or whatever other label is found, then have found a region containing this one.
            #mask1 = np.zeros_like(gpred)
            #mask2 = np.zeros_like(gpred)

            # Draw contour recently found, dilate it, and re-extract.
            #cv2.drawContours(mask1, [contour], -1, 255, -1)
            #cv2.drawContours(mask2, [contour], -1, 255, -1)
            #mask2 = binary_dilation(mask2, structure=np.ones((3, 3))).astype(np.uint8)
            #xor_mask = np.logical_xor(mask1, mask2)

            # Visualize XOR
            #tmp = np.copy(gpred)
            #tmp[np.logical_not(xor_mask)] = 0
            #visualize('gpred', gpred)
            #visualize('xor', tmp)


            #S = np.copy(tmp)
            #visualize('pre', pred2)
            #S = binary_opening(S)
            #S = binary_dilation(S, iterations=itercount).astype(np.uint8)
            #S[cell_mask == 0] = 0
            #_, unknown_contour_estimates, _ = cv2.findContours(S, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 


            unknown_contours.append(unknown_contour + np.array([cell_offset[::-1]]))
            nuclear_contours.append(nuclear_contour + np.array([cell_offset[::-1]]))
            status_tags_list.append('')

        nuclear_rois[ID] = nuclear_contours
        unknown_rois[ID] = unknown_contours
        nuclear_centroids[ID] = nuclear_centroids_list
        status_tags[ID] = status_tags_list

    return nuclear_rois, unknown_rois, nuclear_centroids, status_tags

def pickle_rois(nuclear_rois, unknown_rois, workdir, well):
    path = os.path.join(workdir, 'results', 'nuclear_fractionation', 'nuclear_ROIs', 'rois', well + '.p')
    rois = {ID : {'nuclear' : nuclear_rois[ID], 'unknown' : unknown_rois[ID]} 
            for ID in nuclear_rois.keys()}

    with open(path, 'wb') as f:
        pickle.dump(rois, f)

def load_cell_rois(workdir, well):
    roipath = os.path.join(workdir, 'results', 'rois', well + '.p')
    try:
        with open(roipath, 'rb') as roi_pickle:
            rois = pickle.load(roi_pickle)
    except:
        print('ROIs not found for well: ' + well)
        sys.exit()
    return rois

def make_output_dirs(workdir):
    j = os.path.join
    fileutils.mkdir(j(workdir, 'results'))
    fileutils.mkdir(j(workdir, 'results', 'nuclear_fractionation'))
    fileutils.mkdir(j(workdir, 'results', 'nuclear_fractionation', 'annotated_stacks'))
    fileutils.mkdir(j(workdir, 'results', 'nuclear_fractionation', 'nuclear_ROIs'))
    fileutils.mkdir(j(workdir, 'results', 'nuclear_fractionation', 'nuclear_ROIs', 'rois'))

def run(workdir, config, well):
    # Print out the well being fractionated.
    exp_name = config['experiment']['name']
    print(exp_name + ':: Nuclear fractionation in well: ' + well)

    # Configure the micron-to-pixel transformation for this resolution.
    magnification = config['experiment']['imaging']['magnification']
    microscope = config['experiment']['imaging']['microscope']
    binning = config['experiment']['imaging']['binning']
    um_to_px = lambda microns: transforms.microns_to_pixels(microns, magnification, microscope, binning)

    # Make list of channels with primary channel first.
    primary_channel = config['experiment']['imaging']['primary_channel']
    other_channels = [c for c in config['experiment']['imaging']['fluors'] if c != primary_channel]
    channels = [primary_channel] + other_channels

    # Ensure presence of either DAPI or UV channels. These are typical nuclear markers used in lab.
    if 'DAPI' not in channels and 'UV' not in channels:
        raise ValueError('Neither DAPI nor UV found in channels. No nuclear marker is present.')
    elif 'DAPI' in channels and 'UV' in channels:
        raise ValueError('Both DAPI and UV found in channels. Two nuclear markers are present. Which one to use?')
        sys.exit()

    #Make necessary output directories.
    make_output_dirs(workdir)

    #Set path variables
    stack_path = os.path.join(workdir, 'processed_imgs', 'stacked')
    primary_channel_path = os.path.join(stack_path, primary_channel)
    if 'DAPI' in channels:
        nuclear_channel_path = os.path.join(stack_path, 'DAPI')
    elif 'UV' in channels:
        nuclear_channel_path = os.path.join(stack_path, 'UV')

    cell_rois = load_cell_rois(workdir, well)

    cell_stack = tifffile.imread(os.path.join(primary_channel_path, well + '.tif'))

    #Load nuclei stack
    nuclear_stack = tifffile.imread(os.path.join(nuclear_channel_path, well + '.tif'))

    #Find nuclei corresponding to cells
    nuclear_rois, unknown_rois, nuclear_centroids, status_tags = find_nuclear_contours(cell_rois, cell_stack, np.copy(nuclear_stack), um_to_px)

    # Output rois and annotated stack.
    if nuclear_rois is not None:
        pickle_rois(nuclear_rois, unknown_rois, workdir, well)
        filtered_cell_rois = {ID : cell_rois[ID]['contours'] for ID, _ in nuclear_rois.items()}
        annotated = annotate.annotate_fractionation(well, nuclear_stack, filtered_cell_rois, unknown_rois, nuclear_rois, nuclear_centroids, status_tags)
        tifffile.imsave(os.path.join(workdir, 'results', 'nuclear_fractionation', 'annotated_stacks', well + 'nuc.tif'), annotated)

if __name__ == '__main__':
    # FIX TMP ID FILTER
    import yaml
    from mfileutils.makeconfig import mfile_to_config
    import shutil
    #workdir = r'K:\HA101_120\HA116minitest'
    workdir = r'K:\HA101_120\HA116miniB'
    workdir = r'B:\INC files\INC25'
    shutil.rmtree(os.path.join(workdir, 'results', 'nuclear_fractionation'), ignore_errors=True)
    config = mfile_to_config(workdir)
    run(workdir, config, 'A01')
    #for i in range(1, 11):
        #run(workdir, config, f'A0{i}')
