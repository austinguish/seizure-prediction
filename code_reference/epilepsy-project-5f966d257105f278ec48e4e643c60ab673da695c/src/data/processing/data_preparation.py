import os
import copy
import argparse
import numpy as np


class DataPreparation():

    def __init__(self, args):

        self.data_path = args.data_path
        self.store_final_dir_path = args.store_final_dir_path
        self.patient = int(args.patient)
        self.preictal_duration = (np.float32(args.preictal_duration) * 60) # cast argument to float32 and then to seconds
        self.discard_data_duration = (np.float32(args.discard_data_duration) * 60)
        self.features_names = args.features_names

        self.loadFeatureFiles()


    def loadFeatureFiles(self):
        '''
          Load all features files. The feature name is taken from the input file name.
          Each feature file has the following structure:
                seizure_1_start_time seizure_1_end_time seizure_2_start_time seizure_2_end_time ...
                segment_1_timestamp segment_1_feature_values.... (several float values)
                segment_2_timestamp segment_2_feature_values.... (several float values)
                segment_3_timestamp segment_3_feature_values.... (several float values)
                .
                .
          , where for each file: in the first row are the seizures timestamps, in the first column are the segments timestemps and the rest are feature values
          After we load the content of each file, we have to concatenate feature_values w.r.t segment, e.g. we have to load
          features values from max_correlation.dat and concatenate the features values of segment_1 with features values
          of segment_1 from univariate.dat, and this happens for N features we use (max_correlation, univariate, SPLV, DSTL, etc..)

          At the end we have a list of segments with all the features.
        '''
        segments = segments_timestamps = seizures_start_time = seizures_end_time = []

        print("Preparing the segments of features: ", self.features_names)
        for i, feature_name in enumerate(self.features_names):
            print("Loading segments of feature: ", feature_name)
            feature_file = self.data_path + '/chb{:02d}'.format(self.patient) + '/' + feature_name + '.dat'

            # load content of feature file and extract the data
            segments_i, segments_timestamps, seizures_start_time, seizures_end_time = self.loadFileContent(feature_file)

            # concatenate features values of corresponding segments
            if (i == 0):
                segments = segments_i
            else:
                segments = [(x + y) for x, y in zip(segments, segments_i)]

            segments_timestamps = segments_timestamps
            seizures_start_time = seizures_start_time
            seizures_end_time = seizures_end_time

            print('"Shape of ', feature_name, ' feature": ', np.shape(segments_i))

        # label the data: interictal, preictal or ictal
        self.annotateSegments(segments, segments_timestamps, seizures_start_time, seizures_end_time)


    def discardOverlappingSeizures(self, seizures_start_time, seizures_end_time):
        prev_end_s = seizures_end_time[0]
        discarded_seizures_start = []
        discarded_seizures_end = []

        copy_seizures_start_time = copy.copy(seizures_start_time)
        copy_seizures_end_time = copy.copy(seizures_end_time)

        for i, (seizure_start, seizure_end) in enumerate(zip(seizures_start_time[1:], seizures_end_time[1:])):

            if((seizure_start - (self.preictal_duration + self.discard_data_duration)) <= prev_end_s):
                copy_seizures_start_time.remove(seizure_start)
                copy_seizures_end_time.remove(seizure_end)
                discarded_seizures_start.append(seizure_start)
                discarded_seizures_end.append(seizure_end)

            prev_end_s = seizure_end

        print(len(discarded_seizures_start), 'seizures are discarded from total of', len(seizures_start_time))

        seizures_start_time = copy.copy(copy_seizures_start_time)
        seizures_end_time = copy.copy(copy_seizures_end_time)

        return seizures_start_time, seizures_end_time, discarded_seizures_start, discarded_seizures_end


    def discardSegmentsOfDiscardedSeizures(self, discarded_seizures_start, discarded_seizures_end, segments_timestamps, segments):

        copy_segments_timestamps = copy.copy(segments_timestamps)
        copy_segments = copy.copy(segments)

        # discard segments that fall within seizures that were discarded previously because of overlapping with other seizures
        for seizure_start, seizure_end in zip(discarded_seizures_start, discarded_seizures_end):
            for i, segment_i_timestamp in enumerate(segments_timestamps):
                if (seizure_start <= segment_i_timestamp and seizure_end >= segment_i_timestamp):
                    copy_segments_timestamps.remove(segment_i_timestamp)
                    copy_segments.remove(copy_segments[i])

        segments_timestamps = copy.copy(copy_segments_timestamps)
        segments = copy.copy(copy_segments)

        return segments_timestamps, segments


    def removeDuplicatedSegments(self, segments_to_clean, check_segments_1, check_segments_2, check_segments_3):
        segments_to_clean = [x for x in segments_to_clean if x not in check_segments_1]
        segments_to_clean = [x for x in segments_to_clean if x not in check_segments_2]
        segments_to_clean = [x for x in segments_to_clean if x not in check_segments_3]

        return segments_to_clean

    def annotateSegments(self, segments, segments_timestamps, seizures_start_time, seizures_end_time):

        # lists to hold labeled segments
        preictal_segments = []
        ictal_segments = []
        interictal_segments = []
        discard_segments = []

        # discard overlapping seizures
        seizures_start_time, seizures_end_time, discarded_seizures_start, discarded_seizures_end = self.discardOverlappingSeizures(seizures_start_time, seizures_end_time)

        # discard segments that fall within discarded seizures
        segments_timestamps, segments = self.discardSegmentsOfDiscardedSeizures(discarded_seizures_start, discarded_seizures_end, segments_timestamps, segments)

        print('segments_timestamps', len(segments_timestamps))
        print('segments', len(segments))

        seizureCount = 1
        # split segments into interictal, preictal and ictal
        for seizure_start, seizure_end in zip(seizures_start_time, seizures_end_time):

            count_temp_ictal_seizures = 0
            count_temp_preictal_seizures = 0
            count_temp_interictal_seizures = 0
            temp_preictal_seizures = []
            temp_ictal_seizures = []
            temp_interictal_seizures = []

            for i, segment_i_timestamp in enumerate(segments_timestamps):
                # ictal filter
                if (seizure_start <= segment_i_timestamp and seizure_end >= segment_i_timestamp):
                    if segments[i] not in ictal_segments:
                        ictal_segments.append(segments[i])
                        temp_ictal_seizures.append(segment_i_timestamp)
                        count_temp_ictal_seizures += 1
                        # print('Ictal segment---->: ', seizure_start, segment_i_timestamp, seizure_end)

                # preictal filter
                elif (segment_i_timestamp > (seizure_start - self.preictal_duration) and segment_i_timestamp < seizure_start):
                    if segments[i] not in preictal_segments:
                        preictal_segments.append(segments[i])
                        temp_preictal_seizures.append(segment_i_timestamp)
                        count_temp_preictal_seizures+=1

                # interictal filter
                elif (segment_i_timestamp <= (seizure_start - (self.preictal_duration + self.discard_data_duration))) or (segment_i_timestamp >= (seizure_end + self.discard_data_duration)):
                    if segments[i] not in interictal_segments:
                        interictal_segments.append(segments[i])
                        temp_interictal_seizures.append(segment_i_timestamp)
                        count_temp_interictal_seizures += 1

                # discard data filter
                elif (segment_i_timestamp >= (seizure_start - (self.preictal_duration + self.discard_data_duration)) and segment_i_timestamp <= seizure_end) \
                        or (segment_i_timestamp <= (seizure_end + self.discard_data_duration) and segment_i_timestamp >= seizure_start):
                    if segments[i] not in discard_segments:
                        discard_segments.append(segments[i])

            # print('Seizure', seizureCount, '(', seizure_start, '-', seizure_start, ')', 'has', temp_preictal_seizures, 'segments')
            print('------------------------------------------------------------------------ Seizure', seizureCount, '--------------------------------------------------------------------------')
            print('     ------------------------------------ Ictal segments ------------------------------------')
            print('     Ictal start: ', seizure_start)
            print('     Ictal segments: ', temp_ictal_seizures)
            print('     Ictal end: ', seizure_end)
            print('     Total segments: ', count_temp_ictal_seizures)
            print('     ------------------------------------ Preictal segments ------------------------------------')
            print('     Preictal start: ', (seizure_start - self.preictal_duration))
            print('     Preictal segments: ', temp_preictal_seizures)
            print('     Preictal end: ', seizure_start)
            print('     Total segments: ', count_temp_preictal_seizures)
            print('\n')

            # print('     ------------------------------------ Interictal segments ------------------------------------')
            # print('     Interictal start before: ', (seizure_start - (self.preictal_duration + self.discard_data_duration)))
            # print('     Interictal start after: ', (seizure_end + self.discard_data_duration))
            # print('     Interictal segments: \n', temp_preictal_seizures)
            # print('     Interictal end: ', seizure_start)
            # print('     Total segments: ', count_temp_preictal_seizures)
            seizureCount += 1

        print('# of seizures: ', len(seizures_start_time))
        print('Total segments: ', np.shape(segments))

        # clean duplicates
        interictal_segments = self.removeDuplicatedSegments(interictal_segments, preictal_segments, ictal_segments, discard_segments)
        discard_segments = self.removeDuplicatedSegments(discard_segments, preictal_segments, ictal_segments, interictal_segments)
        preictal_segments = self.removeDuplicatedSegments(preictal_segments, ictal_segments, interictal_segments, discard_segments)

        print('Ictal segments: ', np.shape(ictal_segments))
        print('Preictal segments: ', np.shape(preictal_segments))
        print('Interictal segments: ', np.shape(interictal_segments))
        print('Discarded segments: ', np.shape(discard_segments))

        # CHECK DUPLICATES
        # print('Preictal and interictal duplicates: ', len(list(x for x in preictal_segments if x in interictal_segments)))
        # print('Ictal and interictal duplicates: ', len(list(x for x in ictal_segments if x in interictal_segments)))
        # print('Discard segments and interictal duplicates: ', len(list(x for x in discard_segments if x in interictal_segments)))
        # print('Discard segments and preictal duplicates: ', len(list(x for x in preictal_segments if x in discard_segments)))
        # print('Discard segments and ictal duplicates: ', len(list(x for x in ictal_segments if x in discard_segments)))
        # print('Preictal and ictal duplicates: ', len(list(x for x in ictal_segments if x in preictal_segments)))
        # print('Preictal and interictal duplicates: ', len(list(x for x in interictal_segments if x in preictal_segments)))

        # directory to store final data
        data_final_dir = self.store_final_dir_path + '/chb{:02d}'.format(self.patient)

        # if any directory does not exist, create
        os.makedirs(data_final_dir, exist_ok=True)

        # set names
        ictal_file = data_final_dir + '/ictal_segments.npy'
        preictal_file = data_final_dir + '/preictal_segments.npy'
        interictal_file = data_final_dir + '/interictal_segments.npy'

        # cast lists to np arrays and save to binary files
        np.save(ictal_file, np.asarray(ictal_segments, dtype=np.float32))
        np.save(preictal_file, np.asarray(preictal_segments, dtype=np.float32))
        np.save(interictal_file, np.asarray(interictal_segments, dtype=np.float32))


    def loadFileContent(self, feature_file):
        '''
        Load the content of given feature file
        :param feature_file: file path (e.g. ./data/processed/chb-mit/features/30-sec/chb06/univariate.dat)
        :return: lists of segments, list of seizures start time, list of seizures end time and list of segments timestamps
        '''

        # load file content in a list (list of rows)
        linesList = [line.rstrip(' ') for line in open(feature_file)]

        # convert string lines to list of floats
        segments = [[np.float32(x) for x in lst.split()] for lst in linesList]

        # seizures start and end times
        seizures_start_time = segments[0][0::2] # some_list[start:stop:step]
        seizures_end_time = segments[0][1::2] # some_list[start:stop:step]

        # store segments timestamps
        segments_timestamps = [item[0] for item in segments[1:]]

        # segments consisted of features values only (without seizures times - first row, and without segments timestamps - first column)
        segments = [item[1:] for item in segments[1:]]

        return segments, segments_timestamps, seizures_start_time, seizures_end_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to load feature files (.dat)")
    parser.add_argument("--store_final_dir_path", help="Path to store final data after filtered into icta, preicta, interictal (np binary files)")
    parser.add_argument("--patient", help="Patient number")
    parser.add_argument("--preictal_duration", help="Preictal duration in minutes")
    parser.add_argument("--discard_data_duration", help="Discard data duration in minutes")
    parser.add_argument('--features_names', nargs='+', help="Select the features to prepare data")
    args = parser.parse_args()

    DataPreparation(args)


if __name__ == '__main__':
    main()
