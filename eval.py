import numpy as np
import json
import tensorflow as tf
import math

class LaneDetectionEval:
    offset_thresh = (256 / 32) / 2  # Acceptable offset difference for correct detection
    pt_thresh = 0.85  # Threshold for match percentage to consider a detection correct

    @staticmethod
    def interpret_model_output(instance, offsets, anchor_axis, y_anchors, x_anchors, max_instance_count):
        """
        Interprets the raw model outputs into a list of detected lanes, where each lane is represented
        as a list of (x, y) coordinates.
        """
        detected_lanes = []
        for instanceIdx in range(max_instance_count):
            lane_points = []
            for dy in range(y_anchors):
                for dx in range(x_anchors):
                    instance_prob = instance[0, dy, dx, instanceIdx]
                    if instance_prob > 0.5:  # Probability threshold to consider a detection valid
                        offset = offsets[0, dy, dx, 0]
                        gx = anchor_axis[0, dy, dx, 0] + offset  # Adjusting for x offset
                        gy = anchor_axis[0, dy, dx, 1]  # Adjusting for y offset, if applicable
                        lane_points.append((gx, gy))
            if lane_points:
                detected_lanes.append(lane_points)
        return detected_lanes

    @staticmethod
    def evaluate_lane(predictions, ground_truths):
        """
        Evaluates the accuracy, false positives, and false negatives for lane detection.
        """
        total_correct, total_predicted, total_ground_truth = 0, 0, 0

        # Iterate over ground truth lanes and find the best matching predicted lane
        matched_predictions = set()
        for gt_lane in ground_truths:
            gt_lane = np.array(gt_lane)  # Convert to numpy array for easier manipulation
            total_ground_truth += len(gt_lane)
            best_match, best_match_score = None, 0
            
            for idx, pred_lane in enumerate(predictions):
                if idx in matched_predictions:  # Skip lanes that have already been matched
                    continue

                pred_lane = np.array(pred_lane)
                match_score = LaneDetectionEval.calculate_match_score(pred_lane, gt_lane)

                if match_score > best_match_score:
                    best_match, best_match_score = idx, match_score
            
            if best_match is not None:
                matched_predictions.add(best_match)
                total_correct += best_match_score

        total_predicted = sum(len(np.array(pred)) for pred in predictions)

        accuracy = total_correct / total_ground_truth if total_ground_truth > 0 else 0
        fp = len(predictions) - len(matched_predictions)  # Lanes predicted but not matched with any GT
        fn = len(ground_truths) - len(matched_predictions)  # GT lanes not matched with any prediction
        
        return accuracy, fp, fn

    @staticmethod
    def calculate_match_score(pred_lane, gt_lane):
        """
        Calculate the match score based on how many predicted points are within the offset_thresh
        of any point in the ground truth lane.
        """
        if len(pred_lane) == 0 or len(gt_lane) == 0:
            return 0
        match_count = 0
        for k, pred_point in enumerate(pred_lane):
            if np.min(np.sqrt(np.sum((gt_lane - pred_point) ** 2, axis=1))) < LaneDetectionEval.offset_thresh:
                match_count += 1
        return match_count

    @staticmethod
    def convert_ground_truths(ground_truth):
        """
        Converts the ground truth data into the same format as the model output.
        """
        # Example conversion from JSON format to the model output format
        instance_label_dim = 1  # Example value; adjust based on your model
        converted_ground_truth = [[],[],[]]
        for dy in range(32):
            for dx in range(32):
                if ground_truth[dy, dx, 0] == 0:
                    continue # This point is not a lane point
                offset = math.exp(ground_truth[dy, dx, 2]) - 0.0001
                gx = dx * (256 / 32) + offset
                gy = dy * (256 / 32)
                lane_class = int(ground_truth[dy, dx, 3] / 50) - 1
                converted_ground_truth[lane_class].append((gx, gy))

        return converted_ground_truth

    @staticmethod
    def evaluate_predictions(model_output, ground_truths):
        """
        Main evaluation function that takes raw model output and ground truth data,
        processes the model output, and evaluates it against the ground truths.
        """
        # Unpack the model output
        instance, offsets, anchor_axis = model_output
        # Example model output structure; adjust as necessary
        y_anchors, x_anchors, max_instance_count = 32, 32, 5  # Example values; adjust based on your model

        ground_truths = [LaneDetectionEval.convert_ground_truths(gt) for gt in ground_truths]
        predictions = LaneDetectionEval.interpret_model_output(instance, offsets, anchor_axis, y_anchors, x_anchors, max_instance_count)
        # Assuming ground_truths is a list of ground truth lanes for the same structure as predictions
        
        total_accuracy, total_fp, total_fn = 0., 0., 0.
        for gt in ground_truths:
            accuracy, fp, fn = LaneDetectionEval.evaluate_lane(predictions, gt)
            total_accuracy += accuracy
            total_fp += fp
            total_fn += fn

        # Calculate average metrics or total counts as needed
        return {
            "accuracy": total_accuracy / len(ground_truths),
            "fp_rate": total_fp / len(predictions),
            "fn_rate": total_fn / len(ground_truths)
        }

# Example usage
# model_output = (instance_tensor, offsets_tensor, anchor_axis_tensor)  # Tensors from the TensorFlow Lite model
# ground_truths = [...]  # Your ground truth data
# results = LaneDetectionEval.evaluate_predictions(model_output, ground_truths)
